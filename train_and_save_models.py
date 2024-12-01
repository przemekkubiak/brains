import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
from scipy.stats import pearsonr

# Load the data
d = np.load('aligned_data_subject_2.npz', allow_pickle=True)
aligned_data = d['aligned_data']
story_features = pd.read_csv('features.csv')

# Drop the first column which contains row numbers
story_features = story_features.drop(columns=['Unnamed: 0'])

# Ensure the second column (index 0 after dropping) is treated as strings
story_features.iloc[:, 0] = story_features.iloc[:, 0].astype(str)

# Ensure the words in story_features are in lowercase and stripped of leading/trailing spaces
story_features.iloc[:, 0] = story_features.iloc[:, 0].str.lower().str.strip()

# Convert numerical values in the dataframe to float types, excluding the first column
story_features.iloc[:, 1:] = story_features.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Group by the word column and compute the mean of the features for each word
grouped_features = story_features.groupby(story_features.columns[0]).mean().reset_index()

# Initialize an empty list to store the averaged linguistic features
averaged_features = []

# Iterate over each dictionary in aligned_data
for entry in aligned_data:
    words = [word.lower().strip() for word in entry['words']]
    
    # Filter grouped_features to get the rows corresponding to these words
    filtered_features = grouped_features[grouped_features.iloc[:, 0].isin(words)]
    
    # Compute the average of the linguistic features for these words
    avg_features = filtered_features.iloc[:, 1:].mean(axis=0)
    
    # Fill NaN values with zeros
    avg_features = avg_features.fillna(0)
    
    # Append the averaged features to the new dataset
    averaged_features.append(avg_features)

# Convert the list of averaged features to a DataFrame
average_features_df = pd.DataFrame(averaged_features)


### STAT TRAINING THE MODEL ###

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data for Model 1: Words to fMRI data
class WordToFmriDataset(Dataset):
    def __init__(self, aligned_data):
        self.inputs = []
        self.targets = []
        for item in aligned_data:
            words = item['words']
            fmri = item['fmri_data']
            text = ' '.join(words)
            encoded_input = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                embedding = bert_model(**encoded_input).last_hidden_state.mean(dim=1).squeeze()
            self.inputs.append(embedding.numpy())
            self.targets.append(fmri.flatten())
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )

# Prepare data for Model 2: Linguistic features to fMRI data
class FeaturesToFmriDataset(Dataset):
    def __init__(self, average_features_df, aligned_data):
        self.inputs = average_features_df.values.astype(np.float32)
        self.targets = []
        for item in aligned_data:
            fmri = item['fmri_data']
            self.targets.append(fmri.flatten())
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )
    
# Prepare data for Model 3: fMRI data to Words (embeddings)
class FmriToWordDataset(Dataset):
    def __init__(self, aligned_data):
        self.inputs = []
        self.targets = []
        for item in aligned_data:
            fmri = item['fmri_data']
            words = item['words']
            text = ' '.join(words)
            encoded_input = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                embedding = bert_model(**encoded_input).last_hidden_state.mean(dim=1).squeeze()
            self.inputs.append(fmri.flatten())
            self.targets.append(embedding.numpy())
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )
    
# Prepare data for Model 4: fMRI data to Linguistic features
class FmriToFeaturesDataset(Dataset):
    def __init__(self, average_features_df, aligned_data):
        self.inputs = []
        self.targets = average_features_df.values.astype(np.float32)
        for item in aligned_data:
            fmri = item['fmri_data']
            self.inputs.append(fmri.flatten())
        self.inputs = np.array(self.inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# Define the model architecture
class EncodingDecodingModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(EncodingDecodingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function with loss printing and model saving
def train_model(model, dataloader, model_name, epochs=10):
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), f'{model_name}.pth')

# Evaluation function
def evaluate_model(model, dataloader):
    model.to(device)
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    mse = np.mean((all_predictions - all_targets) ** 2)
    # Compute voxel-wise Pearson correlation or feature-wise correlation
    correlations = []
    for i in range(all_targets.shape[1]):
        corr, _ = pearsonr(all_targets[:, i], all_predictions[:, i])
        correlations.append(corr)
    mean_correlation = np.nanmean(correlations)
    return mse, mean_correlation

# Split dataset into training and testing sets
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

# Model 1: Words to fMRI data
word_to_fmri_dataset = WordToFmriDataset(aligned_data)
word_train_dataset, word_test_dataset = split_dataset(word_to_fmri_dataset)
word_train_loader = DataLoader(word_train_dataset, batch_size=16, shuffle=True)
word_test_loader = DataLoader(word_test_dataset, batch_size=16, shuffle=False)
word_model = EncodingDecodingModel(input_size=768, output_size=51 * 60 * 49)
train_model(word_model, word_train_loader, model_name='word_to_fmri_model')
word_mse, word_corr = evaluate_model(word_model, word_test_loader)
print(f"Word to fMRI Model - MSE: {word_mse:.4f}, Mean Correlation: {word_corr:.4f}")

# Model 2: Linguistic features to fMRI data
features_to_fmri_dataset = FeaturesToFmriDataset(average_features_df, aligned_data)
features_train_dataset, features_test_dataset = split_dataset(features_to_fmri_dataset)
features_train_loader = DataLoader(features_train_dataset, batch_size=16, shuffle=True)
features_test_loader = DataLoader(features_test_dataset, batch_size=16, shuffle=False)
features_model = EncodingDecodingModel(input_size=195, output_size=51 * 60 * 49)
train_model(features_model, features_train_loader, model_name='features_to_fmri_model')
features_mse, features_corr = evaluate_model(features_model, features_test_loader)
print(f"Features to fMRI Model - MSE: {features_mse:.4f}, Mean Correlation: {features_corr:.4f}")

# Model 3: fMRI data to Word embeddings
fmri_to_word_dataset = FmriToWordDataset(aligned_data)
fmri_word_train_dataset, fmri_word_test_dataset = split_dataset(fmri_to_word_dataset)
fmri_word_train_loader = DataLoader(fmri_word_train_dataset, batch_size=16, shuffle=True)
fmri_word_test_loader = DataLoader(fmri_word_test_dataset, batch_size=16, shuffle=False)
fmri_to_word_model = EncodingDecodingModel(input_size=51 * 60 * 49, output_size=768)
train_model(fmri_to_word_model, fmri_word_train_loader, model_name='fmri_to_word_model')
fmri_word_mse, fmri_word_corr = evaluate_model(fmri_to_word_model, fmri_word_test_loader)
print(f"fMRI to Word Embeddings Model - MSE: {fmri_word_mse:.4f}, Mean Correlation: {fmri_word_corr:.4f}")

# Model 4: fMRI data to Linguistic features
fmri_to_features_dataset = FmriToFeaturesDataset(average_features_df, aligned_data)
fmri_feat_train_dataset, fmri_feat_test_dataset = split_dataset(fmri_to_features_dataset)
fmri_feat_train_loader = DataLoader(fmri_feat_train_dataset, batch_size=16, shuffle=True)
fmri_feat_test_loader = DataLoader(fmri_feat_test_dataset, batch_size=16, shuffle=False)
fmri_to_features_model = EncodingDecodingModel(input_size=51 * 60 * 49, output_size=195)
train_model(fmri_to_features_model, fmri_feat_train_loader, model_name='fmri_to_features_model')
fmri_feat_mse, fmri_feat_corr = evaluate_model(fmri_to_features_model, fmri_feat_test_loader)
print(f"fMRI to Linguistic Features Model - MSE: {fmri_feat_mse:.4f}, Mean Correlation: {fmri_feat_corr:.4f}")


