import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import plotting
from sklearn.preprocessing import StandardScaler
import scipy.io

### LOAD DATA FOR EVALUATION ###

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

### EVALUATE THE MODELS ###

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load the saved models
# Model 1: Words to fMRI data
word_to_fmri_model = EncodingDecodingModel(input_size=768, output_size=51*60*49)
word_to_fmri_model.load_state_dict(torch.load('word_to_fmri_model.pth', map_location=device))
word_to_fmri_model.to(device)
word_to_fmri_model.eval()

# Model 2: Linguistic features to fMRI data
features_to_fmri_model = EncodingDecodingModel(input_size=195, output_size=51*60*49)
features_to_fmri_model.load_state_dict(torch.load('features_to_fmri_model.pth', map_location=device))
features_to_fmri_model.to(device)

# Model 3: fMRI data to Word embeddings
fmri_to_word_model = EncodingDecodingModel(input_size=51 * 60 * 49, output_size=768)
fmri_to_word_model.load_state_dict(torch.load('fmri_to_word_model.pth', map_location=device))
fmri_to_word_model.to(device)
fmri_to_word_model.eval()

# Model 4: fMRI data to Linguistic features
fmri_to_features_model = EncodingDecodingModel(input_size=51 * 60 * 49, output_size=195)
fmri_to_features_model.load_state_dict(torch.load('fmri_to_features_model.pth', map_location=device))
fmri_to_features_model.to(device)
fmri_to_features_model.eval()


# Prepare data for Model 1: Words to fMRI data
class WordFmriDataset(Dataset):
    def __init__(self, aligned_data):
        self.inputs = []
        self.targets = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()
        with torch.no_grad():
            for item in aligned_data:
                words = item['words']
                fmri = item['fmri_data']
                text = ' '.join(words)
                encoded_input = tokenizer(text, return_tensors='pt')
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
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# Prepare data for Model 2: Linguistic features to fMRI data

class FeaturesFmriDataset(Dataset):
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
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# Prepare data for Model 3: fMRI data to Word embeddings
class FmriToWordDataset(Dataset):
    def __init__(self, aligned_data):
        self.inputs = []
        self.targets = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()
        with torch.no_grad():
            for item in aligned_data:
                fmri = item['fmri_data']
                words = item['words']
                text = ' '.join(words)
                encoded_input = tokenizer(text, return_tensors='pt')
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
            torch.tensor(self.targets[idx], dtype=torch.float32)
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
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# Split datasets to get test sets
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

# For Model 1
word_dataset = WordFmriDataset(aligned_data)
_, word_fmri_test_dataset = split_dataset(word_dataset)
word_fmri_test_loader = DataLoader(word_fmri_test_dataset, batch_size=16, shuffle=False)

# For Model 2
features_to_fmri_dataset = FeaturesFmriDataset(average_features_df, aligned_data)
_, features_fmri_test_dataset = split_dataset(features_to_fmri_dataset)
features_fmri_test_loader = DataLoader(features_fmri_test_dataset, batch_size=16, shuffle=False)

# For Model 3
fmri_to_word_dataset = FmriToWordDataset(aligned_data)
_, fmri_word_test_dataset = split_dataset(fmri_to_word_dataset)
fmri_word_test_loader = DataLoader(fmri_word_test_dataset, batch_size=16, shuffle=False)

#ground_truth_words_list = []

#for batch in fmri_word_test_loader:
#    inputs, targets, ground_truth_words = batch
#    # Now you can use 'ground_truth_words' for comparison
#    ground_truth_words_list.append(ground_truth_words)

# For Model 4
fmri_to_features_dataset = FmriToFeaturesDataset(average_features_df, aligned_data)
_, fmri_feat_test_dataset = split_dataset(fmri_to_features_dataset)
fmri_feat_test_loader = DataLoader(fmri_feat_test_dataset, batch_size=16, shuffle=False)

# Function to get model outputs
def get_model_outputs(model, dataloader):
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
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    return all_predictions, all_targets

# Compute Representational Similarity Analysis (RSA)
def compute_rdm(data):
    # data: Num_samples x Num_features
    distance_matrix = pdist(data, metric='euclidean')
    rdm = squareform(distance_matrix)
    return rdm

def compute_rsa(predictions, targets):
    rdm_pred = compute_rdm(predictions)
    rdm_target = compute_rdm(targets)
    triu_indices = np.triu_indices_from(rdm_pred, k=1)
    rdm_pred_flat = rdm_pred[triu_indices]
    rdm_target_flat = rdm_target[triu_indices]
    rsa_score, _ = spearmanr(rdm_pred_flat, rdm_target_flat)
    return rsa_score

# Compute Cosine Similarity
def compute_cosine_similarity(predictions, targets):
    similarities = cosine_similarity(predictions, targets)
    diag_similarities = similarities.diagonal()
    average_similarity = np.mean(diag_similarities)
    return average_similarity

# Compute Mean Squared Error (MSE)
def compute_mse(predictions, targets):
    mse = root_mean_squared_error(targets, predictions)
    return mse

# Evaluate Model 1: Words to fMRI data
predictions_1, targets_1 = get_model_outputs(word_to_fmri_model, word_fmri_test_loader)

# Evaluate Model 2: Linguistic features to fmri data
predictions_2, targets_2 = get_model_outputs(features_to_fmri_model, features_fmri_test_loader)

# Evaluate Model 3: fMRI data to Word embeddings
predictions_3, targets_3 = get_model_outputs(fmri_to_word_model, fmri_word_test_loader)

# Evaluate Model 4: fMRI data to Linguistic features
predictions_4, targets_4 = get_model_outputs(fmri_to_features_model, fmri_feat_test_loader)

# Load the .mat file for the subject
mat_file = f'Wehbe_data/doi_10_5061_dryad_gt413__v20150225/subject_2.mat'
mat_data = scipy.io.loadmat(mat_file)

# Extract the fMRI data and meta data from the .mat file
fmri_data = mat_data['data']  # Adjust the key as needed
meta = mat_data['meta'][0][0]  # Adjust the key as needed

affine = meta['matrix']

# Define original voxel dimensions
voxel_shape = (51, 60, 49)

"""# Select a sample to visualize (e.g., first sample)
sample_index = 50
pred_sample = predictions_1[sample_index].reshape(voxel_shape)
target_sample = targets_1[sample_index].reshape(voxel_shape)

# Exclude zero values by setting them to NaN
pred_sample[pred_sample == 0] = np.nan
target_sample[target_sample == 0] = np.nan

# Create NIfTI images
pred_img = nib.Nifti1Image(pred_sample, affine)
target_img = nib.Nifti1Image(target_sample, affine)

# Plot Actual vs Predicted fMRI Activity
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plotting.plot_stat_map(
    target_img,
    display_mode='ortho',
    title='Actual fMRI Activity',
    cut_coords=(0, 0, 0),
    axes=axes[0],
    colorbar=True,
    threshold=2  # Adjust threshold as needed
)

plotting.plot_stat_map(
    pred_img,
    display_mode='ortho',
    title='Predicted fMRI Activity',
    cut_coords=(0, 0, 0),
    axes=axes[1],
    colorbar=True,
    threshold=2  # Adjust threshold as needed
)

plt.tight_layout()
plt.show()"""

# For Model 1
mse_1 = compute_mse(predictions_1, targets_1)
cos_sim_1 = compute_cosine_similarity(predictions_1, targets_1)
rsa_score_1 = compute_rsa(predictions_1, targets_1)

print("Model 1: Words to fMRI data - Evaluation Metrics:")
print(f"MSE: {mse_1:.4f}")
print(f"Average Cosine Similarity: {cos_sim_1:.4f}")
print(f"RSA Score (Spearman correlation): {rsa_score_1:.4f}")

# Save Model 1 predictions and targets to .npz file
np.savez('model_1_predictions.npz', predictions=predictions_1, targets=targets_1)

# For Model 2
mse_2 = compute_mse(predictions_2, targets_2)
cos_sim_2 = compute_cosine_similarity(predictions_2, targets_2)
rsa_score_2 = compute_rsa(predictions_2, targets_2)

print("\Model 2: Linguistic features to fmri data - Evaluation Metrics:")
print(f"MSE: {mse_2:.4f}")
print(f"Average Cosine Similarity: {cos_sim_2:.4f}")
print(f"RSA Score (Spearman correlation): {rsa_score_2:.4f}")

# Save Model 2 predictions and targets to .npz file
np.savez('model_2_predictions.npz', predictions=predictions_2, targets=targets_2)

# For Model 3
mse_3 = compute_mse(predictions_3, targets_3)
cos_sim_3 = compute_cosine_similarity(predictions_3, targets_3)
rsa_score_3 = compute_rsa(predictions_3, targets_3)

print("Model 3: fMRI data to Word embeddings - Evaluation Metrics:")
print(f"MSE: {mse_3:.4f}")
print(f"Average Cosine Similarity: {cos_sim_3:.4f}")
print(f"RSA Score (Spearman correlation): {rsa_score_3:.4f}")

# For Model 4
mse_4 = compute_mse(predictions_4, targets_4)
cos_sim_4 = compute_cosine_similarity(predictions_4, targets_4)
rsa_score_4 = compute_rsa(predictions_4, targets_4)

print("\Model 4: fMRI data to Linguistic features - Evaluation Metrics:")
print(f"MSE: {mse_4:.4f}")
print(f"Average Cosine Similarity: {cos_sim_4:.4f}")
print(f"RSA Score (Spearman correlation): {rsa_score_4:.4f}")

quit()

def visualize_rdms(pred_rdm, actual_rdm, title_pred='Predicted RDM', title_actual='Actual RDM'):
    """
    Visualize Predicted and Actual Representational Dissimilarity Matrices side by side.
    
    Args:
        pred_rdm (numpy.ndarray): Predicted RDM.
        actual_rdm (numpy.ndarray): Actual RDM.
        title_pred (str): Title for the Predicted RDM plot.
        title_actual (str): Title for the Actual RDM plot.
    """

    scaler = StandardScaler()
    pred_rdms = [scaler.fit_transform(rdm) for rdm in pred_rdms]
    actual_rdms = [scaler.fit_transform(rdm) for rdm in actual_rdms]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(pred_rdm, cmap='viridis', ax=axes[0])
    axes[0].set_title(title_pred)
    axes[0].set_xlabel('Stimuli')
    axes[0].set_ylabel('Stimuli')
    
    sns.heatmap(actual_rdm, cmap='viridis', ax=axes[1])
    axes[1].set_title(title_actual)
    axes[1].set_xlabel('Stimuli')
    axes[1].set_ylabel('Stimuli')
    
    plt.tight_layout()
    plt.show()


def visualize_rdms(pred_rdm, actual_rdm, title_pred='Predicted RDM', title_actual='Actual RDM', model_name='Model'):
    """
    Visualize Predicted and Actual Representational Dissimilarity Matrices side by side.
    
    Args:
        pred_rdm (numpy.ndarray): Predicted RDM.
        actual_rdm (numpy.ndarray): Actual RDM.
        title_pred (str): Title for the Predicted RDM plot.
        title_actual (str): Title for the Actual RDM plot.
        model_name (str): Name of the model for labeling purposes.
    """
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Flatten the RDMs to fit scaler (standard scaler expects 2D input)
    pred_rdm_flat = pred_rdm.flatten().reshape(-1, 1)
    actual_rdm_flat = actual_rdm.flatten().reshape(-1, 1)
    
    # Fit and transform the RDMs
    pred_rdm_scaled = scaler.fit_transform(pred_rdm_flat).reshape(pred_rdm.shape)
    actual_rdm_scaled = scaler.fit_transform(actual_rdm_flat).reshape(actual_rdm.shape)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Predicted RDM
    sns.heatmap(pred_rdm_scaled, cmap='viridis', ax=axes[0])
    axes[0].set_title(f'{model_name} - {title_pred}')
    axes[0].set_xlabel('Stimuli')
    axes[0].set_ylabel('Stimuli')
    
    # Plot Actual RDM
    sns.heatmap(actual_rdm_scaled, cmap='viridis', ax=axes[1])
    axes[1].set_title(f'{model_name} - {title_actual}')
    axes[1].set_xlabel('Stimuli')
    axes[1].set_ylabel('Stimuli')
    
    plt.tight_layout()
    plt.show()



# List of predictions and targets for the four models
predictions = [predictions_1, predictions_2, predictions_3, predictions_4]
targets = [targets_1, targets_2, targets_3, targets_4]

# List of predictions and targets for the four models
predictions = [predictions_1, predictions_2, predictions_3, predictions_4]
targets = [targets_1, targets_2, targets_3, targets_4]
model_names = [
    'Model 1: Words to fMRI',
    'Model 2: Linguistic Features to fMRI',
    'Model 3: fMRI to Words',
    'Model 4: fMRI to Linguistic Features'
]


# Compute RSA Scores and RDMs for all models
rsa_scores = []
pred_rdms = []
actual_rdms = []

for i in range(4):
    rsa = compute_rsa(predictions[i], targets[i])
    rsa_scores.append(rsa)
    print(f'{model_names[i]} RSA Score: {rsa:.4f}')
    
    pred_rdm = compute_rdm(predictions[i])
    actual_rdm = compute_rdm(targets[i])
    
    pred_rdms.append(pred_rdm)
    actual_rdms.append(actual_rdm)
    
    # Visualize RDMs for the current model
    visualize_rdms(
        pred_rdm,
        actual_rdm,
        title_pred='Predicted RDM',
        title_actual='Actual RDM',
        model_name=model_names[i]
    )



def compute_sample_cosine_similarity(predictions, targets):
    """
    Computes cosine similarity for each sample between predictions and targets.

    Args:
        predictions (np.ndarray): Predicted linguistic features (Num_samples x Num_features).
        targets (np.ndarray): Actual linguistic features (Num_samples x Num_features).

    Returns:
        np.ndarray: Cosine similarity scores for each sample.
    """
    similarities = cosine_similarity(predictions, targets)
    # Extract the diagonal elements which represent similarity of each sample with itself
    sample_cosine_sim = similarities.diagonal()
    return sample_cosine_sim

# Example usage:
# predictions_2, targets_2 = get_model_outputs(features_to_fmri_model, features_test_loader)
# For Models 1 and 2, ensure you're using the correct predictions and targets
sample_cosine_sim_1 = compute_sample_cosine_similarity(predictions_1, targets_1)
sample_cosine_sim_2 = compute_sample_cosine_similarity(predictions_2, targets_2)

def compute_feature_cosine_similarity(predictions, targets):
    """
    Computes cosine similarity for each linguistic feature across all samples.

    Args:
        predictions (np.ndarray): Predicted linguistic features (Num_samples x Num_features).
        targets (np.ndarray): Actual linguistic features (Num_samples x Num_features).

    Returns:
        np.ndarray: Cosine similarity scores for each feature.
    """
    num_features = predictions.shape[1]
    feature_cosine_sim = []
    for feature_idx in range(num_features):
        pred_vector = predictions[:, feature_idx].reshape(1, -1)
        target_vector = targets[:, feature_idx].reshape(1, -1)
        cos_sim = cosine_similarity(pred_vector, target_vector)[0][0]
        feature_cosine_sim.append(cos_sim)
    return np.array(feature_cosine_sim)

# Example usage:
feature_cosine_sim_1 = compute_feature_cosine_similarity(predictions_1, targets_1)
feature_cosine_sim_2 = compute_feature_cosine_similarity(predictions_2, targets_2)

def plot_cosine_similarity_histogram(sample_cosine_sim, model_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(sample_cosine_sim, bins=50, kde=True, color='skyblue')
    plt.title(f'Cosine Similarity Distribution for {model_name}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()

# Example usage:
plot_cosine_similarity_histogram(sample_cosine_sim_1, 'Model 1: Words to fMRI')
plot_cosine_similarity_histogram(sample_cosine_sim_2, 'Model 2: Linguistic Features to fMRI')

def plot_feature_cosine_similarity(feature_cosine_sim, model_name, num_top=20):
    # Sort features by similarity
    sorted_indices = np.argsort(feature_cosine_sim)[::-1]
    sorted_similarities = feature_cosine_sim[sorted_indices]
    
    # Select top and bottom features
    top_features = sorted_similarities[:num_top]
    bottom_features = sorted_similarities[-num_top:]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_top), top_features, color='green', label='Top Similarities')
    plt.bar(range(num_top), bottom_features, color='red', label='Bottom Similarities')
    plt.title(f'Feature-wise Cosine Similarity for {model_name}')
    plt.xlabel('Feature Index (sorted)')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.show()

# Example usage:
plot_feature_cosine_similarity(feature_cosine_sim_1, 'Model 1: Words to fMRI')
plot_feature_cosine_similarity(feature_cosine_sim_2, 'Model 2: Linguistic Features to fMRI')

