import numpy as np
import scipy.io
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, image
from nilearn.input_data import NiftiLabelsMasker
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import root_mean_squared_error

# Load Model 1 predictions and targets
data_model_1 = np.load('model_1_predictions.npz')
predictions_1 = data_model_1['predictions']
targets_1 = data_model_1['targets']

# Load Model 2 predictions and targets
data_model_2 = np.load('model_2_predictions.npz')
predictions_2 = data_model_2['predictions']
targets_2 = data_model_2['targets']

# Load Model 3 predictions and targets
data_model_3 = np.load('model_3_predictions.npz')
predictions_3 = data_model_3['predictions']
targets_3 = data_model_3['targets']

# Load Model 4 predictions and targets
data_model_4 = np.load('model_4_predictions.npz')
predictions_4 = data_model_4['predictions']
targets_4 = data_model_4['targets']

# Load fMRI data
fmri_mat_file = 'Wehbe_data/doi_10_5061_dryad_gt413__v20150225/subject_2.mat'
fmri_data = scipy.io.loadmat(fmri_mat_file)
meta = fmri_data['meta'][0, 0]
col_to_roi = meta['colToROInum'][0]  # Array mapping voxels to ROI numbers

def analyse_and_visualise_differences(predictions, targets):

    name = 'Words to fMRI' if predictions is predictions_1 else 'Linguistic Features to fMRI'    

    affine = meta['matrix']

    # Define original voxel dimensions
    voxel_shape = (51, 60, 49)
    n_samples = predictions.shape[0]  # Should be 259

    # Reshape predictions and targets to 4D arrays: (51, 60, 49, n_samples)
    predictions_reshaped = predictions.reshape(n_samples, *voxel_shape).transpose(1, 2, 3, 0)
    targets_reshaped = targets.reshape(n_samples, *voxel_shape).transpose(1, 2, 3, 0)

    # Create 4D NIfTI images
    predictions_img = nib.Nifti1Image(predictions_reshaped, affine)
    targets_img = nib.Nifti1Image(targets_reshaped, affine)

    # Load the AAL atlas
    atlas_dataset = datasets.fetch_atlas_aal()
    atlas_img = nib.load(atlas_dataset.maps)
    labels = atlas_dataset.labels

    # Initialize the masker
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, labels=labels)
    masker.fit()


    # Extract data within ROIs (shape: n_samples x n_rois)
    predictions_data = masker.transform(predictions_img)
    targets_data = masker.transform(targets_img)

    # Define ROI numbers of interest
    roi_numbers = [10, 12, 14, 62, 64, 80] # Adjust the numbering so that it matches the indices in SPM12

    # Map ROI numbers to indices (assuming ROI numbers start at 1)
    roi_indices = roi_numbers

    #roi_indices = [num - 1 for num in roi_numbers if num <= len(labels)]

    # Initialize lists to store labels and average differences
    roi_labels = []
    average_diffs = []
    significance_flags = []

    # Verify mapping
    if not roi_indices:
        print("No valid ROI numbers found in the atlas.")
    else:
        # Compute average differences and perform t-test for specified ROIs
        for roi_num, idx in zip(roi_numbers, roi_indices):
            pred_values = predictions_data[:, idx]  # Shape: (259,)
            targ_values = targets_data[:, idx]      # Shape: (259,)
            diff = pred_values - targ_values
            abs_diff = np.abs(diff)
            mean_diff = np.mean(abs_diff) 
            t_stat, p_value = stats.ttest_1samp(diff, 0)
            significance = p_value < 0.05
            label = labels[idx]
            
            roi_labels.append(label)
            average_diffs.append(mean_diff)
            significance_flags.append(significance)

    # Create bar colors based on significance
    colors = ['red']  #['red' if sig else 'blue' for sig in significance_flags]

    # Create the bar chart with thinner bars
    plt.figure(figsize=(12, 6))
    plt.bar(roi_labels, average_diffs, color=colors, width=0.6)
    plt.xlabel('ROI')
    plt.ylabel('Average Difference per Brain Region')
    #plt.title(f'Average Absolute Difference per ROI: {name}')
    plt.ylim(0, 0.2)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


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


analyse_and_visualise_differences(predictions_1, targets_1)

analyse_and_visualise_differences(predictions_2, targets_2)

# For Model 1
mse_1 = compute_mse(predictions_1, targets_1)
cos_sim_1 = compute_cosine_similarity(predictions_1, targets_1)

print("Model 1: Words to fMRI data - Evaluation Metrics:")
print(f"MSE: {mse_1:.4f}")
print(f"Average Cosine Similarity: {cos_sim_1:.4f}")


# For Model 2
mse_2 = compute_mse(predictions_2, targets_2)
cos_sim_2 = compute_cosine_similarity(predictions_2, targets_2)

print("\nModel 2: Linguistic Features to fMRI data - Evaluation Metrics:")
print(f"MSE: {mse_2:.4f}")
print(f"Average Cosine Similarity: {cos_sim_2:.4f}")

# For Model 3
mse_3 = compute_mse(predictions_3, targets_3)
cos_sim_3 = compute_cosine_similarity(predictions_3, targets_3)

print("Model 3: fMRI data to Word embeddings - Evaluation Metrics:")
print(f"MSE: {mse_3:.4f}")
print(f"Average Cosine Similarity: {cos_sim_3:.4f}")

# For Model 4
mse_4 = compute_mse(predictions_4, targets_4)
cos_sim_4 = compute_cosine_similarity(predictions_4, targets_4)

print("\Model 4: fMRI data to Linguistic features - Evaluation Metrics:")
print(f"MSE: {mse_4:.4f}")
print(f"Average Cosine Similarity: {cos_sim_4:.4f}")

""""
# Visualise the brain activity for a sample
# Define original voxel dimensions
voxel_shape = (51, 60, 49)

# Select a sample to visualize (e.g., first sample)

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
plt.show()


# Plot Actual vs Predicted fMRI Activity
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plotting.plot_stat_map(
    targets_img,
    display_mode='ortho',
    title='Actual fMRI Activity',
    cut_coords=(0, 0, 0),
    axes=axes[0],
    colorbar=True,
    threshold=2  # Adjust threshold as needed
)

plotting.plot_stat_map(
    predictions_img,
    display_mode='ortho',
    title='Predicted fMRI Activity',
    cut_coords=(0, 0, 0),
    axes=axes[1],
    colorbar=True,
    threshold=2  # Adjust threshold as needed
)

plt.tight_layout()
plt.show()"""