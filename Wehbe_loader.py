import scipy.io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import rsatoolbox
import nibabel as nib
from nilearn import plotting, masking
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img
from nilearn.signal import clean
from nilearn.masking import compute_epi_mask
import pandas as pd

# Load feature data
feature_mat_file = 'Wehbe_data/doi_10_5061_dryad_gt413__v20150225/story_features.mat'
feature_data = scipy.io.loadmat(feature_mat_file)
features_array = feature_data['features']

mat = scipy.io.loadmat(feature_mat_file)
mat = {k:v for k, v in mat.items() if k[0] != '_'}


l = []

for i in range (0, 10):
    for item in features_array['names'][0][i]:
        print(len(item))
        if type(item)==np.ndarray:
            for entry in item:
                l.append(entry[0])
        else:
            l.append(item)


rows = []


# Loop through each word (0 to 5175)
for word_index in range(5176):
    concatenated_values = []
    # Loop through each feature array (0 to 9)
    for feature_index in range(10):
        # Get the values for the current word and feature
        values = feature_data['features'][0]['values'][feature_index][word_index]
        # Append the values to the concatenated list
        concatenated_values.extend(values)
    # Append the concatenated values to the rows list
    rows.append(concatenated_values)

# Load fMRI data
fmri_mat_file = 'Wehbe_data/doi_10_5061_dryad_gt413__v20150225/subject_1.mat'
fmri_data = scipy.io.loadmat(fmri_mat_file)
fmri_time_series = fmri_data['data']  # Shape: (n_fmri_timepoints, n_voxels)
meta = fmri_data['meta'][0, 0]
feature_values = mat['features']  # Shape: (n_words, n_features)
time = fmri_data['time']
words = fmri_data['words']

words_list = []

for i in words[0]:
    words_list.append(i[0][0][0][0])


# Convert the list to a DataFrame
data_df = pd.DataFrame(rows, columns=l)
data_df.insert(0, 'word', words_list)

data_df.to_csv('features.csv')
"""
# Check shapes
# print(f"Feature values shape: {feature_values.shape}")        # Expected: (5176, n_features)
# print(f"fMRI time series shape: {fmri_time_series.shape}")    # Expected: (~1294, n_voxels)

# Compute the number of fMRI timepoints and words
n_fmri_timepoints = fmri_time_series.shape[0]
n_words = feature_values.shape[0]
if __name__ == '__main__':
    # Identify ROI numbers relevant for language processing
    # Example ROI numbers (these should be replaced with actual ROI numbers from the AAL atlas)
    language_rois = [7, 8, 9, 10, 11, 12, 67, 68, 69, 70, 85, 86]

    # Extract voxel indices corresponding to these ROIs
    voxel_indices = [i for i, roi_num in enumerate(meta['colToROInum'][0]) if roi_num in language_rois]

    # Determine the midline of the brain in the x-dimension
    x_coords = [meta['colToCoord'][i][0] for i in voxel_indices]
    midline_x = (min(x_coords) + max(x_coords)) / 2
    # print(f"Midline x-coordinate: {midline_x}")

    # Filter voxel indices to include only those in the left hemisphere
    left_hemisphere_voxel_indices = [i for i in voxel_indices if meta['colToCoord'][i][0] > midline_x]

    # Debug: Print the left hemisphere voxel indices
    # print("Left Hemisphere Voxel Indices:")
    # print(left_hemisphere_voxel_indices)

    # Get the spatial dimensions
    dimx = int(meta['dimx'][0][0])
    dimy = int(meta['dimy'][0][0])
    dimz = int(meta['dimz'][0][0])

    # Filter the fMRI data to include only these voxels
    filtered_fmri_data = fmri_time_series[:, left_hemisphere_voxel_indices]

    
    # List of subjects
    subjects = [f'subject_{i}' for i in range(1, 9)]  # subject_1 to subject_8

    # Function to preprocess and save data for a single subject
    def preprocess_and_save(subject):
        # Load the .mat file for the subject
        mat_file = f'Wehbe_data/doi_10_5061_dryad_gt413__v20150225/{subject}.mat'
        mat_data = scipy.io.loadmat(mat_file)

        # Extract the fMRI data and meta data from the .mat file
        fmri_data = mat_data['data']  # Adjust the key as needed
        meta = mat_data['meta'][0][0]  # Adjust the key as needed

        language_rois = [7, 8, 9, 10, 11, 12, 67, 68, 69, 70, 85, 86]

        # Extract voxel indices corresponding to these ROIs
        voxel_indices = [i for i, roi_num in enumerate(meta['colToROInum'][0]) if roi_num in language_rois]

        # Determine the midline of the brain in the x-dimension
        x_coords = [meta['colToCoord'][i][0] for i in voxel_indices]
        midline_x = (min(x_coords) + max(x_coords)) / 2
        # print(f"Midline x-coordinate: {midline_x}")

        # Filter voxel indices to include only those in the left hemisphere
        left_hemisphere_voxel_indices = [i for i in voxel_indices if meta['colToCoord'][i][0] > midline_x]

        # Get the spatial dimensions
        dimx = int(meta['dimx'][0][0])
        dimy = int(meta['dimy'][0][0])
        dimz = int(meta['dimz'][0][0])

        # Filter the fMRI data to include only the left hemisphere voxels
        filtered_fmri_data = fmri_data[:, left_hemisphere_voxel_indices]

        # Create an empty 4D volume (time, x, y, z)
        volume = np.zeros((filtered_fmri_data.shape[0], dimx, dimy, dimz))

        # Fill the 4D volume with the fMRI data
        for idx, voxel_index in enumerate(left_hemisphere_voxel_indices):
            x, y, z = meta['colToCoord'][voxel_index]
            x, y, z = int(x) - 1, int(y) - 1, int(z) - 1  # Adjust for 0-based indexing
            volume[:, x, y, z] = filtered_fmri_data[:, idx]

        # Use the affine matrix from the meta data
        affine = meta['matrix']
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(volume, affine)

        # Create a mask from the fMRI data
        mask_img = masking.compute_epi_mask(nifti_img)

        # Perform smoothing, detrending, and masking
        nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=6, detrend=True, standardize=True)
        nifti_masker.fit(nifti_img)
        preprocessed_img = nifti_masker.transform(nifti_img)

        # Convert the preprocessed data back to 4D
        preprocessed_volume = nifti_masker.inverse_transform(preprocessed_img).get_fdata()

        # Create a new NIfTI image for the preprocessed volume
        preprocessed_volume_img = nib.Nifti1Image(preprocessed_volume, affine)

        # Save the preprocessed volume
        output_file = f'{subject}_preprocessed_volume.nii'
        nib.save(preprocessed_volume_img, output_file)
        print(f'Saved preprocessed data for {subject} to {output_file}')


    # Loop through all subjects and preprocess their data
    for subject in subjects:
        preprocess_and_save(subject)"""

"""if __name__ == '__main__':
    # Compute the average activation over time for the selected voxels
    # average_activation = filtered_fmri_data.mean(axis=1)
    # average_activation = fmri_time_series.mean(axis=1)

    # Plot the average activation over time
    # plt.figure(figsize=(12, 6))
    # plt.plot(np.arange(n_fmri_timepoints) * 2, average_activation)  # Assuming TR = 2 seconds
    # plt.xlabel('Time (s)')
    # plt.ylabel('Average Activation')
    # plt.title('Average Activation in Language-Related Voxels Over Time')
    # plt.show()

    # Create an empty 4D volume (time, x, y, z)
    volume = np.zeros((filtered_fmri_data.shape[0], dimx, dimy, dimz))

    # Fill the 4D volume with the fMRI data
    for idx, voxel_index in enumerate(left_hemisphere_voxel_indices):
        x, y, z = meta['colToCoord'][voxel_index]
        x, y, z = int(x) - 1, int(y) - 1, int(z) - 1  # Adjust for 0-based indexing
        volume[:, x, y, z] = filtered_fmri_data[:, idx]

    # Use the affine matrix from the meta data
    affine = meta['matrix']

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)

    # Create a mask from the fMRI data
    mask_img = masking.compute_epi_mask(nifti_img)

    # Perform smoothing, detrending, and masking
    nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=6, detrend=True, standardize=True)
    nifti_masker.fit(nifti_img)
    preprocessed_img = nifti_masker.transform(nifti_img)

    # Convert the preprocessed data back to 4D
    preprocessed_volume = nifti_masker.inverse_transform(preprocessed_img).get_fdata()

    nifti_image = nib.Nifti1Image(preprocessed_volume, affine)
    nib.save(nifti_image, 'preprocessed_volume.nii')

    # Choose a time point in the middle of the experiment
    timepoint = 600

    # Create an empty 3D volume for the selected time point
    volume_timepoint = np.zeros((dimx, dimy, dimz))

    # Insert the voxel values into the 3D volume
    for idx, voxel_index in enumerate(left_hemisphere_voxel_indices):
        x, y, z = meta['colToCoord'][voxel_index]
        x, y, z = int(x) - 1, int(y) - 1, int(z) - 1  # Adjust for 0-based indexing
        volume_timepoint[x, y, z] = preprocessed_volume[timepoint, x, y, z]

    # Create a NIfTI image for the selected time point
    nifti_img_timepoint = nib.Nifti1Image(volume_timepoint, affine)

    # Visualize the NIfTI image with adjusted display parameters
    plotting.plot_stat_map(nifti_img_timepoint, title='Preprocessed Activation at Timepoint {}'.format(timepoint), display_mode='ortho', cut_coords=(0, 0, 0), draw_cross=True)
    plotting.show()

    # Proceed with the analysis using the filtered fMRI data
    n_complete_fmri_timepoints = n_words // 4
    n_complete_words = n_complete_fmri_timepoints * 4
    feature_values = feature_values[:n_complete_words, :]

    # Aggregate feature values by averaging every 4 words
    aggregated_feature_values = feature_values.reshape(n_complete_fmri_timepoints, 4, -1).mean(axis=1)
    print(f"Aggregated feature values shape: {aggregated_feature_values.shape}")

    # Align fMRI data timepoints with aggregated feature values
    filtered_fmri_data = fmri_time_series[:n_complete_fmri_timepoints, :]
    print(f"Adjusted fMRI time series shape: {filtered_fmri_data.shape}")

    # Ensure the shapes match
    assert aggregated_feature_values.shape[0] == filtered_fmri_data.shape[0], "Mismatch in timepoints after alignment"

    # Hemodynamic delay parameters
    hrf_delay = 4  # seconds
    tr = 2  # seconds
    shift_timepoints = int(hrf_delay / tr)
    print(f"Shifting feature data by {shift_timepoints} timepoints")

    # Shift the feature data
    shifted_feature_values = np.roll(aggregated_feature_values, shift_timepoints, axis=0)
    shifted_feature_values[:shift_timepoints, :] = 0  # Set initial rows to zero

    # Align fMRI data and shifted features
    min_length = min(shifted_feature_values.shape[0], filtered_fmri_data.shape[0])
    shifted_feature_values = shifted_feature_values[:min_length, :]
    filtered_fmri_data_shifted = filtered_fmri_data[:min_length, :]

    # Compute similarity matrices
    feature_similarity = cosine_similarity(shifted_feature_values)
    fmri_similarity = cosine_similarity(filtered_fmri_data_shifted)

    # Flatten upper triangles
    def flatten_upper_triangular(matrix):
        indices = np.triu_indices_from(matrix, k=1)
        return matrix[indices]

    feature_sim_flat = flatten_upper_triangular(feature_similarity)
    fmri_sim_flat = flatten_upper_triangular(fmri_similarity)

    # Compute Spearman correlation
    corr, _ = spearmanr(feature_sim_flat, fmri_sim_flat)
    print(f"Spearman correlation (with HRF delay): {corr}")

    # Visualize the similarity matrices
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(feature_similarity, cmap='viridis')
    plt.title('Feature Similarity Matrix (Shifted)')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(fmri_similarity, cmap='viridis')
    plt.title('fMRI Similarity Matrix')
    plt.colorbar()

    plt.tight_layout()
    plt.show()"""