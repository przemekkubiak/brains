import nibabel as nib
import numpy as np
from Wehbe_loader import words, time, meta
from nilearn import plotting
import pandas as pd

time_data = time

words = words[0]

if __name__ == '__main__':

    # Load the .npy files
    runs_fmri = np.load('Wehbe_data/runs_fmri.npy')
    time_fmri = np.load('Wehbe_data/time_fmri.npy')
    time_words_fmri = np.load('Wehbe_data/time_words_fmri.npy')
    words_fmri = np.load('Wehbe_data/words_fmri.npy')

    # Load the NIfTI file
    nifti_image = nib.load('data/subject_2_preprocessed_volume.nii')

    # Get the data from the NIfTI file
    preprocessed_volume = nifti_image.get_fdata()

    # Get the affine matrix from the NIfTI file
    affine = nifti_image.affine

    # Get the number of images and words
    num_images = preprocessed_volume.shape[0]
    num_words = len(words)

    # Ensure that the number of images matches the number of time points
    assert num_images == len(time_data), "The number of images should match the number of time points in time_data."

    # Align words with images based on time intervals
    aligned_words = []
    aligned_fmri_data = []
    aligned_data = []


    for i in range(num_images):
        image_time = float(time_fmri[i])
        print(image_time) # Assuming the first column of time_data is the time measurement
        start_time = image_time - 6  # Start time of the 2-second interval leading up to the image capture, shifted by 4 seconds
        end_time = image_time - 4  # End time of the interval, shifted by 4 seconds
        words_at_time = [words_fmri[k] for k in range(len(time_words_fmri)) if time_words_fmri[k]<end_time and time_words_fmri[k]>=start_time]
        aligned_words.append(words_at_time)
        print(words_at_time)
        # Ensure words_at_time contains exactly 4 words
        if len(words_at_time) == 4:
            # Divide the fMRI data for each image into four segments, each corresponding to a word
            fmri_data_segments = np.array(preprocessed_volume[i])
            current_dim = fmri_data_segments.shape
            print(f'Current dim: {current_dim}')
            aligned_data.append({'words': words_at_time, 'fmri_data': fmri_data_segments})

    # Convert aligned words and fMRI data to numpy arrays
    #aligned_words_array = np.array(aligned_words, dtype=object)  # Use dtype=object for arrays of lists
    #aligned_fmri_data_array = np.array(aligned_fmri_data, dtype=object)

    # Convert aligned data to a numpy array
    aligned_data_array = np.array(aligned_data, dtype=object)

    # Save the aligned data to an .npz file
    np.savez('aligned_data.npz', aligned_data=aligned_data_array)
    
    # Convert aligned data to a pandas DataFrame
    aligned_df = pd.DataFrame(aligned_data)

    # Save the DataFrame to a .csv file using pandas
    aligned_df.to_csv('aligned_data.csv')