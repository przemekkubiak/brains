## Overview
This repository contains a neural network model designed to predict brain activity from BERT embeddings and linguistic features, and to predict BERT embeddings and linguistic features from brain activity. The model features a single hidden layer with 1024 neurons.

## Model Architecture
- **Input**: BERT word embeddings and linguistic features
- **Hidden Layer**: 1024 neurons
- **Output**: Predicted brain activity or linguistic features

## Code
- **Wehbe_loader**: Loads and preprocessed the fMRI data from the public dataset by Wehbe et al. (2014) available [here]([url](https://www.cs.cmu.edu/~fmri/plosone/)).
- **align_data**: Loads the word-features and aligns them with image snippets.
- **train_and_save_models.py**: Defines the neural network, trains it on each task (word embeddings to fMRI, linguistic features to fMRI, fMRI to word embeddings, fMRI to linguistic features), saves each trained model.
- **get_outputs**: Uses the model on the test split, saves predictions and targets.
- **analyse_outputs**: Statistical Analysis of the model's performance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References
- Wehbe L, Murphy B, Talukdar P, Fyshe A, Ramdas A, et al. (2014) Simultaneously Uncovering the Patterns of Brain Regions Involved in Different Story Reading Subprocesses. PLoS ONE 9(11): e112575. doi:10.1371/journal.pone. 0112575
