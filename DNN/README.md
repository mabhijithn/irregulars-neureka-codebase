## README file for the base detection neural network code for the Neureka data


* Data Loading.ipynb
This file processes a directory structure of EEG recordings, preprocessed with ICLabel or not, and prepares a data structure
fit for training the neural network.

* Training U-net.ipynb
Notebook providing code for training a model on either raw EEG, Wiener filtered EEG, or ICLabel processed EEG. The notebook 
saves the model parameters during training for later use.

* Preparing Predictions.ipynb
This notebook takes previously trained model parameters and a file containing preprocessed EEG and prepares
individual predictions for our three different types of data (raw EEG, Wiener filtered and ICLabel) for fusion
and post-processing.
