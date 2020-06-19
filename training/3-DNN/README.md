## README file for the base detection neural network code for the Neureka data


* 3-load-data.py

This file processes a directory structure of EEG recordings, preprocessed with ICLabel or not, and prepares a data structure
fit for training the neural network.

* 3-train-unet.py

Script providing code for training a model on either raw EEG, Wiener filtered EEG, or ICLabel processed EEG. The notebook 
saves the model parameters during training for later use.

* utils.py

File with utility code for the data generator and building the U-net
