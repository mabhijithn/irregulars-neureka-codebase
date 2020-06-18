# Seizure Detection Codebase used in Neureka Challenge 2020

This repository contains the code corresponding to the 'Biomed Irregulars' submission of the Neureka Challenge 2020. 
The seizure detection algorithm is based on the fusion of multiple attention U-nets, each operating on a distinct view of the EEG data. The outputs of the different U-nets are combined into an LSTM.


## Code: Content 
1. Preprocessing - This folder contains the preprocessing functions: re-referencing, resampling and filtering. Here are also the functions for creating the distinct views of the data, the Multichannel Wiener filter and IC label preprocessing.
2. DNN - This folder contains the code for the Attention U-net for training and predicting.
3. Postprocessing - This folder contains the LSTM based fusion of the three different U-net predictions. This is followed by a set of postprocessing rules which converts the LSTM output seizure annotations.



1. DNN - Directory containing codes and scripts related to the U-Net neural network used in seizure detection
3. Postprocessing - LSTM based fusion of the three different NN predictions. This is followed by a set of postprocessing rules which converts the LSTM output seizure annotations.
4. Seizure Detection Pipeline-Description.docx - An MS Word document describing the seizure prediction pipeline briefly.


- Biomed Irregulars:

  C. Chatzichristos, J. Dan, A.M. Narayanan, N. Seeuws, K. Vandecasteele
