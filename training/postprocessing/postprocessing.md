# Postprocessing

This folder contains the code used to fuse results from the differnt U-Net DNN along with the code to produce the hypothesis text file.

## LSTM fusion
Fusion of the different U-Net DNN results is done using a shallow recurrent neural network.

The recurrent NN is built using two layers:

1. A bidirectional LSTM layer with a state vector of length 4
2. A dense layer combining the the outputs of the LSTM layer

### Training the recurrent NN
The training of the recurrent NN is done on the development. Cells 2 and 3 show how the network was trained and tested.

### Predicting with the recurrent NN
The trained weights of the recurrent NN are stored in a .h5 model and loaded for prediction on the evaluation set in cell 3.

## Postprocessing rules
Several sanity rules are used on the output of the recurrent NN. These rules are used in cell 5 to produce the hypothesis file.
