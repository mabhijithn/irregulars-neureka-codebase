# Training pipeline

The seizure detection pipeline is trained in several independent steps:

1. Train and build the Wiener filters.
2. Remove artifcats with ICLabel 
3. Train U-nets
4. Train LSTM


## 1. Wiener pre-processing

Wiener pre-processing builds a filter bank of spatio-temporal filters based on artifacts identified in the training set.

1. A set of high power non-seizure epochs are identified
2. PCA compressed spatio-temporal covariance matrices are used to represent the artifacts
3. K-means clustering is used to group the artifacts
4. The average representation of the groups is used to pre-compute a spatio-temporal wiener filter

## 2. ICLabel pre-processing

ICLabel pre-processing rejects any ``bad-channels'' and then removes any components of the signal which are clustered as artifacts.

1. High pass filtering of the data
2. Rejection of any bad channels (flat channels for above 20 seconds, channels with high SNR and channels with very low correlation to their estimation based on the rest of the channels)
3. Computation of the Independent Components using SOBI ICA.
4. Classification of the components using the ICLabel package of EEGlab
5. Rejection of all the components with a correlation higher than 0.6 to the following clusters: Muscle, Eye, Heart, Line Noise, Channel Noise

## 3. Train U-net

A separate U-net is trained and stored for each available "view" on the data.

## 4. Train LSTM
Fusion of the different U-Net DNN results is done using a shallow recurrent neural network.

The recurrent NN is built using two layers:

1. A bidirectional LSTM layer with a state vector of length 4
2. A dense layer combining the the outputs of the LSTM layer
