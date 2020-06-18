# Training pipeline

The seizure detection pipeline is trained in several independent steps:

1. Train and build the Wiener filters.
2. Build ICLabel filtered data
3. Train U-nets
4. Train LSTM


## 1. Wiener pre-processing

Wiener pre-processing builds a filter bank of spatio-temporal filters based on artifacts identified in the training set.

1. A set of high power non-seizure epochs are identified
2. PCA compressed spatio-temporal covariance matrices are used to represent the artifacts
3. K-means clustering is used to group the artifacts
4. The average representation of the groups is used to pre-compute a spatio-temporal wiener filter


## 4. Train LSTM
Fusion of the different U-Net DNN results is done using a shallow recurrent neural network.

The recurrent NN is built using two layers:

1. A bidirectional LSTM layer with a state vector of length 4
2. A dense layer combining the the outputs of the LSTM layer