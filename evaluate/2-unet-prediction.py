"""Generate U-net predictions

Requires pre-processed EDF files similar to the training step
Requires the trained U-net model

Produces a results HDF5 file with predictions for every file
"""


# Libraries
import h5py
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import os
import tensorflow as tf

# Import some utilities from the training folder
import sys
sys.path.insert(0, 'training/3-DNN/')
from utils import build_windowfree_unet, setup_tf

# All relevant files
val_path = 'PATH_TO_DATASET.h5' # Pre-processed data file
saved_predictions = 'PATH_TO_PREDICTIONS.h5' # File to store the prediction
network_path = 'PATH_TO_NETWORK_WEIGHTS.h5' # Path to trained weights

# Data settings
fs = 200
n_channels = 18
n_filters = 8

# Tensorflow function to detect GPU properly
setup_tf()

# Loading the signals
with h5py.File(val_path, 'r') as f:
    file_names_test = []
    signals_test = []
    
    file_names_ds = f['filenames']
    signals_ds = f['signals']
    
    for i in range(len(signals_ds)):
        file_names_test.append(file_names_ds[i])
        data = np.asarray(np.vstack(signals_ds[i]).T, dtype=np.float32)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        signals_test.append((data-mean)/(std+1e-8))
        
# Building a windowfree U-net and load trained weights
unet = build_windowfree_unet(n_channels=n_channels, n_filters=n_filters)
unet.load_weights(network_path)

# Predictions using the windowfree U-net on CPU, our GPU ran out of memory
y_probas = []
reduction = 4096//4
with tf.device('cpu:0'):
    for signal in signals_test:
        signal = signal[:len(signal)//reduction*reduction, :]
        prediction = unet.predict(signal[np.newaxis, :, :, np.newaxis])[0, :, 0, 0]
        y_probas.append(prediction)

# Saving predictions
dt_fl = h5py.vlen_dtype(np.dtype('float32'))
dt_str = h5py.special_dtype(vlen=str)
with h5py.File(saved_predictions, 'w') as f:
    dset_signals = f.create_dataset('signals', (len(file_names_test),), dtype=dt_fl)
    dset_file_names = f.create_dataset('filenames', (len(file_names_test),), dtype=dt_str)
    
    for i in range(len(file_names_test)):
        dset_signals[i] = y_probas[i]
        dset_file_names[i] = file_names_test[i]
