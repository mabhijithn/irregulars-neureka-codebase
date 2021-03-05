""" Train LSTM network
LSTM network fuses output of U-nets

This file relies on pre-computed predictions and uses them to train the LSTM.
The model is saved in model-dnn-dnnw-dnnicalbl-lstm-4.h5
"""

import sys
# Root folder of main library
sys.path.insert(0, 'library')
# Root folder of EDF files
EDF_ROOT = '/esat/biomeddata/Neureka_challenge/edf/dev/'
# Root folder of predictions on edf files
PREDICTION_ROOT = 'evaluation'

# custom library
import nedc
import spir

# std lib
import glob
import os

# 3rd party lib
import h5py
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, GRU, LSTM
import numpy as np
import resampy


# +
def load_filenames():
    filenames = list()
    with h5py.File(os.path.join(PREDICTION_ROOT, 'prediction_test_iclabel.h5'), 'r') as f:
        filenames = list(f['filenames'])
    return filenames


def prepare_file(file_i, filename, classifiers, f_unet, model_type, fs):    
    # Load data
    x = list()
    for classifier in classifiers:
        if classifier['format'] == 'unet':
            z = list(f_unet[classifier['name']]['filenames'])
            file_i =  z.index(filename)
            predictions = f_unet[classifier['name']]['signals'][file_i]
            predictions = downsample(predictions, 200, fs)
        x.append(np.array(predictions, dtype=float))
        
    x = np.array(x)
    x = np.transpose(x)
    if model_type == 'lstm' or model_type == 'gru':
        x = x.reshape((len(x), 1, len(x[0])))
    # Collect the true lables
    seizures = nedc.loadTSE(os.path.join(EDF_ROOT,filename[:-4]+'.tse'))
    
    # Create labels at fs sampling rate
    y = spir.eventList2Mask(seizures, len(x), fs)
    
    return x,y


class AvgModel:
    def fit(*argv, **kwargs):
        return 0
    
    def reset_states(*argv, **kwargs):
        return 0
    
    def predict(x, *argv, **kwargs):
        if np.ndim(x) > 1:
            return np.mean(x, axis=1)
        else:
            return x


def downsample(x, oldFs, newFs):
    return resampy.resample(x, oldFs, newFs)


def findTse(filename):
    result = glob.glob(os.path.join(EDF_ROOT, '*', filename[3:6], filename.split('_')[0], filename.split('_')[1] + '_' + '[0-9_]*', filename + '.tse'))
    return result[0]


def build_model(n_input, model_type, complexity=None):
    if model_type == 'lstm':
        model = Sequential()
        model.add(Bidirectional(LSTM(complexity, stateful=True, return_sequences=False),
                                input_shape=(1, n_input), batch_size=1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')
    elif model_type == 'gru':
        model = Sequential()
        model.add(Bidirectional(GRU(complexity, stateful=True, return_sequences=False),
                                input_shape=(1, n_input), batch_size=1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')
    elif model_type == 'dense':
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', input_shape=(n_input, ), batch_size=1))
        model.compile(loss='mse', optimizer='adam')
    elif model_type == 'avg':
        model = AvgModel
    return model


def train(model, model_type, classifiers, filenames, fs=1):
    if model_type == 'avg':
        return 0
    
    # Preload U-Net data
    f_unet = dict()
    for classifier in classifiers:
        if classifier['format'] == 'unet':
            f_unet[classifier['name']] = h5py.File(classifier['file'], 'r')
    
    # Train
    for i, filename in enumerate(filenames):
        x, y = prepare_file(i, filename, classifiers, f_unet, model_type, fs)
        if np.any(y):
            model.fit(x, y, batch_size=1, epochs=15, verbose=1)
        else:
            model.fit(x, y, batch_size=1, epochs=1, verbose=1)
        model.reset_states()
        
    # Close U-Net data
    for key in f_unet:
        f_unet[key].close()


# +
fs = 1 # LSTM prediction sampling frequency

classifiers = [{
    'name': 'ICA',
    'file': os.join(PREDICTION_ROOT, 'prediction_test_iclabel.h5'),
    'fs': 200,
    'format': 'unet',    
},
    {
    'name': 'DNN',
    'file': os.join(PREDICTION_ROOT, 'prediction_test_raw.h5'),
    'fs': 200,
    'format': 'unet',    
},
{
    'name': 'DNN-wiener',
    'file': os.join(PREDICTION_ROOT, 'prediction_test_wiener.h5'),
    'fs': 200,
    'format': 'unet',
}
]

modeltype = 'lstm'
complexity = 4

filenames = load_filenames()
model = build_model(len(classifiers), modeltype, complexity)
train(model, modeltype, classifiers, filenames, fs)
model.save('model-dnn-dnnw-dnnicalbl-lstm-4.h5')
