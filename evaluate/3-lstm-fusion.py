"""Fuse U-net predictions with an LSTM network

Requires predictions from the U-nets
Requires the trained LSTM model

Produces a results pickle file to be used by the postprocessing
"""

# +
import sys
# Root folder of main library
sys.path.insert(0, 'library')
# Root folder of EDF files
EDF_ROOT = '/esat/biomeddata/Neureka_challenge/edf/dev/'
# Root folder of predictions on edf files
PREDICTION_ROOT = 'evaluation'

# std lib
import os
import pickle

# 3rd party lib
import h5py
from keras.models import load_model
import numpy as np
import resampy


# +
def load_filenames():
    filenames = list()
    with h5py.File(os.joind(PREDICTION_ROOT, 'prediction_test_iclabel.h5'), 'r') as f:
        filenames = list(f['filenames'])
    return filenames


def prepare_file(file_i, filename, classifiers, f_nick, model_type):    
    # Load data
    x = list()
    for classifier in classifiers:
        if classifier['format'] == 'nick':
            z = list(f_nick[classifier['name']]['filenames'])
            file_i =  z.index(filename)
            predictions = f_nick[classifier['name']]['signals'][file_i]
            predictions = downsample(predictions, 200, fs)
        x.append(np.array(predictions, dtype=float))
        
    x = np.array(x)
    x = np.transpose(x)
    if model_type == 'lstm' or model_type == 'gru':
        x = x.reshape((len(x), 1, len(x[0])))
    
    return x

def downsample(x, oldFs, newFs):
    return resampy.resample(x, oldFs, newFs)


        
def test(model, modeltype, classifiers, filenames):
    # Preload Nick data
    f_nick = dict()
    for classifier in classifiers:
        if classifier['format'] == 'nick':
            f_nick[classifier['name']] = h5py.File(classifier['file'], 'r')
    
    # Predict probabilities
    results = list()
    for i, filename in enumerate(filenames):
        x, y = prepare_file(i, filename, classifiers, f_nick, modeltype)
        u = model.predict(x, batch_size=1)
        model.reset_states()
        results.append(u)
        
    with open('lstm-results.pkl', 'wb') as filehandler:
        pickle.dump(results, filehandler)
                    
    # Close Nick data
    for key in f_nick:
        f_nick[key].close()


# +
fs = 1

classifiers = [{
    'name': 'ICA',
    'file': os.join(PREDICTION_ROOT, 'prediction_test_iclabel.h5'),
    'fs': 200,
    'format': 'nick',    
},
    {
    'name': 'DNN',
    'file': os.join(PREDICTION_ROOT, 'prediction_test_raw.h5'),
    'fs': 200,
    'format': 'nick',    
},
{
    'name': 'DNN-wiener',
    'file': os.join(PREDICTION_ROOT, 'prediction_test_wiener.h5'),
    'fs': 200,
    'format': 'nick',
}
]

modeltype = 'lstm'
complexity = 4

filenames = load_filenames()
model = load_model('model-dnn-dnnw-dnnicalbl-lstm-4.h5')
test(model, modeltype, classifiers, filenames)