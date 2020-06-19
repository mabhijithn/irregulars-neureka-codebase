""" Load all edf files.

This file reads all edf files in a given directory and combines them in a data structure
to be used during training of the U-net.
Has to be run for every U-net that is going to be trained.
"""


# Importing of necessary libraries

import sys
# Root folder of main library
sys.path.insert(0, 'library')
import loading
import nedc

# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py


# Variables that have to be set accordingly
base_directory = 'PATH_OF_ORIGINAL_DATASET' # Path to directory with EDF files
save_path = 'PATH_OF_LOADED_FILE.h5' # Path to file containing the loaded data structure

includes_tse = False # Boolean flag, whether .tse files with ground-truth labels can also be loaded
wiener = False # Boolean flag, whether Wiener filtering is applied

def wrangle_tse(tse_path, length, fs=200):
    """
    Function to process a given .tse file into a time series of labels.
    tse_path: Path to .tse file
    length: Number of time samples in the corresponding EEG time series
    fs: Sampling rate of the time series
    """
    label_dct = {
        'bckg': 0,
        'seiz': 1
            }

    label = np.zeros(shape=(length,))

    df_label = pd.read_csv(
            filepath_or_buffer=tse_path,
            header=None,
            sep=' ',
            names=['start', 'stop', 'label', 'confidence'],
            skiprows=2,
            na_filter=False
        )

    for i, time_s in enumerate(df_label.start):
        label_str = df_label.label[i]
        label[int(time_s*fs):] = label_dct[label_str]
    
    return label


"""
Function that loads a given recording, resamples and re-references. 
Applies Wiener filtering, and loads the .tse file; both when appropriate.
"""
if includes_tse:
    def process_file(file_path):
        file_name = file_path[-22-7:-4-7]
        
        try:
            (fs, data, mount) = loading.loadRecording(file_path, wiener=wiener)
            signal = np.asarray(data, dtype=np.float32)
        
        label = wrangle_tse(file_path[:-4]+'.tse_bi', length=signal.shape[1])
        label = np.asarray(label, dtype=np.uint8)
        except TypeError:
            signal = 0
        
        return file_name, signal, label
else:
    def process_file(file_path):
        file_name = file_path[-22-7:-4-7]
        
        try:
            (fs, data, mount) = loading.loadRecording(file_path, wiener=wiener)
            signal = np.asarray(data, dtype=np.float32)

        except TypeError:
            signal = 0

        return file_name, signal
    
    
# Walks through the given directory and creates list of all .edf files
edf_files = []
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith(".edf"):
             edf_files.append(os.path.join(root, file))

# Processes all files in parallel
pool = Pool()

if includes_tse:
    file_names, signals, labels = zip(*pool.map(process_file, edf_files))
else:
    file_names, signals = zip(*pool.map(process_file, edf_files))

# Necessary manipulation, the ICLabel procedure rejects some training files due to noise.
file_names, signals = zip(*[[file_name, signal] for file_name, signal in zip(file_names, signals) 
                          if len(np.asarray(signal).shape)!=0])

# Saves new data structure to disk
if includes_tse:
    dt_fl = h5py.vlen_dtype(np.dtype('float32'))
    dt_int = h5py.vlen_dtype(np.dtype('uint8'))
    dt_str = h5py.special_dtype(vlen=str)
    
    with h5py.File(save_path, 'w') as f:
        dset_signals = f.create_dataset('signals', (len(signals), 18), dtype=dt_fl)
        dset_labels = f.create_dataset('labels', (len(labels),), dtype=dt_int)
        dset_file_names = f.create_dataset('filenames', (len(file_names),), dtype=dt_str)
        
        for i in range(len(signals)):
            dset_signals[i] = signals[i]
            dset_labels[i] = labels[i]
            dset_file_names[i] = file_names[i]
else:
    dt_fl = h5py.vlen_dtype(np.dtype('float32'))
    dt_str = h5py.special_dtype(vlen=str)
    
    with h5py.File(save_path, 'w') as f:
        dset_signals = f.create_dataset('signals', (len(signals), 18), dtype=dt_fl)
        dset_file_names = f.create_dataset('filenames', (len(file_names),), dtype=dt_str)
        
        for i in range(len(signals)):
            dset_signals[i] = signals[i]
            dset_file_names[i] = file_names[i]