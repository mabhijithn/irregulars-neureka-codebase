
"""Build hypothesis file by applying post-processing rules
to LSTM output

Requires LSTM output

Produces a hypothesis txt file
"""

# +
import sys
# Root folder of main library
sys.path.insert(0, 'library')
# Root folder of EDF files
EDF_ROOT = '/esat/biomeddata/Neureka_challenge/edf/dev/'
# Root folder of predictions on edf files
PREDICTION_ROOT = 'evaluation'

# custom lib
import spir

# std lib
import os
import pickle

# 3rd party lib
import h5py
import numpy as np


# +
def load_filenames():
    filenames = list()
    with h5py.File(os.join(PREDICTION_ROOT, 'prediction_test_iclabel.h5'), 'r') as f:
        filenames = list(f['filenames'])
    return filenames


filenames = load_filenames()
with open('lstm-results.pkl', 'rb') as filehandler:
    results = pickle.load(filehandler)


threshold = 0.55

for i, filename in enumerate(filenames):
    # Apply threshold on baseline corrected prediction
    hyp = spir.mask2eventList((results[i].flatten() - np.median(results[i].flatten())) > threshold, fs)
    # Merge events closer than 30seconds
    hyp = spir.merge_events(hyp, 30)
    
    # Remove events with mean prediction < 82% of event with max prediction
    if len(hyp):
        amp = list()
        for event in hyp:
            amp.append(np.mean(results[i].flatten()[int(event[0]*fs):int(event[1]*fs)]))
        amp = np.array(amp)
        amp /= np.max(amp)

        hyp = list(np.array(hyp)[amp > 0.82])
    
    with open('hyp_lstm.txt', 'a') as handle:
        for event in hyp:
            # Remove short events
            if event[1] - event[0] > 15:
                amp = np.mean(results[i][int(event[0]*fs):int(event[1]*fs)])
                # Shorten events by 2 seconds
                handle.write('{} {} {} {} 16\n'.format(filename, event[0]+1, event[1]-1), amp)
