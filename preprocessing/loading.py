import numpy as np
import resampy

import nedc
import preprocess

def loadRecording(filename, wiener=False):
    """Loads and preprocess an EDF file.
    
    Loads an EDF file. Converts it to a bipolar montage, resamples it to 200Hz
    and applies a preprocessing pipeline to the data.
    
    Args:
        filename: EDF filename
        wiener: apply Wiener filter (Default=False)
    Return
        fs: sampling frequency of the signal in Hz (200 Hz)
        data: np.array containing the channels as rows and samples as columns
        labels_mount: list of the labels of the bipolar montage
    """
    
    # Load Data
    (fsamp, sig, labels) = nedc.nedc_load_edf(filename)
    fs = fsamp[nedc._index(labels, 'FP1')]
    (sig_mont, labels_mont) = nedc.rereference(sig, labels)
    data = np.array(sig_mont, dtype=np.float32)
    
    # Resample data
    data = resampy.resample(data, fs, 200)
    fs = 200
    
    # Preprocess
    data = preprocess.preprocess(data, fs, wiener)
    
    return (fs, data, labels_mont)

