import pyedflib
import re
import sys
import numpy as np

#------------------------------------------------------------------------------
# function: nedc_load_edf
#
# arguments: 
#   fname: filename (input)
#
# return: 
#   labels: store the EDF signal labels
#   fsamp: store the EDF signal sample frequency
#   sig: signals in the EDF file
#
# this function loads the EDF and return the signals
#
def nedc_load_edf(fname_a):

    # open an EDF file 
    #
    fp = pyedflib.EdfReader(fname_a)
    
    # get the metadata that we need:
    #  convert the labels to ascii and remove whitespace 
    #  to make matching easier
    #
    num_chans = fp.signals_in_file
    labels_tmp = fp.getSignalLabels()
    labels = [str(lbl.replace(' ', '')) for lbl in labels_tmp]

    # load each channel
    #
    sig = []
    fsamp = []
    for i in range(num_chans):
        sig.append(fp.readSignal(i))
        fsamp.append(fp.getSampleFrequency(i))

    # exit gracefully
    #
    return (fsamp, sig, labels)
#
# end of function 


# + endofcell="--"
#------------------------------------------------------------------------------
# function: rereference
#
# arguments:
#   sig: the signal data
#   labels: the channel labels
#
# return:
#   sig_mont: output signal data
#   labels_mont: output channel labels
#
# This rereferences the signal to a bipolar montage.
#
def rereference(sig, labels):
    sig_mont = list()
    
    # Define target bipolar montage
    #
    labels_mont = ['FP1-F7',
                        'F7-T3',
                        'T3-T5',
                        'T5-O1',
                        'FP2-F8',
                        'F8-T4',
                        'T4-T6',
                        'T6-O2',
                        'T3-C3',
                        'C3-CZ',
                        'CZ-C4',
                        'C4-T4',
                        'FP1-F3',
                        'F3-C3',
                        'C3-P3',
                        'P3-O1',
                        'FP2-F4',
                        'F4-C4']
    bipolarPairs = [('FP1', 'F7'),
                    ('F7', 'T3'),
                    ('T3', 'T5'),
                    ('T5', 'O1'),
                    ('FP2', 'F8'),
                    ('F8', 'T4'),
                    ('T4', 'T6'),
                    ('T6', 'O2'),
                    ('T3', 'C3'),
                    ('C3', 'CZ'),
                    ('CZ', 'C4'),
                    ('C4', 'T4'),
                    ('FP1', 'F3'),
                    ('F3', 'C3'),
                    ('C3', 'P3'),
                    ('P3', 'O1'),
                    ('FP2', 'F4'),
                    ('F4', 'C4')]
       
    
    # Apply montage to signal
    #
    for i, pair in enumerate(bipolarPairs):
        try:
            sig_mont.append(
                sig[_index(labels, pair[0])] - sig[_index(labels, pair[1])])
        except TypeError:
            sig_mont.append(np.zeros_like(sig[_index(labels, 'FP1')]))

    # exit gracefully
    #
    return (sig_mont, labels_mont)
#
# end of function


def _index(labels, match):
    regex = re.compile('^EEG\ ?{}-(REF|LE)'.format(match))
    for i, item in enumerate(labels):
        if re.search(regex, item):
            return i


# -


#------------------------------------------------------------------------------
# function: loadTSE
#
# arguments:
#   tfile_a: TSE event file
#
# return:
#   seizures: output list of seizures. Each event is tuple of 4 items:
#              (seizure_start [s], seizure_end [s], seizure_type, probability)
#   labels_mont: output channel labels
#
# Load seizure events from a TSE file.
#
def loadTSE(tfile_a):
    VERSION = 'version = tse_v1.0.0\n'
    SEIZURES = ('seiz', 'fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'cnsz',
                'tcsz', 'atsz', 'mysz', 'nesz')
    seizures = list()
    
    # Parse TSE file
    #
    with open(tfile_a, 'r') as tse:
        firstLine = tse.readline()
        
        # Check valid TSE
        #
        if firstLine != VERSION:
            raise ValueError(
                'Expected "{}" on first line but read \n {}'.format(VERSION,
                                                                    firstLine))
        
        # Skip empty second line
        #
        tse.readline()
        
        # Read all events
        #
        for line in tse.readlines():
            fields = line.split(' ')
            
            if fields[2] in SEIZURES:
                # Parse fields
                #
                start = float(fields[0])
                end = float(fields[1])
                seizure = fields[2]
                prob = float(fields[3][:-1])

                seizures.append((start, end, seizure, prob))
    
    # exit gracefully
    #
    return seizures
#
# end of function
# --
