"""
Train Wiener filters based on clusters of similar artefacts

This script is heavy in computation and RAM (~50GB) as it computes and loads many large covariance matrices.
This script was originally a jupyter notebook (converted with jupytext). Some plots are intended as interactive
steps to optimize the algorithm parameters (such as number of clusters).

The script produces a pickle object containing the filters
"""

import sys
# Adapt these two path to the root of the EDF data and to the root of the codebase
EDF_ROOT = '/esat/biomeddata/Neureka_challenge/edf/train'
sys.path.insert(0, 'library')

# custom library
import nedc
import loading
import spir

# std lib
import glob
from joblib import Parallel, delayed
import multiprocessing
import os
from pathlib import Path
import pickle

# 3rd party lib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

num_cores = multiprocessing.cpu_count()


# -
# ## Find interference
# +
def find_interference(data, fs, seizures, maxThresh=500, minTresh=70):
    """Find interference in raw data.

    Args:
        data: raw data contained in an array (row = channels, column = samples)
        fs: sampling frequency in Hz
        duration: duration of each event in seconds [default:30]
        minTreshold: minimum detection threshold in mV [default:70]
    Return:
        interferences: indices of detected events (sorted with decreasing power)
    """
    threshold = 9999
    power = spir.rolling_rms(data, int(1*fs))
    
    dpower = np.diff(power)
    s_len = int(fs/2)
    dpower = convolve(dpower, np.ones((dpower.shape[0],s_len))/s_len, 'same')
    seizure_mask = spir.eventList2Mask(seizures, power.shape[1], fs)

    interference_mask = np.zeros((data.shape[1],))
    
    for c, channel in enumerate(power):
        np.putmask(channel, seizure_mask, 0)
        events = list()
        while threshold > minTresh and len(events) < 50:
            i = np.argmax(channel)
            threshold = channel[i]

            # Start
            i0 = i - s_len
            while i0 > 0 and dpower[c, i0] > 0:
                i0 -= 1
            i0 += s_len
            #End
            i1 = i + s_len
            while i1 < dpower.shape[1] and dpower[c, i1] < 0:
                i1 += 1
            i1 -= s_len

            np.put(channel, np.arange(
                max(0, i0-s_len),
                min(len(channel), i1+s_len)), 0)
            if threshold > minTresh and threshold < maxThresh and i1-i0 < 60*fs and i1-i0 > fs:
                events.append((i0/fs, i1/fs))
        eventmask = spir.eventList2Mask(events, len(interference_mask), fs)
        interference_mask = np.logical_or(interference_mask, eventmask)
        
    return spir.mask2eventList(interference_mask, fs)


# +
# Find artefacts and compute covariance matrix
lag = 50
total_events = 0
total_seizures = 0
event_dict = dict()
eventFiles = list()
allEvents = list()

rnns = list()

for filename in glob.iglob(EDF_ROOT + '/**/*.edf', recursive=True):
    seizures = nedc.loadTSE(filename[:-3] + 'tse')
    (fs, data, labels) = loading.loadRecording(filename)

    events = find_interference(data, fs, seizures, maxThresh=500, minTresh=100)

    if len(events) > 0:
        event_dict[filename] = events
        allEvents += events
        total_events += len(events)
        total_seizures += len(seizures)

        for i in range(len(events)):
            eventFiles.append(filename)

        rnn = Parallel(n_jobs=num_cores)(delayed(spir.build_cov)(data, [event], lag, fs) for event in events)
        rnns += rnn
    
    # Limit to 2000 events
    if total_events > 2000:
        break


# -
# ## Compress Rnns
# +
tmp = list()
for rnn in rnns:
    tmp.append(rnn.flatten()/np.sum(np.diag(rnn))) # Normalize
pca = PCA(0.99)
pca.fit(tmp)
compressed = pca.fit_transform(tmp)
print('Number of compressed components: {}'.format(compressed.shape[1]))


# -
# ## Perform K-means clustering
# +
## Find n-clusters
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


sse = calculate_WSS(compressed, 12)

plt.figure(figsize=(16, 6))
plt.plot(sse[:])
plt.ylabel('# SSE')
plt.xlabel('# of clusters')
plt.title('Choice of # of cluster')
plt.show()
# -
n_clusters = 6 # Select 6 clusters (based on SSE plot)

kmeans = KMeans(n_clusters=n_clusters).fit(compressed)

import pickle 
with open('kmeans.pkl', 'wb') as filehandler:
    pickle.dump(kmeans, filehandler)

plt.figure(figsize=(16, 6))
plt.hist(kmeans.labels_)
plt.xlabel('Cluster labels')
plt.title('histogram of cluster labels')
plt.show()

plt.figure(figsize=(9, 9))
for label in range(n_clusters):
    examples = np.where(kmeans.labels_ == label)[0]
    plt.scatter(compressed[examples, 0], compressed[examples, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Scatter of the first two principle components')
plt.legend(range(6))
plt.show()


# +
## Calculate filters

# Average cluster
avg = list()
for label in range(n_clusters):
    avg.append(np.zeros_like(rnns[0]))
    examples = np.where(kmeans.labels_ == label)[0]
    for example in examples:
        avg[-1] += rnns[example] / np.sum(np.diag(rnns[example])) / len(examples)

# Filter
filters = list()
for label in range(n_clusters):
    w, v = np.linalg.eig(avg[label])
    index_i = np.argmax(np.cumsum(np.real(w))/np.sum(np.real(w)) > 0.99)
    filters.append(np.real(v[:,:index_i]))

# Write filters
with open('filters.pkl', 'wb') as filehandler:
    pickle.dump(filters, filehandler)