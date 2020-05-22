import numpy as np
import pickle

filters = None
with open('filters.pickle', 'rb') as handle:
    filters = pickle.load(handle)

def wiener_filter(data, v):
    """Apply Wiener filter.

    Args:
        data: data contained in an array (row = channels, column = samples)
        v: Wiener filter
    Return:
        out: filtered data
    """
    lag = int(v.shape[0]/data.shape[0])
    filtered = list()
    for j in range(v.shape[1]):
        v_shaped = np.reshape(v[:,j], (data.shape[0], lag))
        out = np.convolve(v_shaped[0, :], data[0, :], 'full')
        for i in range(1, v_shaped.shape[0]):
            out += np.convolve(v_shaped[i, :], data[i, :], 'full')
        filtered.append(out)
    t = np.arange(0, v.shape[0], step=lag, dtype=int)
    filtered = np.dot(v[t,:], filtered)
    return np.array(filtered[:,:data.shape[1]])
