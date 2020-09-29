import numpy as np
from scipy.signal import convolve


# ## COVARIANCE MATRIX ESTIMATION ###

def build_lagged_copies(data, lag, ram=True, file=None):
    """ Build matrix containing lagged copies of data
    
    Create a matrix with lagged copies of the rows in data.
    Args:
        data: data contained in an array (row = channels, column = samples)
        lag: number of sample lags
        ram: if True store in RAM; else as a memmap
        file: location of memmap. If ram is True and file is None. memmap is
              stored in /tmp
    Return
        noise: list of noise events in seconds. Each row contains two
                 columns: [start time, end time]
        noise_mask: bool mask containing True on samples marked as noise.
    """
    if ram:
        # Build copies matrix in RAM
        lagged_data = np.empty((len(data)*(lag+1), len(data[0])))
    else:
        # Build copies as a memmap
        if file is None:
            with tempfile.NamedTemporaryFile() as f:
                file = f.name
                lagged_data = np.memmap(f.name, dtype='float64', mode='w+', shape=(len(data)*(lag+1), len(data[0])))
        else:
            lagged_data = np.memmap(file, dtype='float64', mode='w+', shape=(len(data)*(lag+1), len(data[0])))
        logging.info('Created file ({}) containing lagged copies of data'.format(file))
    for num in range(lag + 1):
        for ch in range(len(data)):
            if num > 0:
                lagged_data[ch*(lag+1)+num,:-num] = data[ch,num:]
                lagged_data[ch*(lag+1)+num,-num:] = 0
            else:
                lagged_data[ch*(lag+1)+num,:] = data[ch,:]
    return lagged_data


def build_cov(data, events, lag, fs):
    """Build signal covariance matrix.
    
    Covariance matrix is calculated over all event epochs.
    
    Args:
        data: data contained in an array (row = channels, column = samples)
        events: list of events times in seconds. Each row contains two
                 columns: [start time, end time]
        lag: positive lag in samples
        fs: sampling frequency of the data in Hertz
        
    Return:
        Rxx: covariance matrix of event epochs (accross channels and lags)
    """
    t = 0
    Rxx = np.zeros((len(data)*(lag+1), len(data)*(lag+1)))
    for event in events:
        lagged = build_lagged_copies(data[:,np.arange(int(event[0]*fs), int(event[1]*fs))], lag)
        t += len(lagged)
        Rxx += np.dot(lagged, np.transpose(lagged))
    Rxx = Rxx / t
    return Rxx


def build_rnn(data, seizures, lag, fs):
    """Build noise covariance matrix.
    
    Covariance matrix is calculated over all non-seizure epochs.
    
    Args:
        data: data contained in an array (row = channels, column = samples)
        seizures: list of events times in seconds. Each row contains two
                 columns: [start time, end time]
        lag: positive lag in samples
        fs: sampling frequency of the data in Hertz
        
    Return:
        Rnn: covariance matrix of noise epochs (accross channels and lags)
    """    
    # Get noise events from seizure events
    seizure_mask = eventList2Mask(seizures, len(data[0]), fs)
    noise_mask = np.logical_not(seizure_mask)
    noise_events = mask2eventList(noise_mask, fs)
    
    Rnn = build_cov(data, noise_events, lag, fs)

    return Rnn



# ## MAX-SNR filtering ##

def find_noise(data, fs):
    """Detect noise events on a channel by channel basis

    Noise is defined as samples above 600uV

        Args:
            data: data contained in an array (row = channels, column = samples)
            fs: sampling frequency of the data in Hertz
        Return
            noise_list: list of noise masks. Each element of the list is a
                a channel.
    """
    noise_list = np.zeros((len(data), len(data[0])), dtype=bool)
    for i in range(len(data)):
        noise_mask = np.abs(data[i]) > 400
        noise = mask2eventList(noise_mask, fs)
        noise = extend_event(noise, 1.5, len(data[0])/fs)
        noise_list[i, :] = eventList2Mask(noise, len(data[0]), fs)
    return noise_list


def maxspir_filter(data, v, noise):
    """Apply maxSPIR filter.

    Args:
        data: data contained in an array (row = channels, column = samples)
        v: maxSPIR filter as a flattened vector
        noise: noise binary mask contained in an array of the same size as data
    Return:
        out: filtered data
    """
    lag = int(len(v)/len(data))
    v_shaped = np.reshape(v, (len(data), lag))
    out = np.convolve(v_shaped[0, :], data[0, :]*np.logical_not(noise[0, :]), 'same')
    for i in range(1, v_shaped.shape[0]):
        out += np.convolve(v_shaped[i, :], data[i, :]*np.logical_not(noise[i, :]), 'same')
    return out


def calculate_filter(rss, rnn, regularization_strategy='none'):
    """Calculate max-SNR filter
    """
    ws, vs = np.linalg.eig(rss)
    index_s = np.argmax(np.cumsum(ws)/np.sum(ws)> 0.95)
    wn, vn = np.linalg.eig(rnn)
    index_n = np.argmax(np.cumsum(wn)/np.sum(wn)> 0.9)
    vt = np.concatenate((vn[:,:index_n+1], vs[:,:index_s+1]), axis=1)
    u, s, v = np.linalg.svd(vt, full_matrices=False)
    index_t = np.argmax(np.cumsum(s)/np.sum(s)> 0.99)
    i = index_t
    t = u

    if regularization_strategy == 'none':
        w, v = np.linalg.eig(np.dot(np.linalg.inv(rnn), rss))
    elif regularization_strategy == 'pca':
        w, v = np.linalg.eig(np.dot(np.dot(
        np.dot(t[:,0:i], np.linalg.inv(np.dot(np.dot(np.transpose(t[:,0:i]), rnn), t[:,0:i]))),
        np.transpose(t[:,0:i])), rss))

    return np.real(v)


def find_events(seizures, power, data, fs, duration=30, minTresh=70):
    """Find events in power of filtered data.

    Args:
        power: power of filtered data (single channel)
        data: raw data contained in an array (row = channels, column = samples)
        fs: sampling frequency in Hz
        duration: duration of each event in seconds [default:30]
        minTreshold: minimum detection threshold in mV [default:70]
    Return:
        events: indices of detected events (sorted with decreasing power)
    """
    threshold = 9999
    power_derivative = np.diff(power)
    s_len = int(fs/2)
    smooth_derivative = np.convolve(power_derivative, np.ones((s_len,))/s_len, 'same')
    power_copy = np.copy(power)
    seizure_mask = eventList2Mask(seizures, len(power), fs)
    np.putmask(power_copy, seizure_mask, 0)
    events = list()
    while threshold > minTresh and len(events) < 150:
        i = np.argmax(power_copy)
        threshold = power_copy[i]

        # Start
        i0 = i - s_len
        while i0 > 0 and smooth_derivative[i0] > 0:
            i0 -= 1
        i0 += s_len
        #End
        i1 = i + s_len
        while i1 < len(smooth_derivative) and smooth_derivative[i1] < 0:
            i1 += 1
        i1 -= s_len
        
        np.put(power_copy, np.arange(
            max(0, i0-s_len),
            min(len(power_copy), i1+s_len)), 0)
        if threshold > minTresh and i1-i0 < 60*fs and i1-i0 > fs:
            events.append((i0, i1))

    return events


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
    power = rolling_rms(data, int(1*fs))
    
    
    dpower = np.diff(power)
    s_len = int(fs/2)
    dpower = convolve(dpower, np.ones((dpower.shape[0],s_len))/s_len, 'same')
    seizure_mask = eventList2Mask(seizures, power.shape[1], fs)

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
        eventmask = eventList2Mask(events, len(interference_mask), fs)
        interference_mask = np.logical_or(interference_mask, eventmask)
        
    return mask2eventList(interference_mask, fs)


def rolling_rms(data, duration):
    """ Calculate rolling average RMS.

    Args:
        data: data contained in a vector
        duration: rolling average window duration in samples
    Return:
        power: rolling average RMS
    """
    power = np.square(data)
    if data.ndim == 2:
        power = convolve(power, np.ones((data.shape[0], duration))/duration, mode='same')
    elif data.ndim == 1:
        power = convolve(power, np.ones((duration,))/duration, mode='same')
    else:
        TypeError('Dimmension of data should be 1 or 2 to convolve.')
    return np.sqrt(power)


# ## EVENT & MASK MANIPULATION ###

def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.
    
    Returns a logical array of length totalLen.
    All event epochs are set to True
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,), dtype=bool)
    for event in events:
        for i in range(min(int(event[0]*fs), totalLen-1), min(int(event[1]*fs), totalLen-1)):
            mask[i] = True
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.
        
    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]
    
    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask)-1)/fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, end_i[0]/fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[start_i[-1]/fs, (len(mask)-1)/fs]]
                start_i = np.delete(start_i, len(start_i)-1)
        for i in range(len(start_i)):
            events.append([start_i[i]/fs, end_i[i]/fs])
        events += tmp
    return events


def extend_event(events, time, max_time):
    """Extends events in event list by time.
    
    The start time of each event is moved time seconds back and the end
    time is moved time seconds later
    
    Args:
        events: list of events. Each event is a tuple
        time: time to extend each event in seconds
        max_time: maximum end time allowed of an event.
    Return
        extended_events: list of events which each event extended.
    """
    extended_events = events.copy()
    for i, event in enumerate(events):
        extended_events[i] = [max(0, event[0] - time),
                                  min(max_time, event[1] + time)]
    return extended_events


def merge_events(events, distance):
    i = 1
    tot_len = len(events)
    while i < tot_len:
        if events[i][0] - events[i-1][1] < distance:
            events[i-1][1] = events[i][1]
            events.pop(i)
            tot_len -= 1
        else:
            i += 1
    return events
