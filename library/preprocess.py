from scipy.signal import butter, filtfilt


def preprocess(data, fs, wienerFilt=False):
    """Apply EEG pre-processing pipeline.
    
    Currently preprocessing only consists of a high pass (0.5Hz) and notch
    filter.
    
    Args:
        data: np.array containing the channels as rows and samples as columns
        fs: sampling frequency of the signal in Hz (200 Hz)
        wiener: apply Wiener filter (Default=False)
    Return:
        data: in-place transformed data np.array
    """
    
    # Low pass filter
    b, a = butter(4, 1/(fs/2), 'high')
    data = filtfilt(b, a, data)
    # 50 Hz notch
    b, a = butter(4, [47.5/(fs/2), 52.5/(fs/2)], 'bandstop')
    data = filtfilt(b, a, data)
    # 60 Hz notch
    b, a = butter(4, [57.5/(fs/2), 62.5/(fs/2)], 'bandstop')
    data = filtfilt(b, a, data)
    
    # Apply Wiener filter
    if wienerFilt:
        import wiener
        for filt in wiener.filters:
            filtered = wiener.wiener_filter(data, filt)
            data -= filtered
    
    return data


