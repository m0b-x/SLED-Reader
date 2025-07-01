import numpy as np
from scipy.signal import butter, sosfiltfilt, firwin, lfilter

def compute_magnitude(x, y, z):
    xyz = np.vstack((x, y, z)).astype(float)
    return np.linalg.norm(xyz, axis=0)

def fir_hamming_filter(data, cutoff_hz=5.0, fs=100.0, numtaps=11):
    nyq = fs / 2.0
    normalized_cutoff = cutoff_hz / nyq
    fir_coeff = firwin(numtaps=numtaps, cutoff=normalized_cutoff, window='hamming')
    return lfilter(fir_coeff, 1.0, data)

def sliding_window_filter(data, window_size=15):
    data = np.asarray(data)
    half_window = window_size // 2
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        smoothed[i] = np.mean(data[start:end])
    return smoothed

def butter_filter(data, cutoff, fs, order, kind):
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple)):
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype=kind, analog=False, output='sos')
    return sosfiltfilt(sos, data)

def minmax_scale(data, target_min=0.0, target_max=1.0):
    data = np.asarray(data)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.full_like(data, target_min)
    return (data - min_val) / (max_val - min_val) * (target_max - target_min) + target_min

def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def net_magnitude_filter(data, window_size):
    data = np.asarray(data)
    net_mag = np.zeros_like(data)
    for i in range(len(data)):
        if i < window_size:
            net_mag[i] = data[i]
        else:
            net_mag[i] = data[i] - np.mean(data[i - window_size:i])
    return net_mag

def kalman_filter(data):
    n = len(data)
    x_est = np.zeros(n)
    p = np.zeros(n)
    x_est[0] = data[0]
    p[0] = 1.0
    q = 0.01  # process variance
    r = 1.0   # measurement variance
    for k in range(1, n):
        x_pred = x_est[k - 1]
        p_pred = p[k - 1] + q
        k_gain = p_pred / (p_pred + r)
        x_est[k] = x_pred + k_gain * (data[k] - x_pred)
        p[k] = (1 - k_gain) * p_pred
    return x_est

