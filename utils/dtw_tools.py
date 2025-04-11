# dtw_tools.py
import pdb

import numpy as np
from fastdtw import fastdtw
import librosa
from scipy.spatial.distance import euclidean


def dtw_distance(ts1, ts2):
    """
    Compute the DTW distance between two sequences, using appropriate distance metric
    for 1D or 2D input.

    Parameters:
    - ts1, ts2: np.ndarray
        Time series arrays of shape (time,) or (features, time)

    Returns:
    - dist: float
        The DTW distance between ts1 and ts2
    """
    assert ts1.ndim == ts2.ndim, f"x_series has shape {ts1.shape}\ny_series has shape {ts2.shape}"

    if ts1.ndim == 1:
        # 1D signals (scalar per frame)
        dist, _ = fastdtw(ts1, ts2, dist=lambda u, v: abs(u - v))
    else:
        # 2D signals (vector per frame) — transpose so DTW sees [time][features]
        ts1 = ts1.T if ts1.shape[0] < ts1.shape[1] else ts1
        ts2 = ts2.T if ts2.shape[0] < ts2.shape[1] else ts2
        dist, _ = fastdtw(ts1, ts2, dist=euclidean)

    return dist


def extract_mfcc(signal, sample_rate, n_mfcc=13, max_len=300):
    """
    Extract MFCC features from a 1D time-series audio signal.

    Parameters:
    - signal: np.ndarray, audio signal (mono)
    - sample_rate: int, sampling rate of the signal
    - n_mfcc: int, number of MFCC features to extract
    - max_len: int, max number of time frames (pad or truncate)

    Returns:
    - mfcc_feat: np.ndarray of shape (n_mfcc, max_len)
    """
    mfcc = librosa.feature.mfcc(y=signal.astype(float), sr=sample_rate, n_mfcc=n_mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def extract_mfcc_librosa(file_path, n_mfcc=13, max_len=300):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc