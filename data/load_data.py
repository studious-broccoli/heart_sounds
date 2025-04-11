from scipy.io import wavfile
import numpy as np
import pandas as pd
import glob
from .config import *
from utils.dtw_tools import extract_mfcc
from pathlib import Path
import glob


def load_labels(path, count):
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, args=('coerce',))
    return df.loc[:, 1].values[:count]


def load_wav_files(pattern, count):
    files = sorted(glob.glob(pattern))[:count]
    sigs = []
    rates = []
    for f in files:
        fs, sig = wavfile.read(f)
        if sig.ndim > 1:
            sig = np.mean(sig, axis=1)  # Convert stereo to mono
        rates.append(fs)
        sigs.append(sig)
    return np.array(sigs)


def load_wav_files_mfcc(pattern, count, n_mfcc=13, max_len=300):
    files = sorted(glob.glob(pattern))[:count]
    sigs, mfccs = [], []

    for f in files:
        try:
            fs, signal = wavfile.read(f)
            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)

            mfcc = extract_mfcc(signal, fs, n_mfcc=n_mfcc, max_len=max_len)
            sigs.append(signal)
            mfccs.append(mfcc)

        except Exception as e:
            print(f"[Warning] Skipped {f} due to error: {e}")

    return np.stack(mfccs)

