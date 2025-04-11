import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def pcm2float(sig, dtype=np.float64):
    sig = np.asarray(sig)
    assert sig.dtype.kind == 'i'
    return sig.astype(dtype) / dtype.type(-np.iinfo(sig.dtype).min)

def plot_signals(a, b, title='DTW Comparison', sample_rate=500):
    time = np.linspace(0, 20, sample_rate)
    amplitude_a = a[:sample_rate]
    amplitude_b = b[:sample_rate]
    distance, _ = fastdtw(amplitude_a, amplitude_b, radius=1, dist=euclidean)
    plt.figure(figsize=(12, 4))
    plt.plot(time, amplitude_a, label='Signal A')
    plt.plot(time, amplitude_b, label='Signal B')
    plt.title(f"{title} | DTW Distance: {distance:.2f}")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
