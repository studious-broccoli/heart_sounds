import pdb
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.dtw_tools import dtw_distance


def pcm2float(sig, dtype=np.float64):
    """Convert PCM int signal to float."""
    sig = np.asarray(sig)
    assert sig.dtype.kind == 'i', "'sig' must be signed integer PCM data"
    return sig.astype(dtype) / dtype.type(-np.iinfo(sig.dtype).min)


def plot_signals(a, b, filename, title='DTW Comparison', sample_rate=500):
    """Plot and save waveform comparison with DTW distance."""
    time = np.linspace(0, 20, sample_rate)
    amplitude_a = a[:sample_rate]
    amplitude_b = b[:sample_rate]

    # Optional debug:
    # pdb.set_trace()

    distance = dtw_distance(amplitude_a, amplitude_b)

    plt.figure(figsize=(12, 4))
    plt.plot(time, amplitude_a, label='Signal A')
    plt.plot(time, amplitude_b, label='Signal B')
    plt.title(f"{title} | DTW Distance: {distance:.2f}")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_mfcc_plot(mfcc, filename="mfcc_plot.png", title="MFCC", sr=22050):
    """Plot and save MFCC as a heatmap."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time Frames')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def animate_mfcc(mfcc, filename, title="MFCC Animation"):
    """Animate MFCC over time. Optionally save as GIF/MP4 if filename provided."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(title)
    img = ax.imshow(mfcc[:, :1], aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("MFCC Coefficients")

    def update(frame):
        img.set_array(mfcc[:, :frame+1])
        ax.set_xlim(0, mfcc.shape[1])
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=mfcc.shape[1], interval=30, blit=True)
    ani.save(filename, writer='pillow', fps=30)  # requires pillow for GIF
    plt.close()


def plot_mfcc_difference(ts1, ts2, filename, title="MFCC Difference"):
    """Plot absolute difference between two MFCCs and save."""
    assert ts1.shape == ts2.shape, f"Shape mismatch: {ts1.shape} vs {ts2.shape}"
    diff = np.abs(ts1 - ts2)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(diff, x_axis='time', cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
