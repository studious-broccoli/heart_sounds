from data.load_data import load_wav_files, load_wav_files_mfcc
from data.config import *
from utils.audio_utils import plot_signals, save_mfcc_plot, animate_mfcc, plot_mfcc_difference
from train_cnn import train_cnn
from train_knn import train_knn


# Load Data
if DATA_TYPE == "RAW_SIGNAL":
    time_series = load_wav_files(TRAIN_AUDIO_PATH, count=N_train)
    plot_signals(time_series[0], time_series[1],
                 filename=f"./results/train_signals_sample.png",
                 title='DTW Comparison', sample_rate=500)

elif DATA_TYPE == "MFC_COEFFICIENTS" and MODEL_TYPE != "CNN":
    time_series = load_wav_files_mfcc(TRAIN_AUDIO_PATH, count=N_train)
    ts1, ts2 = time_series[0], time_series[1]
    save_mfcc_plot(ts1, filename="./results/signal1_mfcc.png", title="MFCC - Signal 1")
    save_mfcc_plot(ts2, filename="./results/signal2_mfcc.png", title="MFCC - Signal 2")
    animate_mfcc(ts1, filename="./results/animate_mfcc.png", title="Signal 1 - MFCC Animation")
    plot_mfcc_difference(ts1, ts2,
                         filename="./results/mfcc_diff.png", title="MFCC Difference - Signal 1 vs 2")


if MODEL_TYPE == "KNN":
    train_knn()
elif MODEL_TYPE == "CNN":
    train_cnn()
