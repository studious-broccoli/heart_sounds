import pdb
from data.load_data import load_labels, load_wav_files, load_wav_files_mfcc
from models.knn_dtw import KnnDtw
from data.config import *
from sklearn.metrics import classification_report
from utils.plot_utils import basic_cm_snsp
from utils.audio_utils import plot_signals, save_mfcc_plot, animate_mfcc, plot_mfcc_difference

DATA_TYPE = "MFC_COEFFICIENTS"  # "RAW_SIGNAL", "MFC_COEFFICIENTS"

N_train = 1000
N_test = 100

# Import labels
y_train = load_labels(TRAIN_LABELS_PATH, count=N_train)
y_test = load_labels(TEST_LABELS_PATH, count=N_test)

# Load Data
if DATA_TYPE == "RAW_SIGNAL":
    x_train = load_wav_files(TRAIN_AUDIO_PATH, count=N_train)
    x_test = load_wav_files(TEST_AUDIO_PATH, count=N_test)

    # Plot Sample
    plot_signals(x_train[0], x_train[1],
                 filename=f"./results/train_signals_sample.png",
                 title='DTW Comparison', sample_rate=500)

elif DATA_TYPE == "MFC_COEFFICIENTS":
    x_train = load_wav_files_mfcc(TRAIN_AUDIO_PATH, count=N_train)
    x_test = load_wav_files_mfcc(TEST_AUDIO_PATH, count=N_test)
    ts1, ts2 = x_train[0], x_train[1]
    save_mfcc_plot(ts1, filename="./results/signal1_mfcc.png", title="MFCC - Signal 1")
    save_mfcc_plot(ts2, filename="./results/signal2_mfcc.png", title="MFCC - Signal 2")
    animate_mfcc(ts1, filename="./results/animate_mfcc.png", title="Signal 1 - MFCC Animation")
    plot_mfcc_difference(ts1, ts2,
                         filename="./results/mfcc_diff.png", title="MFCC Difference - Signal 1 vs 2")


# Train & Predict
model = KnnDtw(n_neighbors=3, max_warping_window=10000)
model.fit(x_train, y_train)
y_pred, proba = model.predict(x_test)

# Results
print("Predicted:", y_pred)
print("Actual:   ", y_test)
print("Probabilities:", proba)

print(classification_report(y_test, y_pred, target_names=PCG_LABELS.values()))

# Confusion matrix
basic_cm_snsp(y_test, y_pred, filename=f"./results/confusion_matrix_{DATA_TYPE}.png")
