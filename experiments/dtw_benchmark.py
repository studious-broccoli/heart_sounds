# experiments/dtw_benchmark.py

import time
import matplotlib.pyplot as plt
import numpy as np
from models.knn_dtw import KnnDtw
from data.load_data import load_wav_files, load_labels
from data.config import TRAIN_AUDIO_PATH, TEST_AUDIO_PATH, TRAIN_LABELS_PATH, TEST_LABELS_PATH

# Load a subset of the training/testing data for the benchmark
x_train, _ = load_wav_files(TRAIN_AUDIO_PATH, 20)
y_train = load_labels(TRAIN_LABELS_PATH, 20)
x_test, _ = load_wav_files(TEST_AUDIO_PATH, 20)
y_test = load_labels(TEST_LABELS_PATH, 20)

windows = [1, 2, 5, 10, 50, 100, 500, 1000, 5000]
time_taken = []

for w in windows:
    start = time.time()
    model = KnnDtw(n_neighbors=1, max_warping_window=w, show_progress=False)
    model.fit(x_train, y_train)
    # Run prediction on the testing data (using the same data for timing purposes)
    model.predict(x_test)
    end = time.time()
    time_taken.append(end - start)

# Plot the DTW execution time experiment results
plt.figure(figsize=(12, 5))
plt.plot(windows, [t / 400.0 for t in time_taken], lw=4)
plt.title('DTW Execution Time with varying Max Warping Window')
plt.ylabel('Execution Time (seconds)')
plt.xlabel('Max Warping Window')
plt.xscale('log')
plt.show()
