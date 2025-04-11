from data.load_data import load_labels, load_wav_files
from models.knn_dtw import KnnDtw
from data.config import *
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load Data
x_train, _ = load_wav_files(TRAIN_AUDIO_PATH, 20)
x_test, _ = load_wav_files(TEST_AUDIO_PATH, 5)
y_train = load_labels(TRAIN_LABELS_PATH, 20)
y_test = load_labels(TEST_LABELS_PATH, 5)

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
conf_mat = confusion_matrix(y_test, y_pred)
plt.imshow(conf_mat, cmap='summer')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(2), list(PCG_LABELS.values()), rotation=90)
plt.yticks(range(2), list(PCG_LABELS.values()))
plt.show()
