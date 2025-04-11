from data.load_data import load_labels, load_wav_files, load_wav_files_mfcc
from data.config import *
from sklearn.metrics import classification_report
from utils.plot_utils import basic_cm_snsp
from models.knn_dtw import KnnDtw


def train_knn():
    # Import labels
    y_train = load_labels(TRAIN_LABELS_PATH, count=N_train)
    y_test = load_labels(TEST_LABELS_PATH, count=N_test)

    if DATA_TYPE == "RAW_SIGNAL":
        x_train = load_wav_files(TRAIN_AUDIO_PATH, count=N_train)
        x_test = load_wav_files(TEST_AUDIO_PATH, count=N_test)

    elif DATA_TYPE == "MFC_COEFFICIENTS" and MODEL_TYPE != "CNN":
        x_train = load_wav_files_mfcc(TRAIN_AUDIO_PATH, count=N_train)
        x_test = load_wav_files_mfcc(TEST_AUDIO_PATH, count=N_test)

    # Train & Predict
    model = KnnDtw(n_neighbors=3, max_warping_window=10000)
    model.fit(x_train, y_train)
    y_pred, proba = model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=PCG_LABELS.values()))
    # Confusion matrix
    basic_cm_snsp(y_test, y_pred, filename=f"./results/confusion_matrix_{DATA_TYPE}.png")


if __name__ == '__main__':
    train_knn()