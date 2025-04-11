# config.py

TRAIN_LABELS_PATH = "/Users/ariannapryor/Documents/studious-broccoli/data_heart_sounds/filter_signals/LabelEx.csv"
TEST_LABELS_PATH = "/Users/ariannapryor/Documents/studious-broccoli/data_heart_sounds/filter_signals/TestA/LabelExTest.csv"
TRAIN_AUDIO_PATH = "/Users/ariannapryor/Documents/studious-broccoli/data_heart_sounds/filter_signals/TrainSetA/*.wav"
TEST_AUDIO_PATH = "/Users/ariannapryor/Documents/studious-broccoli/data_heart_sounds/filter_signals/TestA/*.wav"

PCG_LABELS = {-1: 'ABNORMAL', 1: 'NORMAL'}

DATA_TYPE = "MFC_COEFFICIENTS"  # "RAW_SIGNAL", "MFC_COEFFICIENTS"
MODEL_TYPE = "CNN"  # "KNN", "CNN"

N_train = 1000
N_test = 100

NUM_EPOCHS = 20

int_to_label = {idx: name for name, idx in {'ABNORMAL': -1, 'NORMAL': 1}.items()}
int_to_label_01 = {idx: name for name, idx in {'ABNORMAL': 0, 'NORMAL': 1}.items()}