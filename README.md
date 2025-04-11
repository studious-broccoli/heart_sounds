# heart_sounds
Detect abnormal heart sounds

heart_sounds_project/
│
├── data/
│   ├── load_data.py            # Handles all data loading and preprocessing
│   └── config.py               # Constants like file paths and label mappings
│
├── models/
│   └── knn_dtw.py              # Contains the KnnDtw class
│
├── utils/
│   └── audio_utils.py          # Utilities like pcm2float, plotting, etc.
│
├── notebooks/
│   └── main.ipynb              # Jupyter notebook for running/visualizing results
│
├── results/
│   └── confusion_matrix.png    # Output figures like confusion matrix
│
├── main.py                     # Script version of your pipeline
└── requirements.txt            # List of dependencies
