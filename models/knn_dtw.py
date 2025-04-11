# models/knn_dtw.py
import pdb
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.stats import mode
import numpy as np
from tqdm import tqdm
from utils.dtw_tools import dtw_distance


class KnnDtw:
    """
    k-Nearest Neighbor classifier using Dynamic Time Warping (DTW)
    as the distance measure.

    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        Number of neighbors to use for the classifier.
    max_warping_window : int, optional (default=10000)
        Currently not used directly with fastdtw.
    subsample_step : int, optional (default=1)
        Step size for the time series downsampling.
    show_progress : bool, optional (default=False)
        Show a progress bar during distance computation.
    """

    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1, show_progress=True):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
        self.show_progress = show_progress

    def fit(self, x, l):
        """Store training data and labels."""
        self.x = x
        self.l = l

    def _preprocess_series(self, s):
        """Convert to mono if stereo, then subsample."""
        if s.ndim > 1:
            s = np.mean(s, axis=1)
        return s[::self.subsample_step]

    def _dist_matrix(self, x, y):
        """Compute full DTW distance matrix between x and y."""
        n, m = len(x), len(y)
        dm = np.zeros((n, m))

        iter_range = tqdm(range(n), desc="Computing DTW", disable=not self.show_progress)

        for i in iter_range:
            x_series = self._preprocess_series(x[i])
            for j in range(m):
                y_series = self._preprocess_series(y[j])
                dm[i, j] = dtw_distance(x_series, y_series)

        return dm

    def predict(self, x):
        """Predict class labels and probabilities for given test data."""
        dm = self._dist_matrix(x, self.x)
        knn_idx = dm.argsort()[:, :self.n_neighbors]
        knn_labels = self.l[knn_idx]
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors
        return mode_label.ravel(), mode_proba.ravel()
