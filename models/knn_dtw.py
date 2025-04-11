from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.stats import mode
import numpy as np
import scipy.spatial.distance

class KnnDtw:
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l):
        self.x = x
        self.l = l

    def _dist_matrix(self, x, y):
        dm = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                dm[i, j], _ = fastdtw(x[i, ::self.subsample_step], y[j, ::self.subsample_step], dist=euclidean)
        return dm

    def predict(self, x):
        dm = self._dist_matrix(x, self.x)
        knn_idx = dm.argsort()[:, :self.n_neighbors]
        knn_labels = self.l[knn_idx]
        mode_label, mode_proba = mode(knn_labels, axis=1)
        return mode_label.ravel(), (mode_proba / self.n_neighbors).ravel()
