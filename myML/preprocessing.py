import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        calculate mean and scale
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self

    def transform(self, X):
        assert X.ndim == 2, "Only suport dimension is 2"
        res = np.empty(X.shape, dtype=float)
        for i in range(X.shape[1]):
            res[:,i] = (X[:,i] - self.mean_[i]) / self.scale_[i]
        return res

