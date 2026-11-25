"""k-nearest neighbours classifier."""
import numpy as np


def minkowski_distance(a, b, p=2):
    """(sum |a_i - b_i|^p)^(1/p). p=1 is Manhattan, p=2 Euclidean."""
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b)) ** p) ** (1 / p))


class KNNClassifier:
    """Majority vote among the k nearest training points (Minkowski distance)."""

    def __init__(self, k=5, p=2):
        if k <= 0 or p <= 0:
            raise ValueError("k and p must be positive")
        self.k = k
        self.p = p

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.X_train_, self.y_train_ = X, y
        self.classes_ = np.unique(y)
        return self

    def _distances(self, point):
        # Distance from one point to every training point (vectorised).
        return np.sum(np.abs(self.X_train_ - point) ** self.p, axis=1) ** (1 / self.p)

    def _neighbour_labels(self, point):
        nearest = np.argsort(self._distances(point), kind="stable")[: self.k]
        return self.y_train_[nearest]

    @staticmethod
    def _most_common(labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._most_common(self._neighbour_labels(x)) for x in X])

    def predict_proba(self, X):
        """Share of the k neighbours in each class (columns follow ``classes_``)."""
        X = np.asarray(X, dtype=float)
        return np.array([
            [np.mean(self._neighbour_labels(x) == c) for c in self.classes_] for x in X
        ])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
