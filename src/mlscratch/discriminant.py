"""Fisher's linear discriminant analysis."""
import numpy as np


class LDA:
    """Linear discriminant analysis for dimensionality reduction and classification.

    The projection directions are the leading eigenvectors of S_W^-1 S_B,
    where S_W and S_B are the within- and between-class scatter matrices.
    New samples are assigned to the class whose projected mean is closest.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        n_components = self.n_components or min(n_features, len(self.classes_) - 1)
        self.n_components_ = min(n_components, n_features, len(self.classes_) - 1)

        overall_mean = X.mean(axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in self.classes_:
            X_c = X[y == c]
            mean_c = X_c.mean(axis=0)
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            diff = (mean_c - overall_mean).reshape(-1, 1)
            S_B += X_c.shape[0] * diff @ diff.T

        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        # S_W^-1 S_B is not symmetric, so eig can return complex numbers with
        # zero imaginary part: keep the real part.
        eigen_pairs = sorted(zip(eigen_values.real, eigen_vectors.real.T), key=lambda pair: pair[0], reverse=True)
        self.weights_ = np.array([vector for _, vector in eigen_pairs[: self.n_components_]])

        self.class_means_ = np.array([self.transform(X[y == c]).mean(axis=0) for c in self.classes_])
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.weights_.T

    def predict(self, X):
        Z = self.transform(X)
        distances = np.array([np.linalg.norm(Z - mean, axis=1) for mean in self.class_means_])
        return self.classes_[np.argmin(distances, axis=0)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
