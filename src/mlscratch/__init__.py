"""Classical machine learning algorithms implemented from scratch with NumPy."""
from .discriminant import LDA
from .linear_model import LinearRegression, LogisticRegression, simple_linear_regression
from .model_selection import k_fold_cross_validation
from .neighbors import KNNClassifier, minkowski_distance

__all__ = [
    "LDA",
    "KNNClassifier",
    "LinearRegression",
    "LogisticRegression",
    "k_fold_cross_validation",
    "minkowski_distance",
    "simple_linear_regression",
]
