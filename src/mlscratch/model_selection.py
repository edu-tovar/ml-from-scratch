"""Model evaluation by cross-validation."""
import numpy as np


def k_fold_cross_validation(model, X, y, n_folds=5):
    """Mean and standard deviation of ``model.score`` over k contiguous folds.

    Folds are consecutive blocks of rows (no shuffling); when the number of
    rows is not divisible by ``n_folds`` the remaining rows are always used for
    training. ``n_folds = len(X)`` gives leave-one-out cross-validation.
    """
    X, y = np.asarray(X), np.asarray(y)
    n = X.shape[0]
    if not 2 <= n_folds <= n:
        raise ValueError("n_folds must be between 2 and the number of samples")
    fold_size = n // n_folds
    scores = []
    for i in range(n_folds):
        valid = np.arange(i * fold_size, (i + 1) * fold_size)
        train = np.concatenate([np.arange(0, i * fold_size), np.arange((i + 1) * fold_size, n)])
        model.fit(X[train], y[train])
        scores.append(model.score(X[valid], y[valid]))
    return float(np.mean(scores)), float(np.std(scores))
