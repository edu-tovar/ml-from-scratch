"""Feature preprocessing."""
import numpy as np


def one_hot_encode(X, categorical_columns, drop_first=False):
    """Replace each listed column of X by one indicator column per category.

    The indicator columns are inserted where the original column was, in the
    order of the sorted unique values. Works with string categories (X is
    returned with dtype object in that case).
    """
    X = np.asarray(X, dtype=object)
    # Process columns from right to left so earlier indices stay valid.
    for col in sorted(categorical_columns, reverse=True):
        values = X[:, col]
        categories = np.unique(values)
        if drop_first:
            categories = categories[1:]
        one_hot = np.column_stack([(values == c).astype(float) for c in categories])
        X = np.concatenate([X[:, :col], one_hot, X[:, col + 1:]], axis=1)
    try:
        return X.astype(float)
    except (TypeError, ValueError):
        return X
