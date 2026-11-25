"""Feature preprocessing."""
import numpy as np


def one_hot_encode(X, categorical_indices, drop_first=False):
    """Replace each listed column of X by one indicator column per category.

    The indicator columns are inserted where the original column was, in the
    order of the sorted unique values. Works with string categories.
    """
    X_transformed = X.copy()
    # Right to left, so the indices of the columns still to process do not shift.
    for index in sorted(categorical_indices, reverse=True):
        categorical_column = X_transformed[:, index]
        unique_values = np.unique(categorical_column)

        one_hot = np.zeros((categorical_column.shape[0], unique_values.shape[0]))
        for i, value in enumerate(unique_values):
            one_hot[:, i] = (categorical_column == value).astype(float)
        if drop_first:
            one_hot = one_hot[:, 1:]

        X_transformed = np.delete(X_transformed, index, axis=1)
        position = index
        for column in one_hot.T:
            X_transformed = np.insert(X_transformed, position, column, axis=1)
            position += 1
    return X_transformed
