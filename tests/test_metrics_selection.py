import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

from mlscratch import k_fold_cross_validation
from mlscratch.metrics import (auc, binary_classification_metrics, calibration_curve, mae, r2_score,
                               rmse, roc_curve)
from mlscratch.preprocessing import one_hot_encode

rng = np.random.default_rng(42)


def test_regression_metrics_match_sklearn():
    y, y_hat = rng.normal(size=50), rng.normal(size=50)
    assert r2_score(y, y_hat) == pytest.approx(skm.r2_score(y, y_hat))
    assert rmse(y, y_hat) == pytest.approx(np.sqrt(skm.mean_squared_error(y, y_hat)))
    assert mae(y, y_hat) == pytest.approx(skm.mean_absolute_error(y, y_hat))


def test_classification_metrics_match_sklearn():
    y, y_hat = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    m = binary_classification_metrics(y, y_hat)
    tn, fp, fn, tp = skm.confusion_matrix(y, y_hat).ravel()
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (tn, fp, fn, tp)
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, y_hat))
    assert m["precision"] == pytest.approx(skm.precision_score(y, y_hat))
    assert m["recall"] == pytest.approx(skm.recall_score(y, y_hat))
    assert m["specificity"] == pytest.approx(skm.recall_score(y, y_hat, pos_label=0))
    assert m["f1"] == pytest.approx(skm.f1_score(y, y_hat))


def test_metrics_with_string_labels():
    m = binary_classification_metrics(["a", "b", "a"], ["a", "a", "a"], positive_label="a")
    assert (m["tp"], m["fp"]) == (2, 1)


def test_roc_auc_matches_sklearn():
    y = rng.integers(0, 2, 300)
    scores = y * 0.5 + rng.random(300)
    fpr, tpr, _ = roc_curve(y, scores)
    assert auc(fpr, tpr) == pytest.approx(skm.roc_auc_score(y, scores))


def test_calibration_matches_sklearn_on_non_empty_bins():
    y = rng.integers(0, 2, 500)
    probs = rng.random(500)
    centres, frac = calibration_curve(y, probs, n_bins=5)
    ref_true, _ = sk_calibration_curve(y, probs, n_bins=5, strategy="uniform")
    np.testing.assert_allclose(frac[~np.isnan(frac)], ref_true)
    np.testing.assert_allclose(centres, [0.1, 0.3, 0.5, 0.7, 0.9])


def test_k_fold_matches_sklearn_kfold():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:560], y[:560]  # divisible by 5 and 10, so the folds are identical
    model = KNeighborsClassifier(n_neighbors=5)
    for k in (5, 10):
        mean, std = k_fold_cross_validation(model, X, y, n_folds=k)
        ref = cross_val_score(model, X, y, cv=KFold(n_splits=k))
        assert mean == pytest.approx(ref.mean())
        assert std == pytest.approx(ref.std())


def test_leave_one_out_runs():
    X, y = load_breast_cancer(return_X_y=True)
    mean, _ = k_fold_cross_validation(KNeighborsClassifier(3), X[:60], y[:60], n_folds=60)
    assert 0 <= mean <= 1


def test_one_hot_matches_sklearn():
    X = np.array([[1.0, "red", 3.0], [2.0, "blue", 1.0], [3.0, "red", 2.0]], dtype=object)
    ours = one_hot_encode(X, [1])
    ref = OneHotEncoder(sparse_output=False).fit_transform(X[:, [1]])
    np.testing.assert_allclose(ours[:, 1:3], ref)
    np.testing.assert_allclose(ours[:, [0, 3]], X[:, [0, 2]].astype(float))
    assert one_hot_encode(X, [1], drop_first=True).shape == (3, 3)
