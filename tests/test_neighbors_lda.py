import numpy as np
import pytest
from scipy.spatial.distance import minkowski
from sklearn.datasets import load_iris, load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from mlscratch import LDA, KNNClassifier, minkowski_distance


@pytest.mark.parametrize("p", [1, 2, 3])
def test_minkowski_matches_scipy(p):
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=5), rng.normal(size=5)
    assert minkowski_distance(a, b, p) == pytest.approx(minkowski(a, b, p))


def _wine_split():
    X, y = load_wine(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te


@pytest.mark.parametrize("k,p", [(1, 2), (5, 2), (7, 1)])
def test_knn_matches_sklearn(k, p):
    X_tr, X_te, y_tr, y_te = _wine_split()
    ours = KNNClassifier(k=k, p=p).fit(X_tr, y_tr).predict(X_te)
    ref = KNeighborsClassifier(n_neighbors=k, p=p).fit(X_tr, y_tr).predict(X_te)
    np.testing.assert_array_equal(ours, ref)


def test_knn_probabilities_sum_to_one():
    X_tr, X_te, y_tr, _ = _wine_split()
    proba = KNNClassifier(k=5).fit(X_tr, y_tr).predict_proba(X_te)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba.shape == (X_te.shape[0], 3)


def test_knn_rejects_bad_parameters():
    with pytest.raises(ValueError):
        KNNClassifier(k=0)


def test_lda_projection_spans_same_subspace_as_sklearn():
    X, y = load_iris(return_X_y=True)
    ours = LDA().fit(X, y).transform(X)
    ref = LinearDiscriminantAnalysis(solver="eigen").fit(X, y).transform(X)
    # Discriminant directions are defined up to scale and sign: compare by correlation.
    for j in range(ref.shape[1]):
        corr = np.corrcoef(ours[:, j], ref[:, j])[0, 1]
        assert abs(corr) > 0.999


def test_lda_classifies_iris_and_wine():
    X, y = load_iris(return_X_y=True)
    assert LDA().fit(X, y).score(X, y) > 0.95
    X_tr, X_te, y_tr, y_te = _wine_split()
    assert LDA().fit(X_tr, y_tr).score(X_te, y_te) > 0.95


def test_lda_default_components():
    X, y = load_iris(return_X_y=True)
    assert LDA().fit(X, y).n_components_ == 2
