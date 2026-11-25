import numpy as np
import pytest
from sklearn import linear_model as sk
from sklearn.datasets import load_breast_cancer, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler

from mlscratch import LinearRegression, LogisticRegression, simple_linear_regression


def test_simple_regression_matches_polyfit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = 3 * x - 2 + rng.normal(scale=0.1, size=100)
    slope, intercept = simple_linear_regression(x, y)
    expected_slope, expected_intercept = np.polyfit(x, y, 1)
    assert slope == pytest.approx(expected_slope)
    assert intercept == pytest.approx(expected_intercept)


def test_simple_regression_rejects_constant_x():
    with pytest.raises(ValueError):
        simple_linear_regression(np.ones(5), np.arange(5))


def test_normal_equation_matches_sklearn():
    X, y = load_diabetes(return_X_y=True)
    ours = LinearRegression().fit(X, y)
    ref = sk.LinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-6, atol=1e-6)
    assert ours.intercept_ == pytest.approx(ref.intercept_)


def test_gradient_descent_converges_to_normal_equation():
    X, y = make_regression(n_samples=200, n_features=4, noise=5, random_state=0)
    X = StandardScaler().fit_transform(X)
    gd = LinearRegression(method="gradient_descent", learning_rate=0.05, n_iterations=2000, random_state=0).fit(X, y)
    exact = LinearRegression().fit(X, y)
    np.testing.assert_allclose(gd.coef_, exact.coef_, rtol=1e-4)
    assert gd.loss_history_[-1] < gd.loss_history_[0]


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        LinearRegression().predict(np.zeros((2, 2)))


def _breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def test_logistic_regression_close_to_sklearn():
    X, y = _breast_cancer()
    ours = LogisticRegression(learning_rate=0.5, n_iterations=5000).fit(X, y)
    # sklearn's C is the inverse regularisation strength: a huge C means almost none.
    ref = sk.LogisticRegression(C=1e6, max_iter=10000).fit(X, y)
    agreement = np.mean(ours.predict(X) == ref.predict(X))
    assert agreement > 0.98
    assert ours.score(X, y) > 0.97


def test_logistic_loss_decreases():
    X, y = _breast_cancer()
    model = LogisticRegression(n_iterations=300).fit(X, y)
    assert np.all(np.diff(model.loss_history_) <= 1e-12)


@pytest.mark.parametrize("penalty", ["l1", "l2", "elasticnet"])
def test_regularisation_shrinks_weights(penalty):
    X, y = _breast_cancer()
    plain = LogisticRegression(n_iterations=2000).fit(X, y)
    reg = LogisticRegression(penalty=penalty, alpha=200.0, n_iterations=2000).fit(X, y)
    assert np.linalg.norm(reg.coef_) < np.linalg.norm(plain.coef_)


def test_l1_drives_weights_towards_zero():
    X, y = _breast_cancer()
    l1 = LogisticRegression(penalty="l1", alpha=500.0, n_iterations=3000).fit(X, y)
    l2 = LogisticRegression(penalty="l2", alpha=500.0, n_iterations=3000).fit(X, y)
    assert np.sum(np.abs(l1.coef_) < 1e-2) > np.sum(np.abs(l2.coef_) < 1e-2)


def test_unknown_penalty_raises():
    with pytest.raises(ValueError):
        LogisticRegression(penalty="l3")
