"""Linear and logistic regression implemented with NumPy."""
import numpy as np


def simple_linear_regression(x, y):
    """Closed-form least squares for one predictor.

    slope = cov(x, y) / var(x),  intercept = mean(y) - slope * mean(x)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x_mean, y_mean = x.mean(), y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        raise ValueError("x has zero variance")
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    return float(slope), float(y_mean - slope * x_mean)


class LinearRegression:
    """Ordinary least squares, solved with the normal equation or gradient descent.

    Parameters
    ----------
    method : {"normal", "gradient_descent"}
    learning_rate, n_iterations : used only by gradient descent.
    """

    def __init__(self, method="normal", learning_rate=0.01, n_iterations=1000, random_state=None):
        if method not in ("normal", "gradient_descent"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if self.method == "normal":
            self._fit_normal(X, y)
        else:
            self._fit_gradient_descent(X, y)
        return self

    def _fit_normal(self, X, y):
        # theta = (X^T X)^+ X^T y, with a column of ones for the intercept.
        # The pseudo-inverse also handles collinear columns.
        Xb = np.column_stack([np.ones(X.shape[0]), X])
        theta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]

    def _fit_gradient_descent(self, X, y):
        # Minimise the MSE: gradients are (2/m) * sum(error) and (2/m) * X^T error.
        rng = np.random.default_rng(self.random_state)
        m = X.shape[0]
        self.coef_ = rng.random(X.shape[1]) * 0.01
        self.intercept_ = rng.random() * 0.01
        self.loss_history_ = []
        for _ in range(self.n_iterations):
            error = self.intercept_ + X @ self.coef_ - y
            self.loss_history_.append(float(np.mean(error ** 2)))
            self.intercept_ -= self.learning_rate * (2 / m) * np.sum(error)
            self.coef_ -= self.learning_rate * (2 / m) * (X.T @ error)
        self.intercept_ = float(self.intercept_)

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("model is not fitted")
        return self.intercept_ + _as_2d(X) @ self.coef_

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))


class LogisticRegression:
    """Binary logistic regression trained with batch gradient descent.

    Parameters
    ----------
    penalty : {None, "lasso", "ridge", "elasticnet"}
    alpha : float
        Regularisation strength (larger means stronger). Note that this is the
        inverse of scikit-learn's ``C``.
    l1_ratio : float
        Mix between L1 (1.0) and L2 (0.0); only used with ``penalty="elasticnet"``.
    """

    def __init__(self, penalty=None, alpha=1.0, l1_ratio=0.5, learning_rate=0.1, n_iterations=1000):
        if penalty not in (None, "lasso", "ridge", "elasticnet"):
            raise ValueError(f"unknown penalty {penalty!r}")
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def log_loss(y, p, eps=1e-15):
        """Negative log-likelihood averaged over samples."""
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _penalty_gradient(self, m):
        w = self.coef_
        l1 = (self.alpha / m) * np.sign(w)
        l2 = (self.alpha / m) * w
        if self.penalty == "lasso":
            return l1
        if self.penalty == "ridge":
            return l2
        if self.penalty == "elasticnet":
            return self.l1_ratio * l1 + (1 - self.l1_ratio) * l2
        return 0.0

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        m, n = X.shape
        self.coef_ = np.zeros(n)
        self.intercept_ = 0.0
        self.loss_history_ = []
        for _ in range(self.n_iterations):
            p = self.predict_proba(X)
            self.loss_history_.append(self.log_loss(y, p))
            error = p - y
            # Gradient of the negative log-likelihood, plus the penalty term
            # (the intercept is not regularised).
            dw = (1 / m) * X.T @ error + self._penalty_gradient(m)
            db = (1 / m) * np.sum(error)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        return self

    def predict_proba(self, X):
        """Probability of the positive class."""
        return self.sigmoid(_as_2d(X) @ self.coef_ + self.intercept_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def _as_2d(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X
