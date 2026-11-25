"""Regression and classification metrics."""
import numpy as np


# ---------------------------------------------------------------- regression

def r2_score(y_true, y_pred):
    """1 - RSS / TSS."""
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - rss / tss) if tss != 0 else 0.0


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((_flat(y_true) - _flat(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(_flat(y_true) - _flat(y_pred))))


# ---------------------------------------------------------------- classification

def binary_classification_metrics(y_true, y_pred, positive_label=1):
    """Confusion matrix counts and the metrics derived from them."""
    t = np.asarray(y_true) == positive_label
    p = np.asarray(y_pred) == positive_label
    tp, tn = int(np.sum(t & p)), int(np.sum(~t & ~p))
    fp, fn = int(np.sum(~t & p)), int(np.sum(t & ~p))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def roc_curve(y_true, scores, positive_label=1, thresholds=None):
    """False and true positive rates when predicting positive for score >= threshold.

    By default every distinct score is used as a threshold, plus +inf so the
    curve starts at (0, 0).
    """
    t = np.asarray(y_true) == positive_label
    scores = np.asarray(scores, dtype=float)
    if thresholds is None:
        thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    fpr, tpr = [], []
    for thr in thresholds:
        pred = scores >= thr
        tpr.append(_safe_div(np.sum(t & pred), np.sum(t)))
        fpr.append(_safe_div(np.sum(~t & pred), np.sum(~t)))
    return np.array(fpr), np.array(tpr), np.asarray(thresholds)


def auc(x, y):
    """Area under a curve with the trapezoidal rule (x must be monotonic)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(abs(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)))


def calibration_curve(y_true, probs, positive_label=1, n_bins=10):
    """Fraction of positives in equal-width probability bins.

    Returns the bin centres and the observed fraction of positives (NaN for empty bins).
    """
    t = (np.asarray(y_true) == positive_label).astype(float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    fraction = np.array([t[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    return centres, fraction


def _flat(a):
    return np.asarray(a, dtype=float).ravel()


def _safe_div(num, den):
    return float(num / den) if den != 0 else 0.0
