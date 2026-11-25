"""Compare each mlscratch model with its scikit-learn counterpart on held-out data.

Writes benchmarks/results.md and the figures in figures/.

Usage:
    python benchmarks/compare_sklearn.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import linear_model as sk_lm  # noqa: E402
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve as sk_roc_curve  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mlscratch import LDA, KNNClassifier, LinearRegression, LogisticRegression, k_fold_cross_validation  # noqa: E402
from mlscratch.metrics import auc, r2_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def split_scaled(X, y, stratify=True):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y if stratify else None)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te


rows = []

# Linear regression: diabetes (442 samples, 10 features)
X_tr, X_te, y_tr, y_te = split_scaled(*load_diabetes(return_X_y=True), stratify=False)
ref = sk_lm.LinearRegression().fit(X_tr, y_tr)
ne = LinearRegression().fit(X_tr, y_tr)
gd = LinearRegression(method="gradient_descent", learning_rate=0.05, n_iterations=3000, random_state=0).fit(X_tr, y_tr)
for name, model in [("Linear regression (normal equation)", ne), ("Linear regression (gradient descent)", gd)]:
    rows.append((name, "Diabetes", "test R²", r2_score(y_te, model.predict(X_te)), r2_score(y_te, ref.predict(X_te)),
                 f"max coef. difference {np.max(np.abs(model.coef_ - ref.coef_)):.1e}"))

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(gd.loss_history_)
ax.axhline(np.mean((ref.predict(X_tr) - y_tr) ** 2), color="r", ls="--", label="normal equation")
ax.set_xlabel("Iteration")
ax.set_ylabel("Training MSE")
ax.set_title("Gradient descent convergence (diabetes)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "gd_convergence.png", dpi=120)

# Logistic regression: breast cancer (569 samples, 30 features)
X_tr, X_te, y_tr, y_te = split_scaled(*load_breast_cancer(return_X_y=True))
ours = LogisticRegression(learning_rate=0.5, n_iterations=5000).fit(X_tr, y_tr)
ref = sk_lm.LogisticRegression(C=1e6, max_iter=10000).fit(X_tr, y_tr)
p_ours, p_ref = ours.predict_proba(X_te), ref.predict_proba(X_te)[:, 1]
rows.append(("Logistic regression", "Breast cancer", "test accuracy", ours.score(X_te, y_te), ref.score(X_te, y_te),
             f"prediction agreement {np.mean(ours.predict(X_te) == ref.predict(X_te)):.1%}"))
fpr, tpr, _ = roc_curve(y_te, p_ours)
rows.append(("Logistic regression", "Breast cancer", "test ROC AUC", auc(fpr, tpr), roc_auc_score(y_te, p_ref), ""))

ref_fpr, ref_tpr, _ = sk_roc_curve(y_te, p_ref)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, label=f"mlscratch (AUC {auc(fpr, tpr):.3f})")
ax.plot(ref_fpr, ref_tpr, ls="--", label=f"scikit-learn (AUC {roc_auc_score(y_te, p_ref):.3f})")
ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Logistic regression ROC (breast cancer, test set)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURES / "logistic_roc.png", dpi=120)

# k-NN and LDA: wine (178 samples, 13 features, 3 classes)
X_tr, X_te, y_tr, y_te = split_scaled(*load_wine(return_X_y=True))
ours = KNNClassifier(k=5).fit(X_tr, y_tr)
ref = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
rows.append(("k-NN (k = 5)", "Wine", "test accuracy", ours.score(X_te, y_te), ref.score(X_te, y_te),
             f"prediction agreement {np.mean(ours.predict(X_te) == ref.predict(X_te)):.1%}"))

lda = LDA().fit(X_tr, y_tr)
ref = LinearDiscriminantAnalysis(solver="eigen").fit(X_tr, y_tr)
rows.append(("LDA", "Wine", "test accuracy", lda.score(X_te, y_te), ref.score(X_te, y_te),
             "nearest projected class mean vs. Gaussian Bayes rule"))

Z = lda.transform(np.vstack([X_tr, X_te]))
labels = np.concatenate([y_tr, y_te])
fig, ax = plt.subplots(figsize=(6, 5))
for c in lda.classes_:
    ax.scatter(Z[labels == c, 0], Z[labels == c, 1], s=15, alpha=0.7, label=f"class {c}")
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_title("Wine data projected on the two LDA directions")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "lda_projection.png", dpi=120)

# Cross-validation: 5-fold accuracy of a k-NN on breast cancer (first 565 rows, divisible by 5)
X, y = load_breast_cancer(return_X_y=True)
X, y = X[:565], y[:565]
ours_mean, ours_std = k_fold_cross_validation(KNeighborsClassifier(5), X, y, n_folds=5)
ref_scores = cross_val_score(KNeighborsClassifier(5), X, y, cv=KFold(5))
rows.append(("k-fold cross-validation", "Breast cancer", "mean CV accuracy", ours_mean, ref_scores.mean(),
             f"std {ours_std:.4f} vs {ref_scores.std():.4f}"))

lines = ["| Model | Dataset | Metric | mlscratch | scikit-learn | Notes |", "|---|---|---|---|---|---|"]
lines += [f"| {m} | {d} | {k} | {a:.4f} | {b:.4f} | {n} |" for m, d, k, a, b, n in rows]
table = "\n".join(lines)
(ROOT / "benchmarks" / "results.md").write_text(table + "\n", encoding="utf-8", newline="\n")
print(table)
