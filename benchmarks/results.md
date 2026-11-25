| Model | Dataset | Metric | mlscratch | scikit-learn | Notes |
|---|---|---|---|---|---|
| Linear regression (normal equation) | Diabetes | test R² | 0.3929 | 0.3929 | max coef. difference 8.2e-13 |
| Linear regression (gradient descent) | Diabetes | test R² | 0.3924 | 0.3929 | max coef. difference 1.6e+00 |
| Logistic regression | Breast cancer | test accuracy | 0.9532 | 0.9532 | prediction agreement 98.8% |
| Logistic regression | Breast cancer | test ROC AUC | 0.9958 | 0.9812 |  |
| k-NN (k = 5) | Wine | test accuracy | 0.9630 | 0.9630 | prediction agreement 100.0% |
| LDA | Wine | test accuracy | 1.0000 | 1.0000 | nearest projected class mean vs. Gaussian Bayes rule |
| k-fold cross-validation | Breast cancer | mean CV accuracy | 0.9274 | 0.9274 | std 0.0377 vs 0.0377 |
