# 8-Model Comparison Table

Combined from team member results (5-fold GroupKFold by trip on 80 training trips, refit and evaluated on 10 student-test trips):
- **Sonu**: `tree`, `logreg`
- **Purushotham**: `lgbm`, `hgb`
- **Ayush**: `rf`, `knn`
- **Sahana**: `mlp`, `gnb`

| Model | Algorithm family | Key configuration | CV macro F1 | Top-1 accuracy (student test) | Route continuity | Student-test macro F1 | Log loss (student test) | Inference (ms / 1k obs) |
|---|---|---|---|---|---|---|---|---|
| LightGBM | Boosting | LightGBM 600 rounds, lr=0.05, 63 leaves, early stopping | 0.9349 ± 0.0097 | 0.9042 | 0.9367 | 0.9400 | 0.0849 | 112.3 |
| Random Forest | Ensemble (bagging) | RandomForest 150 trees, max_depth=18, min_samples_leaf=5 | 0.9352 ± 0.0093 | 0.8942 | 0.9289 | 0.9348 | 0.0879 | 15.0 |
| HistGradBoost | Boosting | HistGradientBoosting lr=0.1, max_iter=300, early stopping | 0.9331 ± 0.0102 | 0.8957 | 0.9241 | 0.9341 | 0.0914 | 8.4 |
| MLP | Neural network | MLP (64, 32) ReLU, Adam, early stopping, standardised | 0.9308 ± 0.0099 | 0.8839 | 0.9330 | 0.9280 | 0.0962 | 6.5 |
| Decision Tree | Tree-based | DecisionTree max_depth=12, min_samples_leaf=50 | 0.9300 ± 0.0111 | 0.8932 | 0.9255 | 0.9317 | 0.1085 | 0.8 |
| Logistic Reg. | Linear | LogisticRegression, L2, C=1.0, standardised features | 0.9282 ± 0.0102 | 0.8833 | 0.9357 | 0.9254 | 0.1151 | 1.0 |
| KNN | Instance-based | KNN k=25, distance-weighted, standardised, <=150k training rows | 0.9281 ± 0.0090 | 0.8808 | 0.9181 | 0.9251 | 0.1502 | 626.9 |
| Gaussian NB | Probabilistic | GaussianNB, var_smoothing=1e-9 | 0.9056 ± 0.0119 | 0.8752 | 0.9053 | 0.8963 | 0.5530 | 6.1 |
