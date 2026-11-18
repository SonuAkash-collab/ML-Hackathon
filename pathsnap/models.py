"""Model zoo for the candidate classifier.

Every model is a binary classifier: P(candidate arc is the reference arc).
We picked one or two models from each algorithm family so the comparison
says something about *which kind* of decision rule suits the problem.
Models that are sensitive to feature scale are wrapped with a StandardScaler.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# lightgbm >= 4.6 prefers eval_X/eval_y; eval_set still works and is what older 4.x versions accept
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# name -> (family, short description of the key configuration)
MODEL_INFO = {
    "logreg": ("Linear", "LogisticRegression, L2, C=1.0, standardised features"),
    "gnb": ("Probabilistic", "GaussianNB, var_smoothing=1e-9"),
    "knn": ("Instance-based", "KNN k=25, distance-weighted, standardised, <=150k training rows"),
    "tree": ("Tree-based", "DecisionTree max_depth=12, min_samples_leaf=50"),
    "rf": ("Ensemble (bagging)", "RandomForest 150 trees, max_depth=18, min_samples_leaf=5"),
    "hgb": ("Boosting", "HistGradientBoosting lr=0.1, max_iter=300, early stopping"),
    "lgbm": ("Boosting", "LightGBM 600 rounds, lr=0.05, 63 leaves, early stopping"),
    "mlp": ("Neural network", "MLP (64, 32) ReLU, Adam, early stopping, standardised"),
}

# rows used for fitting (the rest of the pipeline is identical); KNN has to
# store and search its training set so it gets a capped subsample
MAX_TRAIN_ROWS = {"knn": 150_000}


def make_model(name: str, params: dict | None = None):
    params = params or {}
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 1.0), max_iter=500,
                                                                  random_state=SEED))
    if name == "gnb":
        return GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 25),
                                                                    weights="distance", n_jobs=-1))
    if name == "tree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth", 12),
                                      min_samples_leaf=params.get("min_samples_leaf", 50), random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 150),
                                      max_depth=params.get("max_depth", 18),
                                      min_samples_leaf=params.get("min_samples_leaf", 5),
                                      max_features="sqrt", n_jobs=-1, random_state=SEED)
    if name == "hgb":
        return HistGradientBoostingClassifier(learning_rate=params.get("learning_rate", 0.1),
                                              max_iter=params.get("max_iter", 300),
                                              max_leaf_nodes=params.get("max_leaf_nodes", 31),
                                              early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=20, random_state=SEED)
    if name == "lgbm":
        defaults = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
        defaults.update(params)
        return lgb.LGBMClassifier(objective="binary", random_state=SEED, n_jobs=-1, verbose=-1, **defaults)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=params.get("hidden", (64, 32)),
                                                             alpha=params.get("alpha", 1e-4),
                                                             batch_size=512, learning_rate_init=1e-3,
                                                             max_iter=60, early_stopping=True,
                                                             validation_fraction=0.1, n_iter_no_change=6,
                                                             random_state=SEED))
    raise ValueError(f"unknown model {name}")


def fit_model(name: str, model, X, y, X_val=None, y_val=None):
    """Fit, using a validation set for LightGBM early stopping when given."""
    if name == "lgbm" and X_val is not None:
        model.fit(X, y, eval_set=[(X, y), (X_val, y_val)], eval_names=["train", "valid"],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    else:
        model.fit(X, y)
    return model


def training_history(name: str, model) -> dict | None:
    """Epoch/iteration-wise curves, only for models that actually produce them."""
    if name == "lgbm" and getattr(model, "evals_result_", None):
        r = model.evals_result_
        return {"x_label": "boosting round", "metric": "binary log loss",
                "train": list(map(float, r["train"]["binary_logloss"])),
                "valid": list(map(float, r["valid"]["binary_logloss"]))}
    if name == "hgb":
        return {"x_label": "boosting iteration", "metric": "log loss (internal 10% split)",
                "train": list(map(float, -model.train_score_)),
                "valid": list(map(float, -model.validation_score_))}
    if name == "mlp":
        mlp = model[-1]
        return {"x_label": "epoch", "metric": "train: log loss / valid: accuracy (internal 10% split)",
                "train": list(map(float, mlp.loss_curve_)),
                "valid_accuracy": list(map(float, mlp.validation_scores_))}
    return None
