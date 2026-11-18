"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.

Run:  python scripts/06_bayes_opt.py

Protocol
  * objective: mean observation-level top-1 accuracy over a GroupKFold of the
    80 training trips (no student-test or evaluator data is used in the search)
  * LightGBM early stopping uses an inner 10 % split of the *training-fold*
    trips, never the fold being scored
  * every trial is stored in results/optuna/study.db (SQLite) and trials.csv
  * the best configuration is refit on all training trips, compared with the
    untuned configuration under the same protocol, scored once on the student
    test trips and exported to artifacts/

Outputs
  results/optuna/trials.csv, study.db, best_params.json, tuned_vs_untuned.csv
  figures/optuna/*.png
  artifacts/model.joblib, artifacts/metadata.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import ROOT, load_config, path  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import all_metrics  # noqa: E402
from pathsnap.models import fit_model, make_model  # noqa: E402


def search_space(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "lgbm":
        return {
            # small learning rates need many rounds; early stopping picks the count
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            # tree complexity: 15 leaves ~ depth 4, 255 leaves ~ depth 8
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            # minimum rows in a leaf: guards against fitting single trips
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "n_estimators": 1500,
        }
    if model_name == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "max_iter": 800,
        }
    raise ValueError(f"no search space defined for {model_name}")


def fit_with_inner_es(name, params, X, y, groups, seed=42):
    """Fit on (X, y); LightGBM early-stops on 10 % of the given trips."""
    model = make_model(name, params)
    if name == "lgbm":
        ia, ib = next(GroupShuffleSplit(1, test_size=0.1, random_state=seed).split(X, y, groups))
        fit_model(name, model, X[ia], y[ia], X[ib], y[ib])
    else:
        fit_model(name, model, X, y)
    return model


def cross_validate(name, params, tr, X, y, groups, folds, trial=None):
    rows = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        t0 = time.time()
        model = fit_with_inner_es(name, params, X[a], y[a], groups[a])
        fit_s = time.time() - t0
        p = model.predict_proba(X[b])[:, 1]
        m = all_metrics(tr.iloc[b], p)
        rows.append({"fold": k, "top1_accuracy": m["top1_accuracy"], "f1_macro": m["f1_macro"],
                     "log_loss": m["log_loss"], "fit_s": fit_s,
                     "best_iteration": getattr(model, "best_iteration_", None)})
        if trial is not None:
            trial.report(float(np.mean([r["top1_accuracy"] for r in rows])), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return pd.DataFrame(rows)


def evaluate_on_student_test(model, test, Xt):
    t0 = time.time()
    p = model.predict_proba(Xt)[:, 1]
    inf = time.time() - t0
    m = all_metrics(test, p)
    n_obs = test.groupby(["trip_id", "obs"]).ngroups
    return m, 1e6 * inf / n_obs   # ms per 1000 observations


def main():
    cfg = load_config()
    ocfg = cfg["optuna"]
    name = ocfg["model"]
    cache = path(cfg, "cache")
    out = path(cfg, "results") / "optuna"; out.mkdir(parents=True, exist_ok=True)
    fig = path(cfg, "figures") / "optuna"; fig.mkdir(parents=True, exist_ok=True)
    art = ROOT / "artifacts"; art.mkdir(exist_ok=True)

    train = pd.read_parquet(cache / "features_train.parquet")
    test = pd.read_parquet(cache / "features_student_test.parquet")
    tr = train[(train["ref_arc"] >= 0) & (train["obs"] % ocfg["train_stride"] == 0)].reset_index(drop=True)
    X, y, groups = tr[FEATURES].to_numpy(float), tr["label"].to_numpy(), tr["trip_id"].to_numpy()
    Xt = test[FEATURES].to_numpy(float)
    print(f"{name}: {len(tr):,} training rows from {tr['trip_id'].nunique()} trips")

    # ------------------------------------------------------------- search
    storage = f"sqlite:///{(out / 'study.db').as_posix()}"
    study = optuna.create_study(study_name=f"pathsnap_{name}", direction="maximize", storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=ocfg["seed"], n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))

    def objective(trial):
        params = search_space(trial, name)
        t0 = time.time()
        cv = cross_validate(name, params, tr, X, y, groups, ocfg["cv_folds"], trial)
        trial.set_user_attr("cv_f1_macro", float(cv["f1_macro"].mean()))
        trial.set_user_attr("cv_log_loss", float(cv["log_loss"].mean()))
        trial.set_user_attr("cv_top1_std", float(cv["top1_accuracy"].std()))
        trial.set_user_attr("seconds", time.time() - t0)
        trial.set_user_attr("best_iteration_mean", float(pd.to_numeric(cv["best_iteration"]).mean()))
        return float(cv["top1_accuracy"].mean())

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, ocfg["n_trials"] - done)
    t_search = time.time()
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=ocfg["timeout_s"], show_progress_bar=False)
    search_s = time.time() - t_search
    trials = study.trials_dataframe()
    trials.to_csv(out / "trials.csv", index=False)
    best = study.best_trial
    best_params = {**search_space(optuna.trial.FixedTrial(best.params), name)}
    print("best trial", best.number, "top1", round(best.value, 4), best_params)

    # ------------------------------------------------- tuned vs untuned
    rows = []
    models = {}
    for label, params in (("untuned", {}), ("tuned", best_params)):
        cv = cross_validate(name, params, tr, X, y, groups, ocfg["cv_folds"])
        t0 = time.time()
        model = fit_with_inner_es(name, params, X, y, groups)
        fit_s = time.time() - t0
        m, inf_ms = evaluate_on_student_test(model, test, Xt)
        models[label] = model
        rows.append({"model": f"{name} ({label})", "cv_f1_macro": cv["f1_macro"].mean(),
                     "cv_top1": cv["top1_accuracy"].mean(), "cv_log_loss": cv["log_loss"].mean(),
                     "test_top1": m["top1_accuracy"], "test_f1_macro": m["f1_macro"],
                     "test_log_loss": m["log_loss"], "inference_ms_per_1k_obs": inf_ms,
                     "fit_time_s": fit_s, "n_rounds": getattr(model, "best_iteration_", None)})
        print(label, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "tuned_vs_untuned.csv", index=False)

    n_complete = int((trials["state"] == "COMPLETE").sum())
    n_pruned = int((trials["state"] == "PRUNED").sum())
    json.dump({"model": name, "best_trial": best.number, "best_cv_top1": best.value,
               "best_params": best_params, "n_trials_complete": n_complete, "n_trials_pruned": n_pruned,
               "search_seconds_this_run": search_s,
               "total_trial_seconds": float(trials.get("user_attrs_seconds", pd.Series(dtype=float)).sum())},
              open(out / "best_params.json", "w"), indent=2)

    # ------------------------------------------------------------- export
    final = models["tuned"] if comp.iloc[1]["cv_top1"] >= comp.iloc[0]["cv_top1"] else models["untuned"]
    chosen = "tuned" if final is models["tuned"] else "untuned"
    joblib.dump(final, art / "model.joblib", compress=3)
    meta_path = art / "metadata.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta.update({
        "model_name": name, "model_variant": chosen,
        "params": best_params if chosen == "tuned" else {},
        "features": FEATURES, "candidates": cfg["candidates"],
        "sample_spacing_m": cfg["network"]["sample_spacing_m"],
        "label_max_dist": cfg["labels"]["max_dist"],
        "hmm": meta.get("hmm", cfg["hmm"]), "lattice": cfg["lattice"],
        "trained_on": "80 training trips (labelled observations, every %d-th)" % ocfg["train_stride"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"exported {chosen} model to {art}")

    # -------------------------------------------------------------- plots
    t = trials[trials["state"] == "COMPLETE"].sort_values("number")
    f, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(t["number"], t["value"], s=18, label="trial (CV top-1)")
    ax.plot(t["number"], t["value"].cummax(), color="#d62728", label="best so far")
    ax.axhline(comp.iloc[0]["cv_top1"], color="grey", ls="--", label="untuned")
    ax.set_xlabel("trial"); ax.set_ylabel("CV top-1 accuracy"); ax.set_title(f"Optuna TPE search ({name})")
    ax.legend(); ax.grid(alpha=.3); f.tight_layout(); f.savefig(fig / "optimisation_history.png", dpi=120); plt.close(f)

    pcols = [c for c in t.columns if c.startswith("params_")]
    nc = 3
    nr = int(np.ceil(len(pcols) / nc))
    f, axes = plt.subplots(nr, nc, figsize=(11, 3.2 * nr))
    for ax, c in zip(np.ravel(axes), pcols):
        ax.scatter(t[c], t["value"], s=14, c=t["number"], cmap="viridis")
        ax.set_xlabel(c.replace("params_", "")); ax.set_ylabel("CV top-1")
        if c.replace("params_", "") in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda", "reg_alpha", "max_leaf_nodes"):
            ax.set_xscale("log")
        ax.grid(alpha=.3)
    for ax in np.ravel(axes)[len(pcols):]:
        ax.axis("off")
    f.suptitle("Objective vs each hyperparameter (colour = trial number)")
    f.tight_layout(); f.savefig(fig / "param_slices.png", dpi=120); plt.close(f)

    try:
        imp = optuna.importance.get_param_importances(study)
        f, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#1f6fd1")
        ax.set_xlabel("importance (fANOVA)"); ax.set_title("Hyperparameter importance")
        f.tight_layout(); f.savefig(fig / "param_importance.png", dpi=120); plt.close(f)
        json.dump(imp, open(out / "param_importance.json", "w"), indent=2)
    except Exception as exc:  # importance needs a few completed trials
        print("skipping importance plot:", exc)

    f, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, col, lab in zip(axes, ["cv_f1_macro", "test_top1", "test_log_loss"],
                            ["CV macro F1", "student-test top-1", "student-test log loss"]):
        ax.bar(["untuned", "tuned"], comp[col], color=["#9a9a9a", "#d62728"])
        lo, hi = comp[col].min(), comp[col].max()
        pad = max((hi - lo) * 2, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        for i, v in enumerate(comp[col]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
        ax.set_title(lab)
    f.tight_layout(); f.savefig(fig / "tuned_vs_untuned.png", dpi=120); plt.close(f)


if __name__ == "__main__":
    main()
