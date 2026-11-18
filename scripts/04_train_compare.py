"""Train and compare the candidate classifiers.

Run:  python scripts/04_train_compare.py            (all models in configs/models.yaml)
      python scripts/04_train_compare.py lgbm rf    (just some of them)

Protocol
  * fit / cross-validate on the 80 training trips only (GroupKFold by trip)
  * only labelled observations are used for training (unlabelled observations
    have no trustworthy negatives)
  * the final model of each family is refit on all training trips and scored
    once on the 10 student-testing trips
Outputs
  results/models/cv_folds.csv          per-fold CV metrics
  results/models/comparison.csv        one row per model (the report table)
  results/models/details.json          configs, confusion matrices, histories
  cache/probs_student_test.parquet     student-test probabilities per model
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import load_config, path  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.geo import Projector  # noqa: E402
from pathsnap.metrics import all_metrics, route_continuity, route_set_scores, top1_table  # noqa: E402
from pathsnap.models import MAX_TRAIN_ROWS, MODEL_INFO, fit_model, make_model, training_history  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402


def training_rows(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    return df[(df["ref_arc"] >= 0) & (df["obs"] % stride == 0)].reset_index(drop=True)


def cap_rows(name, X, y, seed=42):
    cap = MAX_TRAIN_ROWS.get(name)
    if cap and len(X) > cap:
        idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
        return X[idx], y[idx]
    return X, y


def student_networks(cfg):
    nets = {}
    sdir = path(cfg, "student_test")
    for tid in list_trips(sdir):
        trip = load_trip(sdir, tid)
        proj = Projector(float(np.median(trip.track[:, 0])), float(np.median(trip.track[:, 1])))
        nets[tid] = (RoadNetwork(trip.nodes, trip.arcs, proj), trip.route)
    return nets


def sequence_metrics(top: pd.DataFrame, nets) -> dict:
    cont, f1 = [], []
    for tid, g in top.groupby("trip_id"):
        net, route = nets[tid]
        seq = g.sort_values("obs")["pred_arc"].to_numpy()
        cont.append(route_continuity(seq, net))
        f1.append(route_set_scores(seq, route)["route_f1"])
    return {"route_continuity": float(np.mean(cont)), "route_f1": float(np.mean(f1))}


def main():
    cfg = load_config()
    mcfg = cfg["models"]
    skip_cv = "--no-cv" in sys.argv[1:]  # local review-run switch: refit + student-test scoring only
    names = [a for a in sys.argv[1:] if a != "--no-cv"] or mcfg["candidates"]
    if skip_cv:
        print("cross-validation SKIPPED (--no-cv); cv_* columns will be NaN")
    cache, out = path(cfg, "cache"), path(cfg, "results") / "models"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(cache / "features_train.parquet")
    test = pd.read_parquet(cache / "features_student_test.parquet")
    tr = training_rows(train, mcfg["train_stride"])
    print(f"training rows {len(tr):,} (of {len(train):,}); student-test rows {len(test):,}")
    X, y, groups = tr[FEATURES].to_numpy(float), tr["label"].to_numpy(), tr["trip_id"].to_numpy()
    Xt = test[FEATURES].to_numpy(float)
    nets = student_networks(cfg)
    n_test_obs = test.groupby(["trip_id", "obs"]).ngroups

    folds_path, comp_path, det_path = out / "cv_folds.csv", out / "comparison.csv", out / "details.json"
    fold_rows = pd.read_csv(folds_path).to_dict("records") if folds_path.exists() else []
    comp_rows = pd.read_csv(comp_path).to_dict("records") if comp_path.exists() else []
    details = json.load(open(det_path)) if det_path.exists() else {}
    probs_path = cache / "probs_student_test.parquet"
    probs = pd.read_parquet(probs_path) if probs_path.exists() else test[["trip_id", "obs", "arc"]].copy()

    gkf = GroupKFold(n_splits=mcfg["cv_folds"])
    for name in names:
        family, desc = MODEL_INFO[name]
        print(f"\n=== {name} ({family})")
        fold_rows = [r for r in fold_rows if r["model"] != name]
        cv = []
        for k, (a, b) in enumerate([] if skip_cv else gkf.split(X, y, groups)):
            Xa, ya = cap_rows(name, X[a], y[a])
            model = make_model(name)
            t0 = time.time()
            fit_model(name, model, Xa, ya, X[b], y[b])
            fit_s = time.time() - t0
            t0 = time.time()
            p = model.predict_proba(X[b])[:, 1]
            pred_s = time.time() - t0
            m = all_metrics(tr.iloc[b], p)
            row = {"model": name, "fold": k, "fit_s": fit_s, "predict_s": pred_s,
                   **{kk: v for kk, v in m.items() if kk != "confusion_matrix"}}
            fold_rows.append(row); cv.append(row)
            print(f"  fold {k}: top1 {m['top1_accuracy']:.4f}  macroF1 {m['f1_macro']:.4f}  "
                  f"logloss {m['log_loss']:.4f}  fit {fit_s:.1f}s")
        cvdf = pd.DataFrame(cv, columns=["f1_macro", "top1_accuracy", "log_loss", "fit_s"]) if not cv else pd.DataFrame(cv)

        # refit on all training trips; LightGBM needs a held-out set of *training* trips
        model = make_model(name)
        if name == "lgbm":
            ia, ib = next(GroupShuffleSplit(1, test_size=0.1, random_state=42).split(X, y, groups))
            Xa, ya = X[ia], y[ia]
            t0 = time.time(); fit_model(name, model, Xa, ya, X[ib], y[ib]); fit_s = time.time() - t0
        else:
            Xa, ya = cap_rows(name, X, y)
            t0 = time.time(); fit_model(name, model, Xa, ya); fit_s = time.time() - t0
        t0 = time.time()
        pt = model.predict_proba(Xt)[:, 1]
        inf_s = time.time() - t0
        tm = all_metrics(test, pt)
        seq = sequence_metrics(top1_table(test, pt), nets)
        # training-set (in-sample) top-1 on the rows it was fitted on, to look for overfitting
        p_in = model.predict_proba(Xa)[:, 1]
        in_rows = tr.iloc[ia] if name == "lgbm" else (tr if len(Xa) == len(tr) else None)
        train_top1 = all_metrics(in_rows, p_in)["top1_accuracy"] if in_rows is not None else np.nan
        probs[name] = pt

        comp_rows = [r for r in comp_rows if r["model"] != name]
        comp_rows.append({
            "model": name, "family": family, "config": desc,
            "cv_f1_macro": cvdf["f1_macro"].mean(), "cv_f1_macro_std": cvdf["f1_macro"].std(),
            "cv_top1": cvdf["top1_accuracy"].mean(), "cv_top1_std": cvdf["top1_accuracy"].std(),
            "cv_log_loss": cvdf["log_loss"].mean(),
            "train_top1_in_sample": train_top1,
            "test_top1": tm["top1_accuracy"], "test_f1_macro": tm["f1_macro"],
            "test_precision_macro": tm["precision_macro"], "test_recall_macro": tm["recall_macro"],
            "test_f1_pos": tm["f1_pos"], "test_accuracy": tm["accuracy"], "test_log_loss": tm["log_loss"],
            "test_candidate_miss_rate": tm["candidate_miss_rate"],
            "test_route_continuity": seq["route_continuity"], "test_route_f1": seq["route_f1"],
            "fit_time_s": fit_s, "cv_fit_time_s_mean": cvdf["fit_s"].mean(),
            "inference_ms_per_1k_obs": 1000 * inf_s / n_test_obs * 1000,
            "train_rows": len(Xa),
        })
        details[name] = {"family": family, "config": desc, "params": {k: str(v) for k, v in model.get_params().items()},
                         "test_confusion_matrix": tm["confusion_matrix"],
                         "history": training_history(name, model)}
        print(f"  student test: top1 {tm['top1_accuracy']:.4f}  macroF1 {tm['f1_macro']:.4f}  "
              f"logloss {tm['log_loss']:.4f}  continuity {seq['route_continuity']:.3f}  "
              f"(in-sample top1 {train_top1:.4f})")
        # save after every model so a crash does not lose earlier results
        if fold_rows:  # nothing to write when CV is skipped
            pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
        pd.DataFrame(comp_rows).sort_values("cv_top1", ascending=False).to_csv(comp_path, index=False)
        json.dump(details, open(det_path, "w"), indent=1)
        probs.to_parquet(probs_path, index=False)

    comp = pd.DataFrame(comp_rows).sort_values("cv_top1", ascending=False)
    print("\n", comp[["model", "cv_f1_macro", "cv_top1", "cv_log_loss", "test_top1", "test_f1_macro",
                      "test_log_loss", "test_route_continuity", "inference_ms_per_1k_obs"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
