"""Tune the HMM decoder parameters on out-of-fold training predictions.

Run:  python scripts/07_tune_decoder.py      (after 06_bayes_opt.py)

Why out-of-fold: if the decoder were tuned on probabilities the classifier
produced for trips it was trained on, those probabilities would be over-
confident and the decoder would learn to trust them too much. So each tuning
trip is scored by a model that never saw it (GroupKFold over the 80 training
trips). Only training data is used for the search; the student-test trips are
decoded afterwards with the default and the tuned parameters for reporting.

Outputs
  results/hmm/decoder_grid.csv       every parameter combination tried
  results/hmm/decoder_tuning.json    best parameters + default vs tuned
  artifacts/metadata.json            "hmm" section updated with the tuned values
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import ROOT, load_config, path  # noqa: E402
from pathsnap.features import FEATURES, attach_labels, prepare_trip  # noqa: E402
from pathsnap.hmm import HMMParams, build_lattice, viterbi  # noqa: E402
from pathsnap.inference import PathSnapPredictor  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.metrics import route_continuity  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

bo = import_module("06_bayes_opt")

N_TUNE_TRIPS = 30
MAX_OBS = 6000


def decode_score(lattices, params: HMMParams):
    correct = labelled = 0
    cont = []
    for L, ref, net in lattices:
        path, _ = viterbi(L, params)
        obs = np.fromiter(path.keys(), dtype=int)
        arcs = np.fromiter(path.values(), dtype=int)
        lab = ref >= 0
        pred = np.full(len(ref), -1)
        pred[obs] = arcs
        correct += int((pred[lab] == ref[lab]).sum())
        labelled += int(lab.sum())
        cont.append(route_continuity(pred, net))
    return correct / labelled, float(np.mean(cont))


def main():
    cfg = load_config()
    art = ROOT / "artifacts"
    meta = json.load(open(art / "metadata.json"))
    name, params = meta["model_name"], meta["params"]
    out = path(cfg, "results") / "hmm"; out.mkdir(parents=True, exist_ok=True)
    cache = path(cfg, "cache")

    train = pd.read_parquet(cache / "features_train.parquet")
    stride = cfg["optuna"]["train_stride"]
    tr = train[(train["ref_arc"] >= 0) & (train["obs"] % stride == 0)].reset_index(drop=True)
    X, y, groups = tr[FEATURES].to_numpy(float), tr["label"].to_numpy(), tr["trip_id"].to_numpy()

    inv = pd.read_csv(path(cfg, "results") / "candidates" / "candidate_stats.csv")
    inv = inv[(inv["split"] == "train") & (inv["n_obs"] <= MAX_OBS) & (inv["n_labelled"] > 0.9 * inv["n_obs"])]
    rng = np.random.default_rng(cfg["seed"])
    tune_trips = sorted(rng.choice(inv["trip_id"].astype(str).str.zfill(8).to_numpy(),
                                   size=min(N_TUNE_TRIPS, len(inv)), replace=False))
    print(f"tuning on {len(tune_trips)} training trips (out-of-fold probabilities)")

    trips_all = np.array(sorted(tr["trip_id"].unique()))
    sdir = path(cfg, "train")
    lattices = []
    t0 = time.time()
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(trips_all, groups=trips_all)):
        held = set(trips_all[b]) & set(tune_trips)
        if not held:
            continue
        m = ~np.isin(groups, list(trips_all[b]))
        model = bo.fit_with_inner_es(name, params, X[m], y[m], groups[m])
        for tid in sorted(held):
            trip = load_trip(sdir, tid)
            feats, ctx = prepare_trip(trip, cfg["candidates"], cfg["network"]["sample_spacing_m"])
            attach_labels(feats, ctx, trip.route, cfg["labels"]["max_dist"])
            prob = model.predict_proba(feats[FEATURES].to_numpy(float))[:, 1]
            L = build_lattice(feats, prob, ctx["px"], ctx["py"], ctx["t"], ctx["net"], **cfg["lattice"])
            lattices.append((L, ctx["ref_arc"], ctx["net"]))
        print(f"  fold {k}: {len(held)} trips decoded-ready ({time.time() - t0:.0f}s)")

    base = HMMParams(**cfg["hmm"])
    rows = []
    # stage 1: transition scale x emission weight
    for beta, w in itertools.product([2, 4, 7, 10, 15, 25, 40], [0.5, 1.0, 2.0]):
        p = HMMParams(**{**base.to_dict(), "beta": beta, "emission_weight": w})
        acc, cont = decode_score(lattices, p)
        rows.append({**p.to_dict(), "stage": 1, "accuracy": acc, "continuity": cont})
    best1 = max(rows, key=lambda r: r["accuracy"])
    # stage 2: u-turn penalty x gap threshold around the stage-1 optimum
    for u, gap in itertools.product([0.0, 1.0, 2.0, 4.0, 8.0], [30.0, 60.0, 120.0]):
        p = HMMParams(**{**base.to_dict(), "beta": best1["beta"], "emission_weight": best1["emission_weight"],
                         "uturn_penalty": u, "gap_seconds": gap})
        acc, cont = decode_score(lattices, p)
        rows.append({**p.to_dict(), "stage": 2, "accuracy": acc, "continuity": cont})
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "decoder_grid.csv", index=False)
    best = grid.sort_values(["accuracy", "continuity"], ascending=False).iloc[0]
    tuned = HMMParams(**{k: float(best[k]) for k in base.to_dict()})
    default_acc, default_cont = decode_score(lattices, base)

    # classifier-only reference on the same trips
    clf_correct = clf_lab = 0
    for L, ref, _ in lattices:
        for s in L:
            if ref[s.obs] >= 0:
                clf_lab += 1
                clf_correct += int(s.arcs[int(np.argmax(s.prob))] == ref[s.obs])
    summary = {"tuning_trips": list(map(str, tune_trips)),
               "classifier_only_accuracy": clf_correct / clf_lab,
               "default_params": base.to_dict(), "default_accuracy": default_acc, "default_continuity": default_cont,
               "tuned_params": tuned.to_dict(), "tuned_accuracy": float(best["accuracy"]),
               "tuned_continuity": float(best["continuity"])}

    # report (not tune) on student-test
    meta["hmm"] = tuned.to_dict()
    meta["lattice"] = cfg["lattice"]
    json.dump(meta, open(art / "metadata.json", "w"), indent=2)
    pred = PathSnapPredictor(art)
    st = path(cfg, "student_test")
    st_res = {}
    for label, prm in (("default", base), ("tuned", tuned)):
        pred.hmm_params = prm
        c = n = 0
        for tid in list_trips(st):
            trip = load_trip(st, tid, with_route=False)
            res = pred.predict_trip(trip)
            route = load_trip(st, tid).route
            from pathsnap.labels import reference_arcs
            ref, _, _ = reference_arcs(res.ctx["px"], res.ctx["py"], route, res.ctx["net"], cfg["labels"]["max_dist"])
            lab = ref >= 0
            c += int((res.hmm_arcs[lab] == ref[lab]).sum()); n += int(lab.sum())
        st_res[label] = c / n
    summary["student_test_accuracy"] = st_res
    json.dump(summary, open(out / "decoder_tuning.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
