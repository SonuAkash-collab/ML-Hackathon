"""Build the candidate/feature tables for the train and student_test partitions.

Run:  python scripts/02_build_features.py

Outputs
  cache/features_train.parquet, cache/features_student_test.parquet
  results/candidates/candidate_stats.csv   per-trip candidate statistics
  results/candidates/candidate_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import load_config, path  # noqa: E402
from pathsnap.features import attach_labels, prepare_trip  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402


def build_one(split_dir, tid, cfg):
    t0 = time.time()
    trip = load_trip(split_dir, tid)
    feats, ctx = prepare_trip(trip, cfg["candidates"], cfg["network"]["sample_spacing_m"])
    feats = attach_labels(feats, ctx, trip.route, cfg["labels"]["max_dist"])
    ref = ctx["ref_arc"]
    labelled = np.where(ref >= 0)[0]
    hit = feats.groupby("obs")["label"].max().reindex(labelled).fillna(0)
    nearest = feats[feats["rank"] == 0].set_index("obs")["label"].reindex(labelled).fillna(0)
    stats = dict(
        trip_id=tid, n_obs=len(ctx["px"]), n_labelled=len(labelled),
        n_rows=len(feats), cands_per_obs=len(feats) / max(feats["obs"].nunique(), 1),
        obs_without_candidates=int(len(ctx["px"]) - feats["obs"].nunique()),
        obs_expanded=int(feats.loc[feats["expanded"] == 1, "obs"].nunique()),
        candidate_miss_rate=float(1 - hit.mean()) if len(hit) else np.nan,
        nearest_arc_top1=float(nearest.mean()) if len(nearest) else np.nan,
        seconds=round(time.time() - t0, 2),
    )
    return feats, stats


def main():
    cfg = load_config()
    cache = path(cfg, "cache"); cache.mkdir(parents=True, exist_ok=True)
    out = path(cfg, "results") / "candidates"; out.mkdir(parents=True, exist_ok=True)
    all_stats, summary = [], {}
    for split in ("train", "student_test"):
        sdir = path(cfg, split)
        trips = list_trips(sdir)
        res = Parallel(n_jobs=-1, verbose=0)(delayed(build_one)(sdir, t, cfg) for t in trips)
        feats = pd.concat([r[0] for r in res], ignore_index=True)
        stats = pd.DataFrame([r[1] for r in res]); stats.insert(0, "split", split)
        feats.to_parquet(cache / f"features_{split}.parquet", index=False)
        all_stats.append(stats)
        lab = stats["n_labelled"]
        summary[split] = dict(
            trips=len(trips), rows=int(len(feats)), observations=int(stats["n_obs"].sum()),
            labelled_observations=int(lab.sum()), positive_rate=float(feats["label"].mean()),
            cands_per_obs=float(len(feats) / feats.groupby(["trip_id", "obs"]).ngroups),
            candidate_miss_rate=float(np.average(stats["candidate_miss_rate"].fillna(0), weights=lab)),
            nearest_arc_top1=float(np.average(stats["nearest_arc_top1"].fillna(0), weights=lab)),
            obs_without_candidates=int(stats["obs_without_candidates"].sum()),
        )
        print(split, json.dumps(summary[split], indent=1))
    pd.concat(all_stats).to_csv(out / "candidate_stats.csv", index=False)
    with open(out / "candidate_summary.json", "w") as fh:
        json.dump({"config": cfg["candidates"], **summary}, fh, indent=2)


if __name__ == "__main__":
    main()
