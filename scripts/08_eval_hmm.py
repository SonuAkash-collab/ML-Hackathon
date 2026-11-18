"""Classifier-only vs classifier + HMM on the 10 student-testing trips.

Run:  python scripts/08_eval_hmm.py      (after 06 and 07)

The trips are loaded WITHOUT their route for prediction (exactly as the API
does); the route is loaded separately afterwards for scoring.

Outputs
  results/hmm/student_test_eval.csv     one row per trip
  results/hmm/student_test_summary.json
  figures/hmm/*.png                     comparison plots
  figures/maps/student_test_<trip>.html folium maps (GPS, network, reference, classifier, HMM)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import ROOT, load_config, path  # noqa: E402
from pathsnap.inference import PathSnapPredictor, score_prediction  # noqa: E402
from pathsnap.io import list_trips, load_route, load_trip  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402


def static_map(ctx, route, res, tid, sc, out_png):
    """Static lon/lat picture of the same layers as the folium map (for the PDF report)."""
    net, tr = ctx["net"], ctx["track"]
    f, ax = plt.subplots(figsize=(7, 7))
    nb = np.unique(res.feats["arc"].to_numpy()) if len(res.feats) else []
    for a in nb:
        (la1, lo1), (la2, lo2) = net.arc_lonlat(int(a))
        ax.plot([lo1, lo2], [la1, la2], color="#bbbbbb", lw=0.8, zorder=1)
    for a in route:
        (la1, lo1), (la2, lo2) = net.arc_lonlat(int(a))
        ax.plot([lo1, lo2], [la1, la2], color="#1a9e3a", lw=5, alpha=0.5, zorder=2)
    for a in res.route_arcs:
        (la1, lo1), (la2, lo2) = net.arc_lonlat(int(a))
        ax.plot([lo1, lo2], [la1, la2], color="#d62728", lw=1.8, zorder=3)
    ax.scatter(tr[:, 0], tr[:, 1], s=2, color="#1f6fd1", zorder=4)
    ax.plot([], [], color="#bbbbbb", label="candidate roads")
    ax.plot([], [], color="#1a9e3a", lw=5, alpha=.5, label="reference route")
    ax.plot([], [], color="#d62728", label="predicted route (HMM)")
    ax.scatter([], [], s=8, color="#1f6fd1", label="GPS observations")
    ax.set_aspect(1 / np.cos(np.radians(float(np.median(tr[:, 1])))))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Student-test trip {tid}: HMM accuracy {sc['accuracy_hmm']:.3f}")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=.3)
    f.tight_layout(); f.savefig(out_png, dpi=120); plt.close(f)


def main():
    cfg = load_config()
    out = path(cfg, "results") / "hmm"; out.mkdir(parents=True, exist_ok=True)
    fig = path(cfg, "figures") / "hmm"; fig.mkdir(parents=True, exist_ok=True)
    maps = path(cfg, "figures") / "maps"; maps.mkdir(parents=True, exist_ok=True)
    pred = PathSnapPredictor(ROOT / "artifacts")
    sdir = path(cfg, "student_test")

    rows = []
    for tid in list_trips(sdir):
        trip = load_trip(sdir, tid, with_route=False)
        res = pred.predict_trip(trip)
        route = load_route(sdir, tid)
        sc = score_prediction(res, route, pred.meta["label_max_dist"])
        rows.append({"trip_id": tid, **sc, **res.info, **res.timings})
        print(f"{tid}: classifier {sc['accuracy_classifier']:.4f} -> HMM {sc['accuracy_hmm']:.4f}  "
              f"continuity {sc['continuity_classifier']:.3f} -> {sc['continuity_hmm']:.3f}  "
              f"route F1 {sc['route_f1_hmm']:.3f}  ({res.timings['total_s']:.1f}s)")
        ctx = res.ctx
        m = trip_map(ctx["track"], ctx["net"], ctx["px"], ctx["py"],
                     title=f"Student-test trip {tid}: HMM accuracy {sc['accuracy_hmm']:.3f}",
                     reference_arcs=route, classifier_arcs=res.classifier_arcs, hmm_arcs=res.route_arcs,
                     gps_step=max(1, len(ctx["px"]) // 1500))
        m.save(str(maps / f"student_test_{tid}.html"))
        static_map(ctx, route, res, tid, sc, fig / f"route_{tid}.png")

    df = pd.DataFrame(rows)
    df.to_csv(out / "student_test_eval.csv", index=False)
    w = df["n_labelled"]
    summary = {
        "pointwise_accuracy_classifier": float(np.average(df["accuracy_classifier"], weights=w)),
        "pointwise_accuracy_hmm": float(np.average(df["accuracy_hmm"], weights=w)),
        "route_continuity_classifier": float(df["continuity_classifier"].mean()),
        "route_continuity_hmm": float(df["continuity_hmm"].mean()),
        "route_f1_classifier": float(df["route_f1_classifier"].mean()),
        "route_f1_hmm": float(df["route_f1_hmm"].mean()),
        "route_precision_hmm": float(df["route_precision_hmm"].mean()),
        "route_recall_hmm": float(df["route_recall_hmm"].mean()),
        "candidate_miss_rate": float(np.average(df["candidate_miss_rate"], weights=w)),
        "trips_improved_by_hmm": int((df["accuracy_hmm"] > df["accuracy_classifier"]).sum()),
        "trips_worse_with_hmm": int((df["accuracy_hmm"] < df["accuracy_classifier"]).sum()),
        "mean_seconds_per_trip": float(df["total_s"].mean()),
        "hmm_params": pred.hmm_params.to_dict(),
    }
    json.dump(summary, open(out / "student_test_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # ------------------------------------------------------------ plots
    x = np.arange(len(df))
    f, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, df["accuracy_classifier"], 0.4, label="classifier only", color="#f28e2b")
    ax.bar(x + 0.2, df["accuracy_hmm"], 0.4, label="classifier + HMM", color="#d62728")
    ax.set_xticks(x, df["trip_id"].str[-3:], rotation=0)
    ax.set_ylim(max(0, df[["accuracy_classifier", "accuracy_hmm"]].min().min() - 0.05), 1.0)
    ax.set_xlabel("student-test trip"); ax.set_ylabel("pointwise accuracy")
    ax.set_title("Pointwise accuracy per trip"); ax.legend(); ax.grid(axis="y", alpha=.3)
    f.tight_layout(); f.savefig(fig / "accuracy_per_trip.png", dpi=120); plt.close(f)

    f, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, df["continuity_classifier"], 0.4, label="classifier only", color="#f28e2b")
    ax.bar(x + 0.2, df["continuity_hmm"], 0.4, label="classifier + HMM", color="#d62728")
    ax.set_xticks(x, df["trip_id"].str[-3:])
    ax.set_xlabel("student-test trip"); ax.set_ylabel("route continuity")
    ax.set_title("Share of consecutive predicted arcs that are connected (<= 3 arcs)")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    f.tight_layout(); f.savefig(fig / "continuity_per_trip.png", dpi=120); plt.close(f)

    f, ax = plt.subplots(figsize=(6, 4))
    labels = ["pointwise\naccuracy", "route\ncontinuity", "route F1"]
    a = [summary["pointwise_accuracy_classifier"], summary["route_continuity_classifier"], summary["route_f1_classifier"]]
    b = [summary["pointwise_accuracy_hmm"], summary["route_continuity_hmm"], summary["route_f1_hmm"]]
    xx = np.arange(3)
    ax.bar(xx - 0.2, a, 0.4, label="classifier only", color="#f28e2b")
    ax.bar(xx + 0.2, b, 0.4, label="classifier + HMM", color="#d62728")
    for i in range(3):
        ax.text(xx[i] - 0.2, a[i], f"{a[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xx[i] + 0.2, b[i], f"{b[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xx, labels); ax.set_ylim(0.5, 1.05); ax.legend(loc="lower right")
    ax.set_title("Student-test summary"); ax.grid(axis="y", alpha=.3)
    f.tight_layout(); f.savefig(fig / "classifier_vs_hmm_summary.png", dpi=120); plt.close(f)


if __name__ == "__main__":
    main()
