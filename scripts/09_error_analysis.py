"""Where do the remaining errors come from? (student-test trips)

Run:  python scripts/09_error_analysis.py      (after 08_eval_hmm.py)

Every wrong observation is put in one category:
  reverse   predicted the same street but the opposite direction (two-way street)
  adjacent  predicted an arc that touches the reference arc (boundary between arcs)
  other     anything else (parallel road, wrong branch, ...)

For wrong-direction errors we also check what the GPS itself says: we compare
the direction of travel (from GPS points i-3..i+3) with the predicted and the
reference arc, for points where the vehicle actually moved (> 15 m in that window).

Outputs
  results/hmm/error_breakdown_per_trip.csv
  results/hmm/error_breakdown.csv
  results/hmm/reverse_error_headings.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import ROOT, load_config, path  # noqa: E402
from pathsnap.features import kinematics  # noqa: E402
from pathsnap.geo import angle_diff  # noqa: E402
from pathsnap.inference import PathSnapPredictor  # noqa: E402
from pathsnap.io import list_trips, load_route, load_trip  # noqa: E402
from pathsnap.labels import reference_arcs  # noqa: E402


def category(net, a, b):
    if a < 0:
        return "none"
    if net.is_reverse(a, b):
        return "reverse"
    if len({int(net.arc_from[a]), int(net.arc_to[a])} & {int(net.arc_from[b]), int(net.arc_to[b])}):
        return "adjacent"
    return "other"


def main():
    cfg = load_config()
    out = path(cfg, "results") / "hmm"; out.mkdir(parents=True, exist_ok=True)
    sdir = path(cfg, "student_test")
    pred = PathSnapPredictor(ROOT / "artifacts")
    rows, heads = [], []
    for tid in list_trips(sdir):
        res = pred.predict_trip(load_trip(sdir, tid, with_route=False))
        ctx, net = res.ctx, res.ctx["net"]
        ref, _, _ = reference_arcs(ctx["px"], ctx["py"], load_route(sdir, tid), net, pred.meta["label_max_dist"])
        kin = kinematics(ctx["px"], ctx["py"], ctx["t"])
        for name, p in (("classifier", res.classifier_arcs), ("hmm", res.hmm_arcs)):
            wrong = np.where((ref >= 0) & (p != ref))[0]
            cats = [category(net, int(p[i]), int(ref[i])) for i in wrong]
            counts = pd.Series(cats).value_counts().to_dict() if cats else {}
            rows.append({"trip_id": tid, "decoder": name, "labelled": int((ref >= 0).sum()), "errors": len(wrong),
                         **{c: counts.get(c, 0) for c in ("reverse", "adjacent", "other", "none")}})
            if name == "hmm":
                rev = np.array([c == "reverse" for c in cats], dtype=bool)
                idx = wrong[rev] if len(wrong) else wrong
                moving = idx[kin["move_w"][idx] > 15]
                heads.append({
                    "trip_id": tid, "reverse_errors": len(idx), "while_stationary": int(len(idx) - len(moving)),
                    "moving": len(moving),
                    "median_heading_diff_to_predicted": float(np.median(angle_diff(kin["head_w"][moving], net.bearing[res.hmm_arcs[moving]]))) if len(moving) else np.nan,
                    "median_heading_diff_to_reference": float(np.median(angle_diff(kin["head_w"][moving], net.bearing[ref[moving]]))) if len(moving) else np.nan,
                })
    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(out / "error_breakdown_per_trip.csv", index=False)
    total = per_trip.groupby("decoder")[["labelled", "errors", "reverse", "adjacent", "other", "none"]].sum()
    total.to_csv(out / "error_breakdown.csv")
    h = pd.DataFrame(heads)
    h.to_csv(out / "reverse_error_headings.csv", index=False)
    print(total.to_string())
    print(h.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
