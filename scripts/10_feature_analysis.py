"""Which features matter? (feature selection evidence)

Run:  python scripts/10_feature_analysis.py      (after 02_build_features.py)

Uses the 80 training trips only.
  1. LightGBM gain importance of each of the 23 features
  2. Group ablation: drop one feature group at a time and measure the change in
     3-fold GroupKFold top-1 accuracy (a drop = the group is useful)
  3. Correlation matrix of the features (redundancy check)
  4. Distributions of the strongest features for correct vs wrong candidates

Outputs
  results/features/importance.csv, ablation.csv
  figures/features/importance.png, ablation.png, correlation.png, distributions.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import load_config, path  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import observation_metrics  # noqa: E402
from pathsnap.models import make_model  # noqa: E402

GROUPS = {
    "geometry": ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands", "t", "outside"],
    "heading": ["head_diff", "head_diff_wide", "cos_head_wide", "head_rank"],
    "context (neighbour points)": ["win_dist_mean", "win_dist_max", "win_rank"],
    "kinematics": ["speed", "dt_prev", "dt_next", "move_wide"],
    "topology": ["arc_length", "out_deg_end", "in_deg_start", "has_reverse", "expanded"],
}
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}
GROUP_COLOR = {"geometry": "#1f6fd1", "heading": "#d62728", "context (neighbour points)": "#1a9e3a",
               "kinematics": "#f28e2b", "topology": "#7a1b8c"}


def cv_top1(tr, feats, folds=3):
    X, y, g = tr[feats].to_numpy(float), tr["label"].to_numpy(), tr["trip_id"].to_numpy()
    accs = []
    for a, b in GroupKFold(n_splits=folds).split(X, y, g):
        m = make_model("lgbm", {"n_estimators": 300})
        m.fit(X[a], y[a])
        accs.append(observation_metrics(tr.iloc[b], m.predict_proba(X[b])[:, 1])["top1_accuracy"])
    return float(np.mean(accs))


def main():
    cfg = load_config()
    res = path(cfg, "results") / "features"; res.mkdir(parents=True, exist_ok=True)
    fig = path(cfg, "figures") / "features"; fig.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(path(cfg, "cache") / "features_train.parquet")
    tr = train[(train["ref_arc"] >= 0) & (train["obs"] % cfg["models"]["train_stride"] == 0)].reset_index(drop=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .3})

    # 1. gain importance
    model = make_model("lgbm", {"n_estimators": 300})
    model.fit(tr[FEATURES].to_numpy(float), tr["label"].to_numpy())
    gain = model.booster_.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": FEATURES, "group": [GROUP_OF[f] for f in FEATURES],
                        "gain_share": gain / gain.sum()}).sort_values("gain_share", ascending=False)
    imp.to_csv(res / "importance.csv", index=False)
    f, ax = plt.subplots(figsize=(8, 7))
    d = imp.iloc[::-1]
    ax.barh(d["feature"], d["gain_share"], color=[GROUP_COLOR[g] for g in d["group"]])
    handles = [Patch(color=c, label=g) for g, c in GROUP_COLOR.items()]
    ax.set_xlabel("share of total LightGBM gain"); ax.set_title("Feature importance (LightGBM, training trips)")
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    f.tight_layout(); f.savefig(fig / "importance.png", dpi=120); plt.close(f)
    print(imp.round(4).to_string(index=False))

    # 2. group ablation
    base = cv_top1(tr, FEATURES)
    rows = [{"setting": "all 23 features", "n_features": len(FEATURES), "cv_top1": base, "change": 0.0}]
    for g, fs in GROUPS.items():
        keep = [x for x in FEATURES if x not in fs]
        acc = cv_top1(tr, keep)
        rows.append({"setting": f"without {g}", "n_features": len(keep), "cv_top1": acc, "change": acc - base})
        print(f"without {g:28s} top-1 {acc:.4f} ({acc - base:+.4f})")
    only_dist = cv_top1(tr, ["dist", "dist_rank", "dist_minus_min", "dist_ratio"])
    rows.append({"setting": "distance features only", "n_features": 4, "cv_top1": only_dist, "change": only_dist - base})
    abl = pd.DataFrame(rows)
    abl.to_csv(res / "ablation.csv", index=False)
    f, ax = plt.subplots(figsize=(8, 4))
    d = abl.iloc[1:]
    ax.barh(d["setting"], 100 * d["change"], color=["#d62728" if c < 0 else "#1a9e3a" for c in d["change"]])
    ax.axvline(0, color="k", lw=0.8)
    for i, (c, a) in enumerate(zip(d["change"], d["cv_top1"])):
        ax.text(100 * c, i, f" {100 * c:+.2f} pts (top-1 {a:.3f})", va="center", ha="left" if c >= 0 else "right", fontsize=8)
    ax.set_xlabel("change in CV top-1 accuracy (percentage points)")
    ax.set_title(f"Feature-group ablation (all features: top-1 {base:.4f}, 3-fold group CV)")
    xmin = min(100 * d["change"].min() * 1.6, -1)
    ax.set_xlim(xmin, 1.5)
    f.tight_layout(); f.savefig(fig / "ablation.png", dpi=120); plt.close(f)

    # 3. correlation
    c = tr[FEATURES].sample(min(100_000, len(tr)), random_state=42).corr(method="spearman")
    f, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    ax.grid(False); f.colorbar(im, shrink=0.8, label="Spearman correlation")
    ax.set_title("Feature correlation (training candidates)")
    f.tight_layout(); f.savefig(fig / "correlation.png", dpi=120); plt.close(f)

    # 4. distributions of the strongest features
    top = imp["feature"].head(4).tolist()
    s = tr.sample(min(200_000, len(tr)), random_state=42)
    f, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 3.5))
    for ax, feat in zip(axes, top):
        lo, hi = np.percentile(s[feat], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50) if hi > lo else 20
        ax.hist(s.loc[s.label == 0, feat], bins=bins, density=True, alpha=.6, label="wrong candidate", color="#9a9a9a")
        ax.hist(s.loc[s.label == 1, feat], bins=bins, density=True, alpha=.6, label="reference arc", color="#d62728")
        ax.set_title(feat); ax.set_yticks([])
    axes[0].legend(fontsize=8)
    f.suptitle("Strongest features: reference arc vs other candidates")
    f.tight_layout(); f.savefig(fig / "distributions.png", dpi=120); plt.close(f)


if __name__ == "__main__":
    main()
