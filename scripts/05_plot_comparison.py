"""Plots and the report table for the model comparison.

Run:  python scripts/05_plot_comparison.py     (after 04_train_compare.py)

Reads  results/models/{comparison.csv, cv_folds.csv, details.json}
Writes figures/models/*.png and results/models/comparison_table.md
Every number comes from the CSV/JSON files written by 04_train_compare.py.
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
from pathsnap.config import load_config, path  # noqa: E402

NICE = {"logreg": "Logistic Reg.", "gnb": "Gaussian NB", "knn": "KNN", "tree": "Decision Tree",
        "rf": "Random Forest", "hgb": "HistGradBoost", "lgbm": "LightGBM", "mlp": "MLP"}


def main():
    cfg = load_config()
    res = path(cfg, "results") / "models"
    fig = path(cfg, "figures") / "models"; fig.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(res / "comparison.csv").sort_values("cv_top1", ascending=False).reset_index(drop=True)
    folds = pd.read_csv(res / "cv_folds.csv")
    det = json.load(open(res / "details.json"))
    order = comp["model"].tolist()
    names = [NICE.get(m, m) for m in order]
    sel = cfg["models"].get("selected")
    col = ["#d62728" if m == sel else "#1f6fd1" for m in order]
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .3})

    # 1. CV macro F1 with fold std
    f, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, comp["cv_f1_macro"], yerr=comp["cv_f1_macro_std"], capsize=4, color=col)
    ax.set_ylim(comp["cv_f1_macro"].min() - 0.03, min(1, comp["cv_f1_macro"].max() + 0.02))
    ax.set_ylabel("macro F1 (5-fold GroupKFold)"); ax.set_title("Cross-validation macro F1 (mean ± std over folds)")
    plt.setp(ax.get_xticklabels(), rotation=20)
    f.tight_layout(); f.savefig(fig / "cv_macro_f1.png", dpi=120); plt.close(f)

    # 2. fold stability of top-1
    f, ax = plt.subplots(figsize=(9, 4))
    data = [folds.loc[folds["model"] == m, "top1_accuracy"].to_numpy() for m in order]
    ax.boxplot(data, tick_labels=names)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i), d, s=14, color=col[i - 1], zorder=3)
    ax.set_ylabel("top-1 accuracy per fold"); ax.set_title("Stability across validation folds")
    plt.setp(ax.get_xticklabels(), rotation=20)
    f.tight_layout(); f.savefig(fig / "cv_top1_folds.png", dpi=120); plt.close(f)

    # 3. student-test top-1 and macro F1
    x = np.arange(len(order))
    f, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, comp["test_top1"], 0.4, label="top-1 accuracy (observation level)", color="#1f6fd1")
    ax.bar(x + 0.2, comp["test_f1_macro"], 0.4, label="macro F1 (candidate level)", color="#f28e2b")
    ax.set_xticks(x, names, rotation=20)
    lo = min(comp["test_top1"].min(), comp["test_f1_macro"].min())
    ax.set_ylim(lo - 0.03, 1.0); ax.legend(loc="lower right")
    ax.set_title("Student-testing trips: top-1 accuracy and macro F1")
    f.tight_layout(); f.savefig(fig / "student_test_top1_f1.png", dpi=120); plt.close(f)

    # 4. log loss
    f, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, comp["cv_log_loss"], 0.4, label="CV", color="#9a9a9a")
    ax.bar(x + 0.2, comp["test_log_loss"], 0.4, label="student test", color="#1f6fd1")
    ax.set_xticks(x, names, rotation=20); ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Probability quality (log loss)"); ax.legend()
    f.tight_layout(); f.savefig(fig / "log_loss.png", dpi=120); plt.close(f)

    # 5. compute cost
    f, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, comp["fit_time_s"], color=col); axes[0].set_yscale("log")
    axes[0].set_title("Training time on all training rows (s, log)")
    axes[1].bar(names, comp["inference_ms_per_1k_obs"], color=col); axes[1].set_yscale("log")
    axes[1].set_title("Inference time (ms per 1000 observations, log)")
    for a in axes:
        plt.setp(a.get_xticklabels(), rotation=25)
    f.tight_layout(); f.savefig(fig / "compute_cost.png", dpi=120); plt.close(f)

    # 6. route continuity of classifier-only predictions
    f, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, comp["test_route_continuity"], color=col)
    ax.set_ylim(comp["test_route_continuity"].min() - 0.05, 1.0)
    ax.set_ylabel("route continuity"); ax.set_title("Route continuity of classifier-only top-1 predictions (student test)")
    plt.setp(ax.get_xticklabels(), rotation=20)
    f.tight_layout(); f.savefig(fig / "route_continuity.png", dpi=120); plt.close(f)

    # 7. over/under-fitting: in-sample vs CV vs student test
    f, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.27, comp["train_top1_in_sample"], 0.27, label="training (in-sample)", color="#9a9a9a")
    ax.bar(x, comp["cv_top1"], 0.27, label="cross-validation", color="#f28e2b")
    ax.bar(x + 0.27, comp["test_top1"], 0.27, label="student test", color="#1f6fd1")
    ax.set_xticks(x, names, rotation=20); ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("top-1 accuracy"); ax.set_title("Generalisation gap (KNN in-sample is not meaningful: it memorises)")
    ax.legend(loc="upper right", fontsize=8)
    f.tight_layout(); f.savefig(fig / "generalisation_gap.png", dpi=120); plt.close(f)

    # 8. confusion matrices (student test, threshold 0.5, row-normalised)
    n = len(order)
    f, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(3.2 * np.ceil(n / 2), 6.2))
    for ax, m in zip(np.ravel(axes), order):
        cm = np.array(det[m]["test_confusion_matrix"], float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center",
                        color="white" if cmn[i, j] > .5 else "black", fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(NICE.get(m, m), fontsize=10); ax.grid(False)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    f.suptitle("Candidate-level confusion matrices, student test (row-normalised)")
    f.tight_layout(); f.savefig(fig / "confusion_matrices.png", dpi=120); plt.close(f)

    # 9. training histories, only for models that produce them
    hist = {m: det[m]["history"] for m in order if det[m].get("history")}
    if hist:
        f, axes = plt.subplots(1, len(hist), figsize=(5 * len(hist), 3.8))
        axes = np.atleast_1d(axes)
        for ax, (m, h) in zip(axes, hist.items()):
            ax.plot(h["train"], label="train loss")
            if "valid" in h:
                ax.plot(h["valid"], label="validation loss")
                ax.set_ylabel(h["metric"])
            if "valid_accuracy" in h:
                ax2 = ax.twinx()
                ax2.plot(h["valid_accuracy"], color="#f28e2b", label="validation accuracy")
                ax2.set_ylabel("validation accuracy"); ax2.grid(False)
                ax2.legend(loc="center right", fontsize=8)
                ax.set_ylabel("training log loss")
            ax.set_xlabel(h["x_label"]); ax.set_title(NICE.get(m, m)); ax.legend(loc="upper right", fontsize=8)
        f.suptitle("Training curves (final fit, models with iteration-wise histories only)")
        f.tight_layout(); f.savefig(fig / "training_curves.png", dpi=120); plt.close(f)

    # report table (markdown)
    lines = ["| Model | Algorithm family | Key configuration | CV macro F1 | Top-1 accuracy (student test) | Route continuity | Student-test macro F1 | Log loss (student test) | Inference (ms / 1k obs) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        lines.append(f"| {NICE.get(r['model'], r['model'])} | {r['family']} | {r['config']} | "
                     f"{r['cv_f1_macro']:.4f} ± {r['cv_f1_macro_std']:.4f} | {r['test_top1']:.4f} | "
                     f"{r['test_route_continuity']:.4f} | {r['test_f1_macro']:.4f} | {r['test_log_loss']:.4f} | "
                     f"{r['inference_ms_per_1k_obs']:.1f} |")
    (res / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
