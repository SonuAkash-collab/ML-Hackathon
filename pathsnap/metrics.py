"""Metrics for candidate classification and route matching.

Candidate level (one row = one candidate of one observation, threshold 0.5)
    accuracy, precision, recall, F1 (macro over the two classes and for the
    positive class), log loss, confusion matrix

Observation level
    top1_accuracy      for every labelled observation, is the highest-scoring
                       candidate the reference arc?  Observations whose
                       reference arc is not among the candidates count as wrong.
    candidate_miss_rate  share of labelled observations whose reference arc is
                       not in the candidate set (upper bound on top-1 = 1 - miss)

Sequence level (needs the road network)
    route_continuity   share of consecutive *different* predicted arcs where
                       the second can be reached from the first in <= 3 arcs
    route_precision / recall / f1   predicted route arcs vs the .route arcs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

EPS = 1e-7


def candidate_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    yhat = (p >= thr).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
        "precision_pos": precision_score(y, yhat, zero_division=0),
        "recall_pos": recall_score(y, yhat, zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "confusion_matrix": cm.tolist(),
    }


def top1_table(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, observation)."""
    tmp = df[["trip_id", "obs", "arc"]].copy()
    if "ref_arc" in df:
        tmp["ref_arc"] = df["ref_arc"].to_numpy()
    tmp["p"] = p
    idx = tmp.groupby(["trip_id", "obs"], sort=False)["p"].idxmax()
    out = tmp.loc[idx].rename(columns={"arc": "pred_arc"}).sort_values(["trip_id", "obs"])
    return out.reset_index(drop=True)


def observation_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    top = top1_table(df, p)
    lab = top[top["ref_arc"] >= 0]
    hit = df[df["ref_arc"] >= 0].groupby(["trip_id", "obs"])["label"].max()
    return {
        "top1_accuracy": float((lab["pred_arc"] == lab["ref_arc"]).mean()) if len(lab) else np.nan,
        "candidate_miss_rate": float(1 - hit.mean()) if len(hit) else np.nan,
        "n_labelled_obs": int(len(lab)),
    }


def all_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    m = candidate_metrics(df["label"].to_numpy(), p)
    m.update(observation_metrics(df, p))
    return m


# ------------------------------------------------------------ sequence level
def route_continuity(arc_seq, net, max_hops: int = 3) -> float:
    seq = [int(a) for a in arc_seq if a is not None and a >= 0]
    pairs = [(a, b) for a, b in zip(seq[:-1], seq[1:]) if a != b]
    if not pairs:
        return 1.0
    ok = sum(net.connected_within(a, b, max_hops) for a, b in pairs)
    return ok / len(pairs)


def route_set_scores(pred_arcs, ref_route) -> dict:
    pred = {int(a) for a in pred_arcs if a is not None and a >= 0}
    ref = {int(a) for a in ref_route}
    tp = len(pred & ref)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(ref) if ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def pointwise_accuracy(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = np.asarray(pred), np.asarray(ref)
    m = ref >= 0
    return float((pred[m] == ref[m]).mean()) if m.any() else np.nan
