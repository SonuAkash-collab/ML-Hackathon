"""Candidate road segments for each GPS observation.

For every observation we keep the ``k`` arcs whose straight-line segment is
closest to the point, provided it is within ``radius`` metres. If nothing is
within ``radius`` (bad GPS fix, tunnel, ...), we retry with ``fallback_radius``
and ``fallback_k`` so that the decoder still has something to work with; those
rows are flagged with ``expanded = 1``.

The radius was chosen from the EDA: 99 % of training observations are within
~16 m of their reference arc and the worst case we saw was ~27 m, so 50 m
leaves a comfortable margin while keeping candidate sets small.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import RoadNetwork


def generate_candidates(px: np.ndarray, py: np.ndarray, net: RoadNetwork, radius: float = 50.0,
                        k: int = 8, fallback_radius: float = 150.0, fallback_k: int = 4) -> pd.DataFrame:
    n = len(px)
    near = net.arcs_near(px, py, radius)
    obs_l, arc_l, exp_l = [], [], []
    missing = [i for i in range(n) if len(near[i]) == 0]
    fb = {}
    if missing:
        m = np.array(missing)
        fb_near = net.arcs_near(px[m], py[m], fallback_radius)
        fb = dict(zip(missing, fb_near))
    for i in range(n):
        arcs = near[i]
        expanded = 0
        if len(arcs) == 0:
            arcs = fb.get(i, np.empty(0, dtype=np.int64))
            expanded = 1
        if len(arcs) == 0:
            continue
        obs_l.append(np.full(len(arcs), i))
        arc_l.append(arcs)
        exp_l.append(np.full(len(arcs), expanded))
    if not obs_l:
        return pd.DataFrame(columns=["obs", "arc", "dist", "t", "t_raw", "qx", "qy", "expanded"])
    obs = np.concatenate(obs_l)
    arc = np.concatenate(arc_l)
    expd = np.concatenate(exp_l)
    d, t, t_raw, qx, qy = net.project(px[obs], py[obs], arc)
    df = pd.DataFrame({"obs": obs, "arc": arc, "dist": d, "t": t, "t_raw": t_raw,
                       "qx": qx, "qy": qy, "expanded": expd})
    lim = np.where(df["expanded"].to_numpy() == 1, fallback_radius, radius)
    df = df[df["dist"].to_numpy() <= lim]
    df = df.sort_values(["obs", "dist"], kind="stable")
    df["rank"] = df.groupby("obs").cumcount()
    kk = np.where(df["expanded"].to_numpy() == 1, fallback_k, k)
    df = df[df["rank"].to_numpy() < kk].reset_index(drop=True)
    return df
