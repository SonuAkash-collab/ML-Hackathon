"""Feature engineering for (GPS observation, candidate arc) pairs.

Each row of the feature table is one candidate of one observation. The model
predicts P(candidate is the reference arc | features). Feature groups:

geometry     distance from the GPS point to the arc, how it compares with the
             other candidates of the same point, where the projection falls
heading      difference between the direction of travel (from neighbouring GPS
             points) and the direction of the (directed) arc
context      average / max distance of the neighbouring GPS points (i-2..i+2)
             to the same arc: a single noisy point can be closer to a side
             street, but its neighbours usually are not
kinematics   speed and sampling interval (heading is unreliable when stopped)
topology     arc length, node degrees, whether the street is two-way

Only the GPS track and the road network are used. The reference route is
never an input to any feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .geo import Projector, angle_diff, bearing_xy, point_segment
from .io import Trip, clean_track
from .labels import reference_arcs
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "head_diff", "head_diff_wide", "cos_head_wide", "head_rank",
    "win_dist_mean", "win_dist_max", "win_rank",
    "t", "outside",
    "speed", "dt_prev", "dt_next", "move_wide",
    "arc_length", "out_deg_end", "in_deg_start", "has_reverse", "expanded",
]


def kinematics(px, py, t, wide: int = 3) -> dict:
    n = len(px)
    idx = np.arange(n)
    prev_i = np.maximum(idx - 1, 0)
    next_i = np.minimum(idx + 1, n - 1)
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    med = np.nanmedian(np.diff(t)) if n > 1 else 1.0
    dt_prev = np.where(np.isnan(dt_prev), med, dt_prev)
    dt_next = np.where(np.isnan(dt_next), med, dt_next)
    span_t = np.maximum(t[next_i] - t[prev_i], 1e-6)
    disp = np.hypot(px[next_i] - px[prev_i], py[next_i] - py[prev_i])
    speed = disp / span_t
    head = bearing_xy(px[next_i] - px[prev_i], py[next_i] - py[prev_i])
    lo = np.maximum(idx - wide, 0)
    hi = np.minimum(idx + wide, n - 1)
    head_w = bearing_xy(px[hi] - px[lo], py[hi] - py[lo])
    move_w = np.hypot(px[hi] - px[lo], py[hi] - py[lo])
    return dict(dt_prev=dt_prev, dt_next=dt_next, speed=speed, head=head, head_w=head_w, move_w=move_w)


def candidate_features(px, py, t, net: RoadNetwork, cand_cfg: dict | None = None) -> pd.DataFrame:
    cand_cfg = cand_cfg or {}
    df = generate_candidates(px, py, net, **cand_cfg)
    if df.empty:
        return df
    kin = kinematics(px, py, t)
    o = df["obs"].to_numpy()
    a = df["arc"].to_numpy()
    g = df.groupby("obs")["dist"]
    dmin = g.transform("min").to_numpy()
    df["dist_rank"] = df["rank"]
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio"] = (df["dist"] + 1.0) / (dmin + 1.0)
    df["n_cands"] = g.transform("size").to_numpy()

    arc_bear = net.bearing[a]
    df["head_diff"] = angle_diff(kin["head"][o], arc_bear)
    df["head_diff_wide"] = angle_diff(kin["head_w"][o], arc_bear)
    df["cos_head_wide"] = np.cos(np.radians(df["head_diff_wide"].to_numpy()))
    df["head_rank"] = df.groupby("obs")["head_diff_wide"].rank(method="min").to_numpy() - 1

    n = len(px)
    wd = []
    for off in (-2, -1, 1, 2):
        j = np.clip(o + off, 0, n - 1)
        wd.append(point_segment(px[j], py[j], net.ax[a], net.ay[a], net.bx[a], net.by[a])[0])
    wd = np.column_stack(wd + [df["dist"].to_numpy()])
    df["win_dist_mean"] = wd.mean(axis=1)
    df["win_dist_max"] = wd.max(axis=1)
    df["win_rank"] = df.groupby("obs")["win_dist_mean"].rank(method="min").to_numpy() - 1

    df["outside"] = ((df["t_raw"] < 0) | (df["t_raw"] > 1)).astype(int)
    df["speed"] = kin["speed"][o]
    df["dt_prev"] = kin["dt_prev"][o]
    df["dt_next"] = kin["dt_next"][o]
    df["move_wide"] = kin["move_w"][o]
    df["arc_length"] = net.length[a]
    df["out_deg_end"] = net.out_degree[net.arc_to[a]]
    df["in_deg_start"] = net.in_degree[net.arc_from[a]]
    df["has_reverse"] = net.has_reverse[a].astype(int)
    df["pos"] = df["t"] * df["arc_length"]          # metres along the arc (used by the HMM)
    return df


def prepare_trip(trip: Trip, cand_cfg: dict | None = None, sample_spacing: float = 5.0):
    """Clean + project a trip and build its network and candidate features.

    Returns (features_df, context) where context holds the arrays the decoder
    and the visualisations need.
    """
    tr, clean_report = clean_track(trip.track)
    lon0 = float(np.median(tr[:, 0])) if len(tr) else float(np.median(trip.nodes[:, 0]))
    lat0 = float(np.median(tr[:, 1])) if len(tr) else float(np.median(trip.nodes[:, 1]))
    proj = Projector(lon0, lat0)
    net = RoadNetwork(trip.nodes, trip.arcs, proj, sample_spacing=sample_spacing)
    px, py = proj.to_xy(tr[:, 0], tr[:, 1])
    t = tr[:, 2]
    feats = candidate_features(px, py, t, net, cand_cfg)
    if not feats.empty:
        feats.insert(0, "trip_id", trip.trip_id)
        feats["orig_idx"] = tr[feats["obs"].to_numpy(), 3].astype(int)
    ctx = dict(track=tr, px=px, py=py, t=t, net=net, projector=proj, clean_report=clean_report)
    return feats, ctx


def attach_labels(feats: pd.DataFrame, ctx: dict, route: np.ndarray, max_dist: float = 60.0):
    """Add the reference arc / binary label columns (training & scoring only)."""
    ref_arc, ref_pos, ref_dist = reference_arcs(ctx["px"], ctx["py"], route, ctx["net"], max_dist)
    ctx["ref_arc"] = ref_arc
    ctx["ref_dist"] = ref_dist
    if feats.empty:
        return feats
    o = feats["obs"].to_numpy()
    feats["ref_arc"] = ref_arc[o]
    feats["label"] = (feats["arc"].to_numpy() == ref_arc[o]).astype(int)
    return feats
