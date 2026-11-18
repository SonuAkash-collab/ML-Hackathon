"""Turning the .route file into a label for every GPS observation.

The .route file only lists the arcs the vehicle drove, in order; it does not say
which arc each GPS point belongs to (there are ~3-5 GPS points per route arc).
We assign every observation to one route position with a monotone alignment:

    choose j_0 <= j_1 <= ... <= j_n  (indices into the route)
    minimising  sum_i  dist(gps_i, route_arc[j_i])

so that the assignment follows the direction of travel (this matters where the
route uses the same road twice, or both directions of a street). The DP only
considers route arcs within ``max_dist`` metres of a point; points with no
route arc nearby stay unlabelled (-1) and are excluded from scoring.

Note: the reference route is itself produced by an algorithm, so these labels
are "reference" labels, not verified physical ground truth.
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


def reference_arcs(px: np.ndarray, py: np.ndarray, route: np.ndarray, net: RoadNetwork,
                   max_dist: float = 60.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ref_arc, ref_route_pos, ref_dist) for each observation (-1 / nan if none)."""
    n = len(px)
    ref_arc = np.full(n, -1, dtype=np.int64)
    ref_pos = np.full(n, -1, dtype=np.int64)
    ref_dist = np.full(n, np.nan)
    if n == 0 or len(route) == 0:
        return ref_arc, ref_pos, ref_dist

    route = np.asarray(route, dtype=np.int64)
    valid_route = (route >= 0) & (route < net.n_arcs)
    positions_of: dict[int, list[int]] = {}
    for j, a in enumerate(route):
        if valid_route[j]:
            positions_of.setdefault(int(a), []).append(j)
    route_set = np.fromiter(positions_of.keys(), dtype=np.int64)

    near = net.arcs_near(px, py, max_dist)
    states, costs, back = [], [], []
    prev_pos = prev_cummin = prev_argmin = None
    prev_point = -1
    point_ids = []
    for i in range(n):
        arcs = near[i][np.isin(near[i], route_set)]
        if len(arcs) == 0:
            continue
        d = net.project(px[i], py[i], arcs)[0]
        keep = d <= max_dist
        arcs, d = arcs[keep], d[keep]
        if len(arcs) == 0:
            continue
        pos = np.concatenate([positions_of[int(a)] for a in arcs])
        dd = np.concatenate([[d[k]] * len(positions_of[int(a)]) for k, a in enumerate(arcs)])
        o = np.argsort(pos)
        pos, dd = pos[o], dd[o]
        if prev_pos is None:
            c = dd.copy()
            bp = np.full(len(pos), -1)
        else:
            k = np.searchsorted(prev_pos, pos, side="right") - 1
            ok = k >= 0
            c = np.full(len(pos), np.inf)
            bp = np.full(len(pos), -1)
            c[ok] = dd[ok] + prev_cummin[k[ok]]
            bp[ok] = prev_argmin[k[ok]]
            if not np.isfinite(c).any():
                # this point only matches route arcs *behind* the current
                # position -> treat as noise, leave it unlabelled
                continue
        states.append(pos)
        costs.append(c)
        back.append(bp)
        point_ids.append(i)
        cm = np.minimum.accumulate(c)
        am = np.zeros(len(c), dtype=np.int64)
        for t in range(1, len(c)):
            am[t] = t if c[t] < cm[t - 1] else am[t - 1]
        prev_pos, prev_cummin, prev_argmin = pos, cm, am
        prev_point = i

    if not point_ids:
        return ref_arc, ref_pos, ref_dist
    s = int(np.argmin(costs[-1]))
    for r in range(len(point_ids) - 1, -1, -1):
        i = point_ids[r]
        j = int(states[r][s])
        ref_pos[i] = j
        ref_arc[i] = route[j]
        s = int(back[r][s])
        if s < 0 and r > 0:
            break
    # distances for the labelled points
    lab = ref_arc >= 0
    if lab.any():
        ref_dist[lab] = net.project(px[lab], py[lab], ref_arc[lab])[0]
    return ref_arc, ref_pos, ref_dist
