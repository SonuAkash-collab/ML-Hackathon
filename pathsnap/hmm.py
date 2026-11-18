"""HMM / Viterbi route reconstruction.

Hidden states
    For observation k the hidden state is "the vehicle is on candidate arc j of
    observation k, at the projected position pos_j along that arc". States
    therefore keep the candidate -> road-segment correspondence exactly.

Emission
    log e_k(j) = w * log( max(p_kj / sum_j p_kj, floor) )
    where p_kj is the classifier probability; normalising per observation turns
    independent binary scores into a distribution over the candidate set.

Transition (Newson & Krumm 2009 style)
    d_route = driving distance on the directed network from (arc i, pos_i) to
              (arc j, pos_j), found with a bounded Dijkstra
    d_gc    = straight-line distance between the two GPS points
    log a(i->j) = -|d_route - d_gc| / beta  - uturn_penalty * [j is reverse of i]
    Moving backwards on the same arc by more than a few metres (GPS jitter) is
    not allowed; switching to the reverse arc of a two-way street is allowed
    anywhere along the arc but pays ``uturn_penalty``.
    Transitions that are unreachable within the search cutoff get -inf.

Decoding
    Standard Viterbi recurrence in log space:
        V_k(j) = max_i [ V_{k-1}(i) + log a(i->j) ] + log e_k(j)
    The chain is cut (and decoding restarts from the emissions) when
      * the time gap between observations exceeds ``gap_seconds`` (GPS dropout),
      * or every transition is -inf (disconnected candidates / broken network).
    Observations without candidates are skipped and get no prediction (-1).

The expensive part (network distances) does not depend on the tunable
parameters, so it is computed once in ``build_lattice`` and ``viterbi`` can be
re-run cheaply for parameter tuning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .network import RoadNetwork


@dataclass
class HMMParams:
    beta: float = 10.0
    emission_weight: float = 1.0
    prob_floor: float = 1e-4
    uturn_penalty: float = 2.0
    gap_seconds: float = 60.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Step:
    obs: int
    arcs: np.ndarray
    pos: np.ndarray
    prob: np.ndarray
    dt: float                 # seconds since previous step (inf for the first one)
    gc: float                 # straight-line metres since previous step
    D: np.ndarray | None      # (n_prev, n_cur) route distances, None if not computed
    uturn: np.ndarray | None  # (n_prev, n_cur) bool


def build_lattice(cands: pd.DataFrame, prob: np.ndarray, px, py, t, net: RoadNetwork,
                  max_gap_seconds: float = 120.0, max_speed: float = 50.0,
                  backward_tol: float = 10.0) -> list[Step]:
    df = cands[["obs", "arc", "pos"]].copy()
    df["p"] = prob
    steps: list[Step] = []
    prev = None
    for obs, g in df.groupby("obs", sort=True):
        arcs = g["arc"].to_numpy(np.int64)
        pos = g["pos"].to_numpy(float)
        p = g["p"].to_numpy(float)
        if prev is None:
            steps.append(Step(int(obs), arcs, pos, p, np.inf, np.inf, None, None))
            prev = steps[-1]
            continue
        dt = float(t[obs] - t[prev.obs])
        gc = float(np.hypot(px[obs] - px[prev.obs], py[obs] - py[prev.obs]))
        D = U = None
        if dt <= max_gap_seconds:
            cutoff = min(3.0 * gc + 150.0, max_speed * max(dt, 1.0) + 150.0)
            D = np.full((len(prev.arcs), len(arcs)), np.inf)
            U = np.zeros_like(D, dtype=bool)
            for i, (a, sa) in enumerate(zip(prev.arcs, prev.pos)):
                for j, (b, sb) in enumerate(zip(arcs, pos)):
                    d = net.route_distance(int(a), float(sa), int(b), float(sb), cutoff, backward_tol)
                    if net.is_reverse(int(a), int(b)):
                        # two-way street: allow turning around mid-arc. Position sa on a
                        # is position (L - sa) on its reverse arc b.
                        U[i, j] = True
                        d_turn = sb - (net.length[b] - sa)
                        if d_turn >= -backward_tol:
                            d = min(d, max(d_turn, 0.0))
                    D[i, j] = d
        steps.append(Step(int(obs), arcs, pos, p, dt, gc, D, U))
        prev = steps[-1]
    return steps


def viterbi(steps: list[Step], params: HMMParams) -> tuple[dict, dict]:
    """Return ({obs: arc}, info) for the most likely state sequence."""
    if not steps:
        return {}, {"n_segments": 0, "n_breaks_gap": 0, "n_breaks_disconnected": 0}
    w, floor = params.emission_weight, params.prob_floor
    emis = []
    for s in steps:
        q = s.prob / max(s.prob.sum(), 1e-12)
        emis.append(w * np.log(np.maximum(q, floor)))

    segments = []           # list of (start, end) step indices (inclusive)
    backs: list[np.ndarray | None] = [None] * len(steps)
    scores = emis[0].copy()
    start = 0
    n_gap = n_disc = 0
    finals = {}
    for k in range(1, len(steps)):
        s = steps[k]
        brk = s.D is None or s.dt > params.gap_seconds
        if not brk:
            T = -np.abs(s.D - s.gc) / params.beta - params.uturn_penalty * s.uturn
            cand = scores[:, None] + T
            best_prev = np.argmax(cand, axis=0)
            best = cand[best_prev, np.arange(cand.shape[1])]
            if not np.isfinite(best).any():
                brk = True
                n_disc += 1
            else:
                scores = best + emis[k]
                backs[k] = best_prev
        else:
            n_gap += 1
        if brk:
            finals[(start, k - 1)] = scores
            segments.append((start, k - 1))
            start = k
            scores = emis[k].copy()
            backs[k] = None
    finals[(start, len(steps) - 1)] = scores
    segments.append((start, len(steps) - 1))

    path = {}
    for (a, b) in segments:
        sc = finals[(a, b)]
        j = int(np.argmax(sc))
        for k in range(b, a - 1, -1):
            path[steps[k].obs] = int(steps[k].arcs[j])
            if k > a:
                j = int(backs[k][j])   # every step after a segment start has back-pointers
    info = {"n_segments": len(segments), "n_breaks_gap": n_gap, "n_breaks_disconnected": n_disc}
    return path, info


def classifier_path(steps: list[Step]) -> dict:
    return {s.obs: int(s.arcs[int(np.argmax(s.prob))]) for s in steps}


def fill_route(net: RoadNetwork, obs_arcs: list[int], cutoff: float = 3000.0) -> list[int]:
    """Turn the per-observation arc sequence into a continuous list of arcs.

    Consecutive distinct arcs are joined with the shortest network path between
    them; if no path exists (segment break) the route simply continues.
    """
    seq = [a for a in obs_arcs if a is not None and a >= 0]
    route: list[int] = []
    for a in seq:
        if route and route[-1] == a:
            continue
        if route:
            mid = net.shortest_arc_path(route[-1], a, cutoff)
            if mid:
                route.extend(mid)
        route.append(int(a))
    return route


def path_to_array(path: dict, n_obs: int) -> np.ndarray:
    out = np.full(n_obs, -1, dtype=np.int64)
    for o, a in path.items():
        out[o] = a
    return out
