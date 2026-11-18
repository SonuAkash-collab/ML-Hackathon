"""Reading raw PathSnap trip folders.

Every trip lives in its own folder ``<split>/<trip_id>/`` containing

    <trip_id>.track  lon  lat  t_seconds        (GPS observations, no header)
    <trip_id>.nodes  lon  lat                   (row index = node id)
    <trip_id>.arcs   from_node  to_node         (row index = arc id, directed)
    <trip_id>.route  arc_id                     (reference route, in travel order)

The .route file is optional: the prediction pipeline never reads it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRIP_FILES = ("track", "nodes", "arcs", "route")


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (n, 3) lon, lat, t
    nodes: np.ndarray          # (N, 2) lon, lat
    arcs: np.ndarray           # (A, 2) from_node, to_node
    route: np.ndarray | None = None   # (R,) arc ids, None when not loaded
    meta: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.track)


def _load_numeric(path: Path, ncols: int, dtype=float) -> np.ndarray:
    """Load a whitespace separated file, always returning a 2-D (or 1-D) array.

    Empty files return an empty array of the right shape instead of failing.
    """
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype) if ncols > 1 else np.empty(0, dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols == 1:
        return arr.reshape(-1)
    return arr[:, :ncols]


def list_trips(split_dir: str | os.PathLike) -> list[str]:
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def load_track(trip_dir: Path, trip_id: str) -> np.ndarray:
    track = _load_numeric(trip_dir / f"{trip_id}.track", 3)
    return track


def load_trip(split_dir: str | os.PathLike, trip_id: str, with_route: bool = True) -> Trip:
    """Load one trip.

    ``with_route=False`` is what the prediction pipeline uses, so the
    reference route can never leak into features or predictions.
    """
    trip_dir = Path(split_dir) / trip_id
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"trip folder not found: {trip_dir}")
    track = load_track(trip_dir, trip_id)
    nodes = _load_numeric(trip_dir / f"{trip_id}.nodes", 2)
    arcs = _load_numeric(trip_dir / f"{trip_id}.arcs", 2, dtype=np.int64).astype(np.int64)
    route = None
    if with_route:
        route = _load_numeric(trip_dir / f"{trip_id}.route", 1, dtype=np.int64).astype(np.int64)
    return Trip(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


def load_route(split_dir: str | os.PathLike, trip_id: str) -> np.ndarray:
    """Reference route only (used server-side for scoring)."""
    return _load_numeric(Path(split_dir) / trip_id / f"{trip_id}.route", 1, dtype=np.int64).astype(np.int64)


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Basic GPS cleaning.

    * drop rows with NaN / inf
    * drop rows with impossible coordinates
    * sort by timestamp and drop duplicate timestamps (keep the first)

    Returns the cleaned track and a small report of what was removed. The
    original row index is kept in column 3 so predictions can be mapped back.
    """
    n0 = len(track)
    idx = np.arange(n0, dtype=float)
    tr = np.column_stack([track[:, :3], idx]) if n0 else np.empty((0, 4))
    finite = np.isfinite(tr[:, :3]).all(axis=1)
    valid = finite & (np.abs(tr[:, 0]) <= 180) & (np.abs(tr[:, 1]) <= 90)
    tr = tr[valid]
    order = np.argsort(tr[:, 2], kind="stable")
    tr = tr[order]
    if len(tr):
        keep = np.r_[True, np.diff(tr[:, 2]) > 0]
        n_dup = int((~keep).sum())
        tr = tr[keep]
    else:
        n_dup = 0
    report = {
        "n_raw": n0,
        "n_invalid": int((~valid).sum()),
        "n_duplicate_time": n_dup,
        "was_unsorted": bool(len(track) > 1 and np.any(np.diff(track[:, 2]) < 0)),
        "n_clean": len(tr),
    }
    return tr, report
