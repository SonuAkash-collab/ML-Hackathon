"""Exploratory data analysis of the raw dataset.

Run:  python scripts/01_eda.py

Outputs
  results/eda/inventory.csv        one row per trip (train + student_test)
  results/eda/summary.json         partition-level summary + data-quality notes
  figures/eda/*.png                plots used in section 2 of the report

Only the train and student_test partitions are touched here.
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
from pathsnap.geo import Projector, haversine_m, utm_epsg_for  # noqa: E402
from pathsnap.io import clean_track, list_trips, load_trip  # noqa: E402
from pathsnap.labels import reference_arcs  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402


def inspect_trip(split: str, split_dir: Path, tid: str, cfg: dict):
    trip = load_trip(split_dir, tid)
    d = split_dir / tid
    sizes = {f"size_{ext}_kb": round((d / f"{tid}.{ext}").stat().st_size / 1024, 1)
             if (d / f"{tid}.{ext}").exists() else np.nan for ext in ("track", "nodes", "arcs", "route")}
    tr, rep = clean_track(trip.track)
    dt = np.diff(tr[:, 2]) if len(tr) > 1 else np.array([np.nan])
    proj = Projector(float(np.median(tr[:, 0])), float(np.median(tr[:, 1])))
    net = RoadNetwork(trip.nodes, trip.arcs, proj, cfg["network"]["sample_spacing_m"])
    px, py = proj.to_xy(tr[:, 0], tr[:, 1])
    step = np.hypot(np.diff(px), np.diff(py))
    ncomp, comp = net.connected_components()
    comp_sizes = np.bincount(comp)
    isolated = int(((net.in_degree + net.out_degree) == 0).sum())
    route = trip.route if trip.route is not None else np.empty(0, dtype=int)
    route_valid = route[(route >= 0) & (route < net.n_arcs)]
    route_conn = np.mean(net.arc_to[route_valid[:-1]] == net.arc_from[route_valid[1:]]) if len(route_valid) > 1 else np.nan
    ref_arc, _, ref_dist = reference_arcs(px, py, route_valid, net, cfg["labels"]["max_dist"])
    # projection sanity check: projected distance vs haversine for consecutive points
    hv = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    proj_err = np.nanmax(np.abs(step - hv) / np.maximum(hv, 1.0)) if len(hv) else np.nan
    row = dict(
        split=split, trip_id=tid, n_obs_raw=rep["n_raw"], n_obs=rep["n_clean"],
        n_invalid=rep["n_invalid"], n_dup_time=rep["n_duplicate_time"], unsorted=rep["was_unsorted"],
        duration_s=float(tr[-1, 2] - tr[0, 2]) if len(tr) else 0.0,
        dt_median=float(np.median(dt)), dt_frac_1s=float(np.mean(dt == 1.0)),
        dt_max=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt60s=int((dt > 60).sum()),
        step_m_median=float(np.median(step)) if len(step) else np.nan,
        step_m_max=float(step.max()) if len(step) else np.nan,
        speed_kmh_p95=float(np.percentile(step / np.maximum(dt, 1e-6), 95) * 3.6) if len(step) else np.nan,
        length_km=float(step.sum() / 1000),
        extent_ew_km=float((px.max() - px.min()) / 1000), extent_ns_km=float((py.max() - py.min()) / 1000),
        lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
        n_nodes=net.n_nodes, n_arcs=net.n_arcs, arc_len_median=float(np.median(net.length)),
        arc_len_max=float(net.length.max()), frac_two_way=float(net.has_reverse.mean()),
        n_components=int(ncomp), largest_comp_frac=float(comp_sizes.max() / net.n_nodes),
        isolated_nodes=isolated,
        route_len=len(route), route_invalid_ids=int(len(route) - len(route_valid)),
        route_connected_frac=float(route_conn), route_unique_frac=float(len(np.unique(route_valid)) / max(len(route_valid), 1)),
        route_length_km=float(net.length[route_valid].sum() / 1000),
        frac_obs_labelled=float((ref_arc >= 0).mean()),
        ref_dist_median=float(np.nanmedian(ref_dist)), ref_dist_p95=float(np.nanpercentile(ref_dist, 95)),
        ref_dist_max=float(np.nanmax(ref_dist)), proj_rel_err_max=float(proj_err),
        **sizes,
    )
    return row, dt, ref_dist, tr


def main():
    cfg = load_config()
    out = path(cfg, "results") / "eda"
    fig = path(cfg, "figures") / "eda"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    rows, all_dt, all_ref, tracks = [], {}, {}, {}
    for split in ("train", "student_test"):
        sdir = path(cfg, split)
        trips = list_trips(sdir)
        print(f"{split}: {len(trips)} trips in {sdir}")
        all_dt[split], all_ref[split] = [], []
        for tid in trips:
            row, dt, ref_dist, tr = inspect_trip(split, sdir, tid, cfg)
            rows.append(row)
            all_dt[split].append(dt)
            all_ref[split].append(ref_dist[~np.isnan(ref_dist)])
            tracks[(split, tid)] = tr
            print(f"  {tid}: {row['n_obs']:6d} obs  max gap {row['dt_max']:9.0f}s  "
                  f"{row['n_arcs']:7d} arcs  labelled {row['frac_obs_labelled']:.3f}")
    inv = pd.DataFrame(rows)
    inv.to_csv(out / "inventory.csv", index=False)

    # --------------------------------------------------------------- summary
    summary = {}
    for split, g in inv.groupby("split"):
        dt = np.concatenate(all_dt[split])
        ref = np.concatenate(all_ref[split])
        summary[split] = {
            "n_trips": int(len(g)),
            "n_obs_total": int(g["n_obs"].sum()),
            "obs_per_trip": g["n_obs"].describe().round(1).to_dict(),
            "missing_or_invalid_rows": int(g["n_invalid"].sum()),
            "duplicate_timestamps": int(g["n_dup_time"].sum()),
            "unsorted_trips": int(g["unsorted"].sum()),
            "frac_dt_exactly_1s": float(np.mean(dt == 1.0)),
            "dt_percentiles": {p: float(np.percentile(dt, p)) for p in (50, 90, 99, 99.9)},
            "trips_with_gap_gt60s": g.loc[g["n_gaps_gt60s"] > 0, "trip_id"].tolist(),
            "trips_with_gap_gt1h": g.loc[g["dt_max"] > 3600, "trip_id"].tolist(),
            "nodes_per_trip_median": float(g["n_nodes"].median()),
            "arcs_per_trip_median": float(g["n_arcs"].median()),
            "frac_two_way_arcs_mean": float(g["frac_two_way"].mean()),
            "trips_with_disconnected_network": int((g["n_components"] > 1).sum()),
            "isolated_nodes_total": int(g["isolated_nodes"].sum()),
            "route_connected_frac_min": float(g["route_connected_frac"].min()),
            "route_invalid_ids_total": int(g["route_invalid_ids"].sum()),
            "frac_obs_labelled": float(np.average(g["frac_obs_labelled"], weights=g["n_obs"])),
            "gps_to_reference_m": {p: float(np.percentile(ref, p)) for p in (50, 90, 95, 99, 100)},
            "max_projection_rel_error": float(g["proj_rel_err_max"].max()),
            "n_utm_zones": int(g.apply(lambda r: utm_epsg_for((r.lon_min + r.lon_max) / 2, (r.lat_min + r.lat_max) / 2), axis=1).nunique()),
        }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    # ----------------------------------------------------------------- plots
    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
    colors = {"train": "#1f6fd1", "student_test": "#f28e2b"}

    f, ax = plt.subplots(figsize=(7, 4))
    for split, g in inv.groupby("split"):
        ax.hist(g["n_obs"], bins=30, alpha=0.7, label=f"{split} (n={len(g)})", color=colors[split])
    ax.set_xscale("log")
    ax.set_xlabel("GPS observations per trip (log scale)")
    ax.set_ylabel("trips")
    ax.set_title("Trip length distribution")
    ax.legend()
    f.tight_layout(); f.savefig(fig / "obs_per_trip.png"); plt.close(f)

    f, ax = plt.subplots(figsize=(7, 4))
    for split in all_dt:
        dt = np.concatenate(all_dt[split])
        ax.hist(np.clip(dt, 0.5, 1e5), bins=np.logspace(-0.3, 5, 60), alpha=0.7, label=split, color=colors[split])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time between consecutive observations (s, log)")
    ax.set_ylabel("count (log)")
    ax.set_title("Sampling intervals (most are exactly 1 s)")
    ax.legend()
    f.tight_layout(); f.savefig(fig / "sampling_intervals.png"); plt.close(f)

    f, ax = plt.subplots(figsize=(7, 4))
    for split in all_ref:
        ref = np.concatenate(all_ref[split])
        ax.hist(ref, bins=np.linspace(0, 40, 81), alpha=0.7, density=True, label=split, color=colors[split])
    ax.axvline(cfg["candidates"]["radius"], color="k", ls="--", lw=1)
    ax.set_xlim(0, 55)
    ax.text(cfg["candidates"]["radius"] - 1, ax.get_ylim()[1] * 0.9, "candidate radius", ha="right")
    ax.set_xlabel("distance from GPS point to its reference arc (m)")
    ax.set_ylabel("density")
    ax.set_title("GPS noise relative to the reference route")
    ax.legend()
    f.tight_layout(); f.savefig(fig / "gps_to_reference_distance.png"); plt.close(f)

    f, ax = plt.subplots(figsize=(7, 4))
    for split, g in inv.groupby("split"):
        ax.scatter(g["n_nodes"], g["n_arcs"], s=18, alpha=0.8, label=split, color=colors[split])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nodes"); ax.set_ylabel("arcs (directed)")
    ax.set_title("Road-network size per trip")
    ax.legend()
    f.tight_layout(); f.savefig(fig / "network_size.png"); plt.close(f)

    f, ax = plt.subplots(figsize=(7, 4))
    g = inv.sort_values("dt_max", ascending=False)
    ax.bar(range(len(g)), g["dt_max"], color=[colors[s] for s in g["split"]])
    ax.set_yscale("log")
    ax.set_xlabel("trips (sorted)"); ax.set_ylabel("largest time gap (s, log)")
    ax.set_title("Largest GPS gap per trip (blue = train, orange = student test)")
    f.tight_layout(); f.savefig(fig / "max_gap_per_trip.png"); plt.close(f)

    # trips are spread over several continents, so plot one marker per trip
    f, ax = plt.subplots(figsize=(10, 5))
    for split, g in inv.groupby("split"):
        ax.scatter((g["lon_min"] + g["lon_max"]) / 2, (g["lat_min"] + g["lat_max"]) / 2, s=25,
                   alpha=0.8, label=f"{split} (n={len(g)})", color=colors[split])
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Where the trips are (one marker per trip)")
    ax.legend()
    f.tight_layout(); f.savefig(fig / "trip_locations.png"); plt.close(f)

    # one trip with its road network, in metres
    split, tid = "train", inv[inv.split == "train"].sort_values("n_obs").iloc[len(inv[inv.split == "train"]) // 2]["trip_id"]
    trip = load_trip(path(cfg, split), tid)
    tr, _ = clean_track(trip.track)
    proj = Projector(float(np.median(tr[:, 0])), float(np.median(tr[:, 1])))
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.to_xy(tr[:, 0], tr[:, 1])
    f, ax = plt.subplots(figsize=(8, 8))
    m = (np.minimum(net.ax, net.bx) < px.max() + 300) & (np.maximum(net.ax, net.bx) > px.min() - 300) & \
        (np.minimum(net.ay, net.by) < py.max() + 300) & (np.maximum(net.ay, net.by) > py.min() - 300)
    for i in np.where(m)[0]:
        ax.plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="#bbbbbb", lw=0.5)
    r = trip.route
    for a in r:
        ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="#1a9e3a", lw=2.5)
    ax.scatter(px, py, s=3, color="#1f6fd1", zorder=3, label="GPS")
    ax.plot([], [], color="#1a9e3a", lw=2.5, label="reference route")
    ax.plot([], [], color="#bbbbbb", lw=1, label="road network")
    ax.set_aspect("equal"); ax.legend()
    ax.set_title(f"Trip {tid} (train) in UTM metres")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    f.tight_layout(); f.savefig(fig / f"trip_{tid}_network.png"); plt.close(f)
    print("EDA written to", out, "and", fig)


if __name__ == "__main__":
    main()
