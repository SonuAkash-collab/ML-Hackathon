"""Interactive folium maps for the EDA section.

Run:  python scripts/03_eda_maps.py [trip_id ...]

By default draws one short and one long training trip plus one student-test
trip: GPS points, the road network near the track and the reference route.
Maps are written to figures/maps/eda_<split>_<trip>.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import load_config, path  # noqa: E402
from pathsnap.features import prepare_trip  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402


def draw(cfg, split, tid, out):
    trip = load_trip(path(cfg, split), tid)
    _, ctx = prepare_trip(trip, cfg["candidates"])
    m = trip_map(ctx["track"], ctx["net"], ctx["px"], ctx["py"],
                 title=f"{split} trip {tid}: raw GPS vs reference route",
                 reference_arcs=trip.route, gps_step=max(1, len(ctx["px"]) // 2000))
    f = out / f"eda_{split}_{tid}.html"
    m.save(str(f))
    print("wrote", f)


def main():
    cfg = load_config()
    out = path(cfg, "figures") / "maps"
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        for tid in sys.argv[1:]:
            split = "train" if tid in list_trips(path(cfg, "train")) else "student_test"
            draw(cfg, split, tid, out)
        return
    draw(cfg, "train", "00000001", out)     # short urban trip
    draw(cfg, "train", "00000054", out)     # very long trip where the route covers only part of the GPS
    draw(cfg, "student_test", list_trips(path(cfg, "student_test"))[0], out)


if __name__ == "__main__":
    main()
