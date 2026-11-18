"""Geometry helpers.

Longitude/latitude are angles, so distances in "degrees" are meaningless: one
degree of latitude is ~111 km everywhere, but one degree of longitude is
111 km x cos(latitude) (~72 km at 50N, ~100 km at 25N). We project every trip to
the UTM zone that contains it (metres) before doing any geometry. The trips in
this dataset are spread over several continents, so the zone is chosen per
trip from the median GPS position; inside one zone the scale error is < 0.1 %.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import Transformer


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=16)
def _transformer(epsg: int, inverse: bool = False) -> Transformer:
    if inverse:
        return Transformer.from_crs(epsg, 4326, always_xy=True)
    return Transformer.from_crs(4326, epsg, always_xy=True)


class Projector:
    """lon/lat <-> metric x/y for one trip."""

    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg_for(lon0, lat0)

    def to_xy(self, lon, lat):
        x, y = _transformer(self.epsg).transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = _transformer(self.epsg, True).transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (all arrays broadcastable).

    Returns (distance, t_clamped, t_raw, proj_x, proj_y) where t is the
    position of the projection along AB (0 at A, 1 at B).
    """
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    seg2 = np.where(seg2 < 1e-9, 1e-9, seg2)
    t_raw = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = np.clip(t_raw, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    d = np.hypot(px - qx, py - qy)
    return d, t, t_raw, qx, qy


def bearing_xy(dx, dy):
    """Compass-style bearing in degrees [0, 360) from a metric displacement."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used as a sanity check of the projection)."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
