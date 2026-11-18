"""Folium maps used in the EDA, the report and the evaluation.

Layer colours are kept the same everywhere:
    GPS observations      blue dots
    road network          light grey
    reference route       green
    classifier-only route orange (dashed)
    HMM / Viterbi route   red
"""
from __future__ import annotations

import folium
import numpy as np
from branca.element import MacroElement, Template

from .network import RoadNetwork

COLORS = {
    "gps": "#1f6fd1",
    "network": "#9a9a9a",
    "reference": "#1a9e3a",
    "classifier": "#f28e2b",
    "hmm": "#d62728",
}

_LEGEND = """
{% macro html(this, kwargs) %}
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #888; border-radius: 6px; font-size: 13px;">
  <b>{{ this.title }}</b><br>
  {% for label, color in this.items %}
    <span style="display:inline-block;width:14px;height:4px;background:{{ color }};
                 margin-right:6px;vertical-align:middle;"></span>{{ label }}<br>
  {% endfor %}
</div>
{% endmacro %}
"""


class Legend(MacroElement):
    def __init__(self, title, items):
        super().__init__()
        self._template = Template(_LEGEND)
        self.title = title
        self.items = items


def _arc_lines(net: RoadNetwork, arcs) -> list:
    return [net.arc_lonlat(int(a)) for a in arcs]


def route_polyline(net: RoadNetwork, arc_seq) -> list:
    """Collapse consecutive duplicates and return [[lat, lon], ...] following the arcs."""
    pts = []
    last = None
    for a in arc_seq:
        if a is None or a < 0 or a == last:
            continue
        seg = net.arc_lonlat(int(a))
        if pts and pts[-1] == seg[0]:
            pts.append(seg[1])
        else:
            pts.extend(seg)
        last = a
    return pts


def nearby_arcs(net: RoadNetwork, px, py, radius: float = 80.0, max_points: int = 4000) -> np.ndarray:
    step = max(1, len(px) // max_points)
    near = net.arcs_near(px[::step], py[::step], radius)
    return np.unique(np.concatenate(near)) if near else np.empty(0, dtype=int)


def trip_map(track_lonlat: np.ndarray, net: RoadNetwork, px, py, title: str,
             reference_arcs=None, classifier_arcs=None, hmm_arcs=None,
             show_network: bool = True, gps_step: int = 1) -> folium.Map:
    lat_c, lon_c = float(np.median(track_lonlat[:, 1])), float(np.median(track_lonlat[:, 0]))
    m = folium.Map(location=[lat_c, lon_c], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    legend = []

    if show_network:
        fg = folium.FeatureGroup(name="Road network (near track)", show=True)
        for line in _arc_lines(net, nearby_arcs(net, px, py)):
            folium.PolyLine(line, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))

    if reference_arcs is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(route_polyline(net, reference_arcs), color=COLORS["reference"], weight=7,
                        opacity=0.55, tooltip="reference route").add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    if classifier_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only prediction", show=False)
        for a in np.unique([x for x in classifier_arcs if x is not None and x >= 0]):
            folium.PolyLine(net.arc_lonlat(int(a)), color=COLORS["classifier"], weight=4,
                            opacity=0.9, dash_array="6,6").add_to(fg)
        fg.add_to(m)
        legend.append(("Classifier-only arcs", COLORS["classifier"]))

    if hmm_arcs is not None:
        fg = folium.FeatureGroup(name="HMM / Viterbi route", show=True)
        folium.PolyLine(route_polyline(net, hmm_arcs), color=COLORS["hmm"], weight=3.5,
                        opacity=0.95, tooltip="HMM route").add_to(fg)
        fg.add_to(m)
        legend.append(("HMM / Viterbi route", COLORS["hmm"]))

    fg = folium.FeatureGroup(name="GPS observations", show=True)
    for i in range(0, len(track_lonlat), gps_step):
        folium.CircleMarker([track_lonlat[i, 1], track_lonlat[i, 0]], radius=2, color=COLORS["gps"],
                            fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"obs {i}  t={track_lonlat[i, 2]:.0f}s" if track_lonlat.shape[1] > 2 else None
                            ).add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))

    folium.Marker([track_lonlat[0, 1], track_lonlat[0, 0]], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([track_lonlat[-1, 1], track_lonlat[-1, 0]], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[track_lonlat[:, 1].min(), track_lonlat[:, 0].min()],
                  [track_lonlat[:, 1].max(), track_lonlat[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().add_child(Legend(title, legend))
    return m
