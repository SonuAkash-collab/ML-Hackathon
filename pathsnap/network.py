"""Road network built from a trip's .nodes and .arcs files.

* arcs are directed: arc i goes from node arcs[i,0] to node arcs[i,1]
* each arc is treated as a straight segment between its two nodes
* a KD-tree over points sampled every few metres along every arc gives fast
  "which arcs are within R metres of this GPS point" queries
* a bounded Dijkstra over the directed graph gives network distances that the
  HMM transition model needs
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from .geo import Projector, bearing_xy, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, projector: Projector,
                 sample_spacing: float = 5.0):
        self.projector = projector
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        x, y = projector.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.node_xy = np.column_stack([x, y])
        self.nodes_lonlat = nodes_lonlat
        self.arc_from = arcs[:, 0].astype(np.int64)
        self.arc_to = arcs[:, 1].astype(np.int64)
        self.ax, self.ay = self.node_xy[self.arc_from, 0], self.node_xy[self.arc_from, 1]
        self.bx, self.by = self.node_xy[self.arc_to, 0], self.node_xy[self.arc_to, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_xy(self.bx - self.ax, self.by - self.ay)

        # adjacency: outgoing arcs per node in CSR form
        order = np.argsort(self.arc_from, kind="stable")
        self._out_arcs = order
        counts = np.bincount(self.arc_from, minlength=self.n_nodes)
        self._out_ptr = np.r_[0, np.cumsum(counts)]
        self.out_degree = counts
        self.in_degree = np.bincount(self.arc_to, minlength=self.n_nodes)

        # reverse-arc lookup (two-way streets appear as two arcs)
        key = self.arc_from * (self.n_nodes + 1) + self.arc_to
        rkey = self.arc_to * (self.n_nodes + 1) + self.arc_from
        self.has_reverse = np.isin(rkey, key)

        self.sample_spacing = float(sample_spacing)
        self._build_index()
        self._sp_cache: dict[int, tuple[float, dict]] = {}

    # ------------------------------------------------------------------ index
    def _build_index(self):
        s = self.sample_spacing
        n_samp = np.maximum(1, np.ceil(self.length / s).astype(int)) + 1
        arc_ids = np.repeat(np.arange(self.n_arcs), n_samp)
        # position of each sample along its arc, 0..1
        starts = np.r_[0, np.cumsum(n_samp)[:-1]]
        local = np.arange(len(arc_ids)) - np.repeat(starts, n_samp)
        frac = local / np.repeat(n_samp - 1, n_samp)
        sx = self.ax[arc_ids] + frac * (self.bx - self.ax)[arc_ids]
        sy = self.ay[arc_ids] + frac * (self.by - self.ay)[arc_ids]
        self._sample_arc = arc_ids
        self._tree = cKDTree(np.column_stack([sx, sy]))

    def arcs_near(self, px: np.ndarray, py: np.ndarray, radius: float) -> list[np.ndarray]:
        """For every point, the unique arc ids with *some* sample within radius + spacing/2."""
        pts = np.column_stack([px, py])
        hits = self._tree.query_ball_point(pts, r=radius + self.sample_spacing / 2.0)
        return [np.unique(self._sample_arc[h]) if len(h) else np.empty(0, dtype=np.int64) for h in hits]

    def project(self, px, py, arc_ids):
        """Exact point-to-segment projection for (point, arc) pairs."""
        return point_segment(px, py, self.ax[arc_ids], self.ay[arc_ids], self.bx[arc_ids], self.by[arc_ids])

    # ------------------------------------------------------------- topology
    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    def node_distances(self, src: int, cutoff: float) -> dict:
        """Shortest directed distances from ``src`` to every node within ``cutoff`` metres.

        Results are cached; a cached run with a larger cutoff is reused.
        """
        cached = self._sp_cache.get(src)
        if cached is not None and cached[0] >= cutoff:
            return cached[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        done = set()
        ptr, out, to, length = self._out_ptr, self._out_arcs, self.arc_to, self.length
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                a = out[k]
                v = int(to[a])
                nd = d + length[a]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._sp_cache[src] = (cutoff, dist)
        if len(self._sp_cache) > 200_000:   # keep memory bounded on very long trips
            self._sp_cache.clear()
        return dist

    def route_distance(self, a: int, pos_a: float, b: int, pos_b: float, cutoff: float,
                       backward_tol: float = 10.0) -> float:
        """Driving distance from position ``pos_a`` (m) on arc ``a`` to ``pos_b`` on arc ``b``.

        Returns ``inf`` when ``b`` cannot be reached within ``cutoff`` metres.
        Small backward moves on the same arc (GPS jitter) are allowed up to
        ``backward_tol`` metres and counted as zero movement.
        """
        if a == b:
            d = pos_b - pos_a
            if d >= -backward_tol:
                return max(d, 0.0)
        rest = self.length[a] - pos_a
        budget = cutoff - rest - pos_b
        if budget < 0:
            return np.inf
        nd = self.node_distances(int(self.arc_to[a]), budget)
        mid = nd.get(int(self.arc_from[b]))
        if mid is None:
            return np.inf
        return rest + mid + pos_b

    def shortest_arc_path(self, a: int, b: int, cutoff: float = 3000.0) -> list[int] | None:
        """Arcs strictly between arc a and arc b on the shortest directed path (None if unreachable)."""
        src, dst = int(self.arc_to[a]), int(self.arc_from[b])
        if src == dst:
            return []
        dist = {src: 0.0}
        pred: dict[int, int] = {}
        heap = [(0.0, src)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            if u == dst:
                break
            done.add(u)
            for arc in self.out_arcs(u):
                v = int(self.arc_to[arc])
                nd = d + self.length[arc]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = int(arc)
                    heapq.heappush(heap, (nd, v))
        if dst not in pred:
            return None
        path, v = [], dst
        while v != src:
            arc = pred[v]
            path.append(arc)
            v = int(self.arc_from[arc])
        return path[::-1]

    def is_reverse(self, a: int, b: int) -> bool:
        return self.arc_from[a] == self.arc_to[b] and self.arc_to[a] == self.arc_from[b]

    def connected_within(self, a: int, b: int, max_hops: int = 3) -> bool:
        """True if arc b can follow arc a using at most ``max_hops`` arcs after a."""
        if a == b:
            return True
        target = int(self.arc_from[b])
        start = int(self.arc_to[a])
        if start == target:
            return True
        seen = {start}
        q = deque([(start, 1)])
        while q:
            u, h = q.popleft()
            if h >= max_hops:
                continue
            for arc in self.out_arcs(u):
                v = int(self.arc_to[arc])
                if v == target:
                    return True
                if v not in seen:
                    seen.add(v)
                    q.append((v, h + 1))
        return False

    def connected_components(self) -> tuple[int, np.ndarray]:
        """Weakly connected components (for EDA)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(self.n_arcs), (self.arc_from, self.arc_to)), shape=(self.n_nodes, self.n_nodes))
        return connected_components(g, directed=True, connection="weak")

    def arc_lonlat(self, arc_id: int) -> list[list[float]]:
        """[[lat, lon], [lat, lon]] of an arc, handy for folium / leaflet."""
        f, t = self.arc_from[arc_id], self.arc_to[arc_id]
        return [[float(self.nodes_lonlat[f, 1]), float(self.nodes_lonlat[f, 0])],
                [float(self.nodes_lonlat[t, 1]), float(self.nodes_lonlat[t, 0])]]
