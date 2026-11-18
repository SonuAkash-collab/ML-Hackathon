"""End-to-end prediction pipeline (used by the evaluation scripts and the API).

    raw trip (GPS + road network, NO route)
      -> clean + project                      (pathsnap.io / geo)
      -> candidate arcs + features            (pathsnap.features)
      -> classifier probabilities             (artifacts/model.joblib)
      -> HMM / Viterbi decoding               (pathsnap.hmm)
      -> matched arc per GPS point + continuous route

Scoring against the reference route is a *separate* function
(``score_prediction``) that only runs after the prediction is finished.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .features import prepare_trip
from .hmm import HMMParams, build_lattice, classifier_path, fill_route, path_to_array, viterbi
from .io import Trip
from .labels import reference_arcs
from .metrics import pointwise_accuracy, route_continuity, route_set_scores


@dataclass
class Prediction:
    trip_id: str
    ctx: dict
    feats: object
    prob: np.ndarray
    classifier_arcs: np.ndarray   # per cleaned observation, -1 = no prediction
    hmm_arcs: np.ndarray
    route_arcs: list
    info: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class PathSnapPredictor:
    def __init__(self, artifact_dir: str | Path):
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "model.joblib")
        with open(artifact_dir / "metadata.json", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.features = self.meta["features"]
        self.cand_cfg = self.meta["candidates"]
        self.spacing = self.meta.get("sample_spacing_m", 5.0)
        self.hmm_params = HMMParams(**self.meta["hmm"])
        self.lattice_cfg = self.meta["lattice"]

    def predict_proba(self, feats) -> np.ndarray:
        return self.model.predict_proba(feats[self.features].to_numpy(float))[:, 1]

    def predict_trip(self, trip: Trip) -> Prediction:
        if trip.route is not None:
            raise ValueError("predict_trip must be given a trip loaded with with_route=False")
        t0 = time.perf_counter()
        feats, ctx = prepare_trip(trip, self.cand_cfg, self.spacing)
        t1 = time.perf_counter()
        n = len(ctx["px"])
        if feats.empty:
            empty = np.full(n, -1, dtype=np.int64)
            return Prediction(trip.trip_id, ctx, feats, np.empty(0), empty, empty, [], {"n_segments": 0})
        prob = self.predict_proba(feats)
        t2 = time.perf_counter()
        steps = build_lattice(feats, prob, ctx["px"], ctx["py"], ctx["t"], ctx["net"], **self.lattice_cfg)
        path, info = viterbi(steps, self.hmm_params)
        hmm_arcs = path_to_array(path, n)
        clf_arcs = path_to_array(classifier_path(steps), n)
        route = fill_route(ctx["net"], list(hmm_arcs))
        t3 = time.perf_counter()
        timings = {"features_s": t1 - t0, "classifier_s": t2 - t1, "decoder_s": t3 - t2, "total_s": t3 - t0}
        return Prediction(trip.trip_id, ctx, feats, prob, clf_arcs, hmm_arcs, route, info, timings)


def score_prediction(pred: Prediction, ref_route: np.ndarray, label_max_dist: float = 50.0) -> dict:
    """Server-side scoring. Never call this before the prediction is complete."""
    ctx = pred.ctx
    ref_arc, _, _ = reference_arcs(ctx["px"], ctx["py"], ref_route, ctx["net"], label_max_dist)
    net = ctx["net"]
    labelled = ref_arc >= 0
    cand_hit = np.zeros(len(ref_arc), dtype=bool)
    if len(pred.feats):
        f = pred.feats
        ok = f["arc"].to_numpy() == ref_arc[f["obs"].to_numpy()]
        cand_hit[np.unique(f["obs"].to_numpy()[ok])] = True
    return {
        "n_observations": int(len(ref_arc)),
        "n_labelled": int(labelled.sum()),
        "accuracy_hmm": pointwise_accuracy(pred.hmm_arcs, ref_arc),
        "accuracy_classifier": pointwise_accuracy(pred.classifier_arcs, ref_arc),
        "candidate_miss_rate": float(1 - cand_hit[labelled].mean()) if labelled.any() else float("nan"),
        "continuity_hmm": route_continuity(pred.hmm_arcs, net),
        "continuity_classifier": route_continuity(pred.classifier_arcs, net),
        **{f"{k}_hmm": v for k, v in route_set_scores(pred.route_arcs, ref_route).items()},
        **{f"{k}_classifier": v for k, v in route_set_scores(pred.classifier_arcs, ref_route).items()},
    }
