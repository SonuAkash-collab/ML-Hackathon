"""PathSnap FastAPI backend (runs on Laptop 1).

Start from the repository root:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            model + data status
    GET  /trips             the evaluation trips available on this laptop
    POST /predict           {"trip_id": "..."} -> predicted route + server-side accuracy

What the UI receives: GPS points, the predicted route, matched points, nearby
road geometry and the accuracy numbers. The reference route itself is never
sent to the UI and is only read *after* the prediction is finished.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.config import load_config  # noqa: E402
from pathsnap.inference import PathSnapPredictor, score_prediction  # noqa: E402
from pathsnap.io import list_trips, load_route, load_trip  # noqa: E402

CFG = load_config()
API_CFG = CFG["api"]


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


EVAL_DIR = _resolve(os.environ.get("PATHSNAP_EVAL_DIR", API_CFG["eval_dir"]))
ARTIFACTS = _resolve(os.environ.get("PATHSNAP_ARTIFACTS", API_CFG["artifacts"]))
LOG_FILE = _resolve(API_CFG["log_file"])
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")])
log = logging.getLogger("pathsnap.api")

predictor: PathSnapPredictor | None = None


@asynccontextmanager
async def lifespan(_app):
    global predictor
    predictor = PathSnapPredictor(ARTIFACTS)
    m = predictor.meta
    log.info("model loaded: %s (%s) from %s", m["model_name"], m.get("model_variant"), ARTIFACTS)
    log.info("HMM params: %s", m["hmm"])
    log.info("evaluation trips folder: %s (%d trips)", EVAL_DIR, len(list_trips(EVAL_DIR)))
    yield


app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan,
              description="Map-matching of noisy GPS tracks with a classifier + HMM/Viterbi decoder")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _f(x):
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.meta["model_name"] if predictor else None,
            "model_variant": predictor.meta.get("model_variant") if predictor else None,
            "eval_dir": str(EVAL_DIR), "n_trips": len(list_trips(EVAL_DIR))}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(EVAL_DIR):
        track = EVAL_DIR / tid / f"{tid}.track"
        with open(track, encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
        out.append({"trip_id": tid, "n_observations": n})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in list_trips(EVAL_DIR):       # also blocks path traversal
        raise HTTPException(status_code=404, detail=f"unknown trip_id {tid!r}")
    t0 = time.perf_counter()
    log.info("POST /predict trip=%s : running pipeline", tid)

    # 1) prediction: the trip is loaded WITHOUT its route
    trip = load_trip(EVAL_DIR, tid, with_route=False)
    res = predictor.predict_trip(trip)
    ctx, net = res.ctx, res.ctx["net"]
    t_pred = time.perf_counter() - t0

    # 2) scoring: the reference route is read only now, server-side
    metrics = None
    route_file = EVAL_DIR / tid / f"{tid}.route"
    if route_file.exists():
        metrics = score_prediction(res, load_route(EVAL_DIR, tid), predictor.meta["label_max_dist"])
        metrics = {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}

    # 3) payload for the UI
    track = ctx["track"]
    step = max(1, int(np.ceil(len(track) / API_CFG["max_gps_points_in_response"])))
    gps = [[_f(track[i, 1]), _f(track[i, 0]), _f(track[i, 2])] for i in range(0, len(track), step)]

    route_coords = []
    for a in res.route_arcs:
        seg = net.arc_lonlat(int(a))
        if route_coords and route_coords[-1] == seg[0]:
            route_coords.append(seg[1])
        else:
            route_coords.extend(seg)

    matched = []
    if len(res.feats):
        f = res.feats
        chosen = f[f["arc"].to_numpy() == res.hmm_arcs[f["obs"].to_numpy()]]
        chosen = chosen.drop_duplicates("obs").iloc[::step]
        lon, lat = ctx["projector"].to_lonlat(chosen["qx"].to_numpy(), chosen["qy"].to_numpy())
        matched = [[_f(a), _f(b)] for a, b in zip(lat, lon)]
        near = np.unique(f["arc"].to_numpy())
        roads = [net.arc_lonlat(int(a)) for a in near]
    else:
        roads = []

    total = time.perf_counter() - t0
    acc = metrics["accuracy_hmm"] if metrics else None
    log.info("POST /predict trip=%s : %d obs, %d route arcs, HMM acc=%s, classifier acc=%s, "
             "pipeline %.2fs, total %.2fs", tid, len(track), len(res.route_arcs), acc,
             metrics["accuracy_classifier"] if metrics else None, t_pred, total)
    return {
        "trip_id": tid,
        "n_observations": int(len(track)),
        "n_predicted": int((res.hmm_arcs >= 0).sum()),
        "gps": gps,
        "predicted_route": route_coords,
        "matched_points": matched,
        "candidate_roads": roads,
        "predicted_arc_ids": [int(a) for a in res.route_arcs],
        "metrics": metrics,
        "decoder": res.info,
        "timings_s": {k: round(v, 3) for k, v in {**res.timings, "total_s": total}.items()},
        "model": {"name": predictor.meta["model_name"], "variant": predictor.meta.get("model_variant")},
    }
