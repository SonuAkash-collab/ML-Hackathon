# PathSnap: map-matching noisy GPS tracks

UE24CS352A Machine Learning hackathon. PathSnap matches every GPS observation of a trip to a road segment (arc) of the trip's road network, and rebuilds the route the vehicle drove.

```
raw trip (.track GPS + .nodes/.arcs road network)
   │  clean, project to UTM metres                      pathsnap/io.py, geo.py
   ▼
candidate arcs within 50 m of each GPS point (≤ 8)     pathsnap/candidates.py, network.py
   ▼
23 features per (point, candidate)                     pathsnap/features.py
   ▼
LightGBM classifier → P(candidate is the true arc)     pathsnap/models.py  (8 models compared)
   ▼                                                    tuned with Optuna (Bayesian, TPE)
HMM / Viterbi over candidates + road connectivity      pathsnap/hmm.py
   ▼
matched arc per GPS point + continuous route           pathsnap/inference.py
   ▼
FastAPI (Laptop 1)  ⇄  browser UI (Laptop 2)            api/, ui/
```

## Results

Share of GPS observations matched to the correct road on the 10 student-test trips:

| Method | Top-1 accuracy |
|---|---|
| Nearest road (baseline) | 54.49% |
| Best single model: LightGBM classifier | 90.42% |
| **Final: tuned LightGBM + HMM/Viterbi** | **92.38%** |

The full 8-model comparison is in `results/models/comparison.csv`. The method and all results are in `report/PathSnap_Report.pdf`.

## Team

| Member | GitHub | Owns |
|---|---|---|
| Sonu Akash | SonuAkash-collab | data loading, EDA, candidate generation, feature engineering, EDA maps |
| Purushotham | Purushothama449 | model zoo, metrics, 5-fold group CV comparison of 8 models, comparison plots |
| Ayush | SpiderByte07 | Bayesian optimisation, HMM/Viterbi decoder, decoder tuning, route evaluation and maps |
| Sahana | sahanagolabanvi34-debug | FastAPI backend, evaluator UI, deployment, run scripts, report assembly |

See `TEAM_SPLIT.md` for what each person should be able to explain in the viva.

## Repository layout

```
configs/      data.yaml · models.yaml · optuna.yaml · hmm.yaml · api.yaml
pathsnap/     the Python package (io, geo, network, labels, candidates, features,
              viz, metrics, models, hmm, inference)
scripts/      01_eda … 09_error_analysis, 10_feature_analysis, run_all.py
api/          FastAPI backend
ui/           evaluator UI (static HTML + Leaflet) and a tiny server
artifacts/    final tuned LightGBM (model.joblib) + HMM settings (metadata.json); created by steps 6–7
results/      CSV / JSON outputs of every experiment
  team/                    each member's two model results (results/team/<member>/results.csv)
  models/comparison.csv    combined 8-model comparison table
figures/      PNG plots and folium HTML maps
report/       final report: REPORT.md (source) + PathSnap_Report.pdf
```

## 1. Setup (both laptops)

Python 3.10–3.13.

```bash
git clone https://github.com/SonuAkash-collab/ml-hack.git
cd ml-hack
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

The raw data is not committed. Copy the folders from the provided `Student folder` into `data/`:

```
data/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
  student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips
  evaluator_test/<trip_id>/...                                 10 trips, given on evaluation day
```

The evaluator-testing trips are not used anywhere in training, tuning or model selection. The API only reads them when a trip is requested.

## 3. Reproduce the experiments (Laptop 1)

```bash
python scripts/run_all.py          # steps 1–9 (01_eda … 09_error_analysis), about 30–40 min on a 2-core laptop
python scripts/10_feature_analysis.py   # feature analysis, run separately (not part of run_all.py)
```

or step by step:

| Step | Command | Output |
|---|---|---|
| EDA | `python scripts/01_eda.py` | `results/eda/`, `figures/eda/` |
| Candidates + features | `python scripts/02_build_features.py` | `cache/*.parquet`, `results/candidates/` |
| EDA folium maps | `python scripts/03_eda_maps.py` | `figures/maps/eda_*.html` |
| Model comparison | `python scripts/04_train_compare.py` | `results/models/` |
| Comparison plots | `python scripts/05_plot_comparison.py` | `figures/models/`, `results/models/comparison_table.md` |
| Bayesian optimisation | `python scripts/06_bayes_opt.py` | `results/optuna/`, `figures/optuna/`, `artifacts/model.joblib` |
| Decoder tuning | `python scripts/07_tune_decoder.py` | `results/hmm/decoder_*`, `artifacts/metadata.json` |
| HMM evaluation | `python scripts/08_eval_hmm.py` | `results/hmm/student_test_*`, `figures/hmm/`, `figures/maps/student_test_*.html` |
| Error analysis | `python scripts/09_error_analysis.py` | `results/hmm/error_breakdown*.csv`, `reverse_error_headings.csv` |
| Feature analysis (not in `run_all.py`) | `python scripts/10_feature_analysis.py` | `results/features/`, `figures/features/` |

`04_train_compare.py` takes model names (`logreg gnb knn tree rf hgb lgbm mlp`; default: all 8) and `--no-cv`. With `--no-cv` it skips cross-validation and only refits and scores the student-test trips. The `cv_*` columns are then NaN, so don't use them in the report.

Random seeds are fixed (42). The Optuna study is stored in `results/optuna/study.db`, so re-running continues the same study.

## 4. Deployment on evaluation day

**Laptop 1: API**

1. Put the 10 evaluator trips in `data/evaluator_test/`.
2. Make sure `artifacts/model.joblib` and `artifacts/metadata.json` exist (committed, or run steps 6–7).
3. Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

4. Find the laptop's IP address (`ipconfig` on Windows, `ip addr` / `ifconfig` elsewhere) and allow port 8000 through the firewall.

Every request is logged to the console and to `logs/api.log`, for example:

```
2026-.. | INFO | POST /predict trip=00000007 : running pipeline
2026-.. | INFO | POST /predict trip=00000007 : 1354 obs, 236 route arcs, HMM acc=0.97, classifier acc=0.94, pipeline 1.1s, total 1.2s
INFO:     192.168.1.23:52144 - "POST /predict HTTP/1.1" 200 OK
```

To test with the student-test trips, run `PATHSNAP_EVAL_DIR=data/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000` (Windows PowerShell: `$env:PATHSNAP_EVAL_DIR="data/student_test"` first).

**Laptop 2: UI**

```bash
python ui/serve.py --api http://<laptop-1-ip>:8000
```

This opens `http://localhost:5500`. The UI lists the trips the API has, sends only the selected `trip_id` to `POST /predict`, and draws the GPS points, the predicted route, the matched points and the candidate roads, with the accuracy the API computed. Map tiles need internet access. Without it the map is blank, but the routes still draw.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | model name, evaluation folder, trip count |
| GET | `/trips` | | `[{trip_id, n_observations}]` |
| POST | `/predict` | `{"trip_id": "00000007"}` | GPS, predicted route, matched points, metrics, timings |

Interactive docs are at `http://<laptop-1-ip>:8000/docs`.

## 5. Data-leakage safeguards

* `load_trip(..., with_route=False)` is the only loader used for prediction, and `PathSnapPredictor.predict_trip` refuses a trip that has a route attached.
* Features use only the GPS track and the road network (`pathsnap/features.py`).
* `score_prediction` reads the reference route only after the prediction is finished. Only the accuracy numbers go to the UI, never the reference route.
* Model selection, Bayesian search and decoder tuning use the training trips (group-aware CV). The student-test trips are used only for development-stage comparison. The evaluator trips are used only by the API.
* No predictions are hard-coded. The API runs the full pipeline on every request.
