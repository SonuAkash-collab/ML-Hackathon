# PathSnap: map-matching noisy GPS tracks

PathSnap matches every GPS observation of a trip to a road segment (arc) of the trip's road network, and rebuilds the route the vehicle drove. A LightGBM classifier scores the nearby candidate roads for each GPS point. An HMM/Viterbi decoder then picks one connected route through those candidates.

**Course:** UE24CS352A Machine Learning, Team PT-059

## Contents

- [Results](#results)
- [Team](#team)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Reproduce the experiments](#reproduce-the-experiments)
- [Usage: the API](#usage-the-api)
- [Deployment on evaluation day](#deployment-on-evaluation-day)
- [Troubleshooting](#troubleshooting)
- [Data-leakage safeguards](#data-leakage-safeguards)

## Results

Share of GPS observations matched to the correct road on the 10 student-test trips:

| Method | Top-1 accuracy |
|---|---|
| Nearest road (baseline) | 54.49% |
| Best single model: LightGBM classifier | 90.42% |
| **Final: tuned LightGBM + HMM/Viterbi** | **92.38%** |

All 8 classifiers, sorted by student-test top-1. These are classifier-only scores, before the HMM:

| Model | Trained by | Student-test top-1 | CV macro F1 (5-fold) | Log loss (student test) |
|---|---|---|---|---|
| **LightGBM** | Purushothama | **90.42%** | 0.9349 | **0.0849** |
| HistGradientBoosting | Purushothama | 89.57% | 0.9331 | 0.0914 |
| Random Forest | Ayush | 89.42% | 0.9352 | 0.0879 |
| Decision Tree | Sonu Akash | 89.32% | 0.9300 | 0.1085 |
| MLP | Sahana | 88.39% | 0.9308 | 0.0962 |
| Logistic Regression | Sonu Akash | 88.33% | 0.9282 | 0.1151 |
| KNN | Ayush | 88.08% | 0.9281 | 0.1502 |
| Gaussian Naive Bayes | Sahana | 87.52% | 0.9056 | 0.5530 |

- Full table: [`results/models/comparison.csv`](results/models/comparison.csv)
- Each member's own rows: [`results/team/`](results/team/)
- Method, plots and discussion: [`report/PathSnap_Report.pdf`](report/PathSnap_Report.pdf)

## Team

| # | SRN | Name | GitHub | Owns | Models trained |
|---|---|---|---|---|---|
| 1 | PES1UG24AM908 | Sonu Akash Soundararajan | SonuAkash-collab | data loading, cleaning, EDA, candidates, features | Logistic Regression, Decision Tree |
| 2 | PES1UG24AM909 | Ayush P Mankal | SpiderByte07 | Optuna tuning, HMM/Viterbi decoder, exported model | Random Forest, KNN |
| 3 | PES1UG25AM810 | Purushothama K C | Purushothama449 | shared trainer, grouped CV, metrics, model comparison | LightGBM, HistGradientBoosting |
| 4 | PES1UG25AM811 | Sahana Kadappa | sahanagolabanvi34-debug | FastAPI backend, UI, deployment, report | Gaussian Naive Bayes, MLP |

[`TEAM_SPLIT.md`](TEAM_SPLIT.md) lists what each person should be able to explain in the viva.

## Quick start

This runs the demo with the committed model in `artifacts/`. No training is needed.

**Windows (PowerShell)**

```powershell
git clone https://github.com/SonuAkash-collab/ML-Hackathon.git
cd ML-Hackathon
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# copy the trip folders into data\ (see "Data"), then start the API:
$env:PATHSNAP_EVAL_DIR = "data/student_test"   # optional: serve the student-test trips instead of data/evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```powershell
.venv\Scripts\Activate.ps1
python ui/serve.py --api http://localhost:8000
```

**macOS / Linux**

```bash
git clone https://github.com/SonuAkash-collab/ML-Hackathon.git
cd ML-Hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# copy the trip folders into data/ (see "Data"), then start the API:
export PATHSNAP_EVAL_DIR=data/student_test     # optional: serve the student-test trips instead of data/evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
source .venv/bin/activate
python ui/serve.py --api http://localhost:8000
```

The UI opens at `http://localhost:5500`. Pick a trip to see the predicted route and its accuracy.

## How it works

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

**Preprocessing** (`pathsnap/io.py`, `geo.py`, `labels.py`)
- `clean_track` cleans each GPS track.
- Each trip is projected to its own UTM zone, so distances are in metres. The trips cover 25 UTM zones.
- Training labels come from the `.route` file. A monotone alignment assigns each GPS point a reference arc. Points more than 50 m from the route stay unlabelled.

**Candidates** (`pathsnap/candidates.py`, `network.py`)
- A KD-tree holds points sampled every 5 m along each arc.
- Each GPS point gets up to 8 candidate arcs within 50 m. If none is that close, the search widens to 4 arcs within 150 m.
- Candidate miss rate on the student-test trips is 0.28%. That is the share of labelled points whose true arc isn't among the candidates.

**Features** (`pathsnap/features.py`): 23 features per (point, candidate) pair, in 5 groups:

| Group | Features |
|---|---|
| Geometry | `dist`, `dist_rank`, `dist_minus_min`, `dist_ratio`, `n_cands`, `t`, `outside` |
| Heading | `head_diff`, `head_diff_wide`, `cos_head_wide`, `head_rank` |
| Context (neighbour points) | `win_dist_mean`, `win_dist_max`, `win_rank` |
| Kinematics | `speed`, `dt_prev`, `dt_next`, `move_wide` |
| Topology | `arc_length`, `out_deg_end`, `in_deg_start`, `has_reverse`, `expanded` |

**Models and selection** (`pathsnap/models.py`, `scripts/04_train_compare.py`)
- 8 classifiers predict P(candidate is the true arc). The top-scoring candidate is the prediction for each point.
- All 8 use 5-fold GroupKFold by trip on the 80 training trips. Training uses every 3rd labelled observation.
- LightGBM was selected. It has the best CV top-1 (0.8945), the best student-test top-1 (90.42%) and the lowest log loss (0.0849). Low log loss matters because the probabilities become the HMM emissions.

**Optuna tuning** (`scripts/06_bayes_opt.py`, `configs/optuna.yaml`)
- 30 trials with the TPE sampler (seed 42) and a median pruner: 20 completed, 10 pruned.
- Objective: mean 3-fold group-CV top-1 on the training trips.
- Best trial: 29, with CV top-1 0.8942.
- Best parameters (`results/optuna/best_params.json`): learning_rate 0.0343, num_leaves 81, min_child_samples 57, subsample 0.835, colsample_bytree 0.836, reg_lambda 0.059, reg_alpha 2.95, n_estimators 1500.

**HMM / Viterbi decoder** (`pathsnap/hmm.py`, `scripts/07_tune_decoder.py`)
- **Hidden states:** the candidate arcs of each GPS point.
- **Emissions:** `w · log(max(p, floor))`, where `p` is LightGBM's probability normalised over the point's candidates.
- **Transitions:** `−|network route distance − straight-line distance| / β`. A U-turn penalty is subtracted when switching to the reverse arc.
- **Decoding:** Viterbi finds the best path. The chain is cut and restarted when the time gap exceeds `gap_seconds`.
- **Tuned settings** (`artifacts/metadata.json`): β = 10 m, emission weight w = 0.5, U-turn penalty 8, gap 120 s, probability floor 0.0001. They were tuned on out-of-fold probabilities of training trips only.
- **Effect on the student-test trips** (`results/hmm/student_test_summary.json`), tuned LightGBM alone vs with the HMM:

| | Tuned LightGBM only | With HMM |
|---|---|---|
| Accuracy | 90.41% | 92.38% |
| Route continuity | 93.48% | 98.86% |
| Route F1 | 0.9018 | 0.9483 |

The HMM improved 9 of the 10 trips; 1 got worse.

## Repository layout

```
configs/      data.yaml · models.yaml · optuna.yaml · hmm.yaml · api.yaml
pathsnap/     the Python package (io, geo, network, labels, candidates, features,
              viz, metrics, models, hmm, inference)
scripts/      01_eda … 09_error_analysis, 10_feature_analysis, run_all.py
api/          FastAPI backend
ui/           evaluator UI (static HTML + Leaflet) and a tiny server
artifacts/    final tuned LightGBM (model.joblib) + HMM settings (metadata.json); created by steps 6–7
results/      CSV / JSON outputs of every experiment (see results/README.md)
  team/                    each member's two model results (results/team/<member>/results.csv)
  models/comparison.csv    combined 8-model comparison table
  eda/                     dataset inventory and summary (step 1)
  candidates/              candidate statistics (step 2)
  optuna/                  Optuna study, trials and best parameters (step 6)
  hmm/                     decoder tuning, HMM evaluation and error analysis (steps 7–9)
  features/                feature importance and group ablation (step 10)
  run_logs/                console logs of steps 04–10
figures/      PNG plots and folium HTML maps (eda/, models/, optuna/, hmm/, features/, maps/)
report/       final report: REPORT.md (source) + PathSnap_Report.pdf
```

## Setup

Python 3.10–3.13.

```bash
git clone https://github.com/SonuAkash-collab/ML-Hackathon.git
cd ML-Hackathon
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The raw data is not committed. Copy the folders from the provided `Student folder` into `data/`:

```
data/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
  student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips
  evaluator_test/<trip_id>/...                                 10 trips, given on evaluation day
```

The evaluator-testing trips are not used anywhere in training, tuning or model selection. The API only reads them when a trip is requested.

## Reproduce the experiments

Run these on Laptop 1. The commands are the same in PowerShell and bash once the venv is active.

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

## Usage: the API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | status, model name and variant, evaluation folder, trip count |
| GET | `/trips` | | `{"trips": [{trip_id, n_observations}]}` |
| POST | `/predict` | `{"trip_id": "00000007"}` | GPS, predicted route, matched points, metrics, timings |

Interactive docs are at `http://<laptop-1-ip>:8000/docs`.

**Example request**

PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body '{"trip_id":"00000052"}'
```

curl (macOS/Linux; on Windows type `curl.exe`, because `curl` in PowerShell 5.1 is an alias for `Invoke-WebRequest`):

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"trip_id":"00000052"}'
```

The trip must be in the folder the API serves: `data/evaluator_test/`, or the folder set in `PATHSNAP_EVAL_DIR`. An unknown `trip_id` returns 404.

**Response fields** (from `api/main.py`)

| Field | Meaning |
|---|---|
| `trip_id`, `n_observations`, `n_predicted` | the trip, its number of GPS points, and how many got a predicted arc |
| `gps` | GPS points (thinned to at most 4,000 for the UI) |
| `predicted_route` | coordinates of the decoded route |
| `predicted_arc_ids` | arc ids of the decoded route |
| `matched_points` | each GPS point snapped onto its matched arc |
| `candidate_roads` | the candidate arcs near the track, for drawing |
| `metrics` | `accuracy_hmm`, `accuracy_classifier`, route continuity, route precision / recall / F1, `candidate_miss_rate`. These are computed server-side after prediction, and are `null` if the trip has no `.route` file. |
| `decoder` | `n_segments`, `n_breaks_gap`, `n_breaks_disconnected` |
| `timings_s` | `features_s`, `classifier_s`, `decoder_s`, `total_s` |
| `model` | model `name` and `variant` |

## Deployment on evaluation day

The setup uses two laptops on the same network, for example the same phone hotspot.

**Laptop 1: API**

1. Put the 10 evaluator trips in `data/evaluator_test/`.
2. Make sure `artifacts/model.joblib` and `artifacts/metadata.json` exist (committed, or run steps 6–7).
3. Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

4. Find the laptop's IP address (`ipconfig` on Windows, `ip addr` / `ifconfig` elsewhere) and allow port 8000 through the firewall (see [Troubleshooting](#troubleshooting)).

Every request is logged to the console and to `logs/api.log`, for example:

```
2026-.. | INFO | POST /predict trip=00000007 : running pipeline
2026-.. | INFO | POST /predict trip=00000007 : 1354 obs, 236 route arcs, HMM acc=0.97, classifier acc=0.94, pipeline 1.1s, total 1.2s
INFO:     192.168.1.23:52144 - "POST /predict HTTP/1.1" 200 OK
```

To test with the student-test trips instead, set `PATHSNAP_EVAL_DIR` before starting the API:

```powershell
# Windows PowerShell
$env:PATHSNAP_EVAL_DIR = "data/student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
# macOS / Linux
PATHSNAP_EVAL_DIR=data/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In PowerShell, the variable stays set for that terminal. Run `Remove-Item Env:PATHSNAP_EVAL_DIR` to go back to `data/evaluator_test/`.

**Laptop 2: UI**

```bash
python ui/serve.py --api http://<laptop-1-ip>:8000
```

This opens `http://localhost:5500`. The UI lists the trips the API has, sends only the selected `trip_id` to `POST /predict`, and draws the GPS points, the predicted route, the matched points and the candidate roads, with the accuracy the API computed. Map tiles need internet access. Without it the map is blank, but the routes still draw.

## Troubleshooting

| Problem | Fix |
|---|---|
| Laptop 2 can't reach the API | Use Laptop 1's **Wi-Fi IPv4 address** from `ipconfig`, not a VirtualBox / host-only adapter address like `192.168.56.x`. |
| Firewall blocks port 8000 | On Laptop 1, in PowerShell **as Administrator**: `New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow` |
| `port 8000 already in use` | Another uvicorn is still running. Stop it (PowerShell: `Get-NetTCPConnection -LocalPort 8000 -State Listen` shows its `OwningProcess`, then `Stop-Process -Id <pid>`; macOS/Linux: `lsof -i :8000`, then `kill <pid>`). Or start on another port with `--port 8001`, and use `--api http://<ip>:8001` for the UI. |
| PowerShell: "running scripts is disabled on this system" when activating the venv | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again. This only affects the current terminal. |

**curl could not connect: checklist**

1. Is the API running? Its console should show `Uvicorn running on http://0.0.0.0:8000`.
2. Was it started with `--host 0.0.0.0`? Without it, only Laptop 1 itself can connect.
3. Does it work on Laptop 1 itself? `curl.exe http://localhost:8000/health`
4. Are both laptops on the same network (same hotspot), and are you using the Wi-Fi IPv4 address?
5. Is the firewall rule for port 8000 in place?
6. On Windows, did you type `curl.exe`, not `curl`?

## Data-leakage safeguards

* `load_trip(..., with_route=False)` is the only loader used for prediction, and `PathSnapPredictor.predict_trip` refuses a trip that has a route attached.
* Features use only the GPS track and the road network (`pathsnap/features.py`).
* `score_prediction` reads the reference route only after the prediction is finished. Only the accuracy numbers go to the UI, never the reference route.
* Model selection, Bayesian search and decoder tuning use the training trips (group-aware CV). The student-test trips are used only for development-stage comparison. The evaluator trips are used only by the API.
* No predictions are hard-coded. The API runs the full pipeline on every request.
