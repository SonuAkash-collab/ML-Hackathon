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

## Team

| Member | GitHub | Owns |
|---|---|---|
| Sonu Akash | SonuAkash-collab | data loading, EDA, candidate generation, feature engineering, EDA maps |
| Purushotham | Purushothama449 | model zoo, metrics, 5-fold group CV comparison of 8 models, comparison plots |
| Ayush | SpiderByte07 | Bayesian optimisation, HMM/Viterbi decoder, decoder tuning, route evaluation and maps |
| Sahana | sahanagolabanvi34-debug | FastAPI backend, evaluator UI, deployment, run scripts, report assembly |

See `TEAM_SPLIT.md` for what each person should be able to explain in the viva.

## Where things are (local working copy)

The team's local working folder is `pathsnap_review/`. Open this folder in VS Code. Its `.venv/` is already set up here, so don't rename or move the folder. The sibling `ml-hack/` folder is the GitHub clone. Leave it as it is.

| What | Path |
|---|---|
| Source code | `pathsnap/` (package), `scripts/` (01–10, `run_all.py`), `api/`, `ui/`, `configs/` |
| Raw data (git-ignored) | `data/train/` (80 trips), `data/student_test/` (10 trips) |
| Prepared features (git-ignored) | `cache/features_train.parquet`, `cache/features_student_test.parquet` |
| Saved student-test probabilities | `cache/probs_student_test.parquet` (from the local LR + tree runs) |
| Local results and logs | `results/models/`, `results/candidates/`, `logs/` |
| Reference run (8 models, Optuna, HMM, saved model) | `reference_outputs/` (unchanged layout; see its `README.md`) |
| Review 1 archive | `archive/review1/` (summary PDF, workbook, top-5 plots, feature analysis, hand-off guides) |
| Report | `report/REPORT.md` (source), `report/PathSnap_Report_DRAFT_1.pdf` (full draft) |

## After Review 1

Review 1 is finished. Its material is in `archive/review1/`. Start there with `REVIEW1_SUMMARY.pdf`.

**Model runs with local evidence (run on this laptop, `--no-cv`):**

| Model | Log | Student-test top-1 | Reference run |
|---|---|---|---|
| Logistic Regression | `logs/04_logreg_nocv.log` | 0.8833 | 0.8833 |
| Decision Tree | `logs/04_tree_nocv.log` | 0.8932 | 0.8930 |

The results are in `results/models/comparison.csv` and `details.json`. The student-test probabilities are in `cache/probs_student_test.parquet`. These local runs skipped cross-validation, so they have no `cv_*` numbers and no `cv_folds.csv`. The feature tables in `cache/` come from the local `02_build_features.py` run (`logs/02_build_features.log`, `results/candidates/`). They are byte-identical to the shared tables from the team-review hand-off.

**Work that has only reference evidence (`reference_outputs/`, not re-run locally):**

- 8-model comparison with 5-fold group CV: `reference_outputs/results/models/` and `figures/models/`
- Optuna tuning of LightGBM (30 trials, best CV top-1 0.894, student-test top-1 0.9071): `reference_outputs/results/optuna/`, including `study.db` and `trials.csv`
- HMM decoder tuning and evaluation (accuracy 0.925, route continuity 0.989, route F1 0.950): `reference_outputs/results/hmm/` and `figures/hmm/`
- Saved tuned model: `reference_outputs/artifacts/model.joblib` and `metadata.json`
- Feature-selection evidence: `archive/review1/feature_analysis/` (code: `scripts/10_feature_analysis.py`)

**Still to verify as a team:**

1. Run cross-validation for LR and the decision tree (without `--no-cv`), and run LightGBM, RF, GNB, kNN, HGB and MLP locally. Then compare the results with `reference_outputs/results/models/comparison.csv`.
2. Re-run steps 6–9 (Optuna, decoder tuning, HMM evaluation, error analysis) on a team laptop to create our own `artifacts/` and `results/optuna|hmm/`. Until then, `artifacts/` does not exist and the API and steps 7–9 have no model. For a demo only, you can run `$env:PATHSNAP_ARTIFACTS="reference_outputs/artifacts"`.
3. Run `scripts/10_feature_analysis.py` locally if the report needs feature numbers from our own run.
4. Update the numbers in `report/REPORT.md` from our own `results/`, and replace the two TODO screenshots.
5. Before evaluation day, go through the checklist in `archive/review1/handoff/COMMIT_GUIDE.md`.

## Repository layout

```
configs/      data.yaml · models.yaml · optuna.yaml · hmm.yaml · api.yaml
pathsnap/     the Python package (io, geo, network, labels, candidates, features,
              viz, metrics, models, hmm, inference)
scripts/      01_eda … 09_error_analysis, 10_feature_analysis, run_all.py
api/          FastAPI backend
ui/           evaluator UI (static HTML + Leaflet) and a tiny server
artifacts/    model.joblib + metadata.json (final model, features, HMM params; created by steps 6–7)
results/      CSV / JSON outputs of every experiment
figures/      PNG plots and folium HTML maps
report/       technical report source
reference_outputs/  outputs of the reference run, kept only to check our own runs
archive/review1/    Review 1 material (read-only)
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
python scripts/run_all.py          # all 8 steps, about 30–40 min on a 2-core laptop
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
