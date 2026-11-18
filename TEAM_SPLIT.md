# Team split

Each member owns one part of the pipeline. You commit your own part from your own GitHub account, re-run it, and must be able to explain it in the viva without notes.

---

## Sonu Akash: data, EDA, candidates, features

**Files:** `pathsnap/io.py`, `geo.py`, `network.py`, `labels.py`, `candidates.py`, `features.py`, `viz.py`, `config.py`, `configs/data.yaml`, `scripts/01_eda.py`, `02_build_features.py`, `03_eda_maps.py`, `requirements.txt`, `.gitignore`

**Be ready to explain**
- What each raw file holds, and how track ↔ nodes ↔ arcs ↔ route relate. Arc id = row index in `.arcs`; arcs are directed.
- Cleaning steps in `clean_track`, and what the EDA found: 0 invalid rows, 96.5% 1 s sampling, 22 trips with gaps over 60 s, two over an hour.
- Why we project each trip to its own UTM zone instead of using degrees (1° lat ≈ 111 km, 1° lon = 111 km × cos lat), and why per trip: the trips are on several continents (25 zones in train). Also the haversine check (≤ 0.55% error).
- How labels are made: monotone DP alignment of GPS points to route arcs (`labels.py`). Why some points are unlabelled (trips 54, 33, 2: the route covers only part of the track).
- Candidate generation: KD-tree over points sampled every 5 m along arcs; 50 m radius and k = 8, chosen from the distance distribution (99th percentile ≈ 27–37 m); candidate miss rate ≈ 0.7%.
- Each feature group and why it helps: heading separates the two directions of a street; window distance handles single noisy points.

## Purushotham: models and comparison

**Files:** `pathsnap/metrics.py`, `pathsnap/models.py`, `configs/models.yaml`, `scripts/04_train_compare.py`, `05_plot_comparison.py`

**Be ready to explain**
- Problem framing: binary classification per (observation, candidate), then top-1 per observation.
- Why GroupKFold by trip: points from one trip are highly correlated, so a random split would leak.
- Why only labelled observations are used, and why every 3rd observation (1 Hz near-duplicates, compute time).
- Each of the 8 models: its family, its main hyperparameters, its strengths and weaknesses on *our* results. For example: GNB has the worst log loss because the features are correlated; KNN is slowest at inference (≈1.6 s / 1k obs); RF and LightGBM overfit a little (in-sample about 0.94–0.95 vs CV 0.89).
- The metrics: top-1 accuracy (our primary metric, and why), macro F1, log loss, candidate miss rate, route continuity.
- Why LightGBM was selected: best CV top-1, best log loss (which matters for the HMM emissions), best student-test score, fast enough.

## Ayush: Bayesian optimisation and HMM

**Files:** `pathsnap/hmm.py`, `pathsnap/inference.py`, `configs/optuna.yaml`, `configs/hmm.yaml`, `scripts/06_bayes_opt.py`, `07_tune_decoder.py`, `08_eval_hmm.py`, `artifacts/`

**Be ready to explain**
- Optuna TPE: what Bayesian optimisation is (a surrogate model of the objective, picking promising points), the search space and why each range, the objective (3-fold group CV top-1), the median pruner, the number of trials, and tuned vs untuned.
- Why LightGBM early stopping uses an inner split of *training* trips, never the fold being scored.
- The HMM: hidden states = candidate arcs; emission = normalised classifier probability; transition = exp(−|route distance − straight-line distance| / β); the U-turn penalty; the Viterbi recurrence; backtracking.
- Gaps and failures: chain cut when dt > gap_seconds or when every transition is impossible; points without candidates get no prediction.
- The bug we found and fixed: the decoder got stuck on the wrong direction of a two-way street. We now allow a penalised mid-arc U-turn.
- Why the decoder is tuned on *out-of-fold* training probabilities.
- Classifier-only vs HMM results, and the trips where the HMM did not help.

## Sahana: API, UI, deployment, report

**Files:** `api/main.py`, `configs/api.yaml`, `ui/*`, `scripts/run_all.py`, `README.md`, `report/`

**Be ready to explain**
- The two-laptop setup: what runs where, how they talk (HTTP and JSON, CORS), the firewall and IP address.
- The `/trips` and `/predict` flow; why the route file is read only after prediction; what the UI does and does not receive.
- Where the logs are, and what the evaluator sees in the console.
- How to run everything from a fresh clone (README).
- The report structure and where each number comes from.

---

## Everyone

- Read `README.md` fully and run `python scripts/run_all.py` at least once on your own laptop.
- Know the final numbers in `results/` and where they come from.
