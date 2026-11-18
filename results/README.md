# Results

The experiment outputs in this folder and in `figures/` come from one full pipeline run on Sonu's laptop on 2026-11-18:
scripts `01`–`10`, random seed 42, 5-fold GroupKFold by trip on the 80 training trips, each model refit on all training trips and scored once on the 10 student-test trips.
The evaluator trips were not used.

- **Per-person model rows:** `results/team/<member>/results.csv`. Each member ran their own two models on their own laptop.
- **Combined 8-model table:** `results/models/comparison.csv` and `results/models/comparison_table.md`. Purushotham built these from the four `results/team/` files, and this run did not overwrite them.

## Which script produced what

| Script | Output in the repo |
|---|---|
| `scripts/01_eda.py` | `results/eda/`, `figures/eda/` |
| `scripts/02_build_features.py` | `results/candidates/` (the feature tables go to `cache/`, which is not committed) |
| `scripts/03_eda_maps.py` | `figures/maps/eda_*.html` |
| `scripts/04_train_compare.py` | `results/models/cv_folds.csv`, `results/models/details.json` |
| `scripts/05_plot_comparison.py` | `figures/models/` |
| `scripts/06_bayes_opt.py` | `results/optuna/` (incl. `study.db`, `trials.csv`), `figures/optuna/` |
| `scripts/07_tune_decoder.py` | `results/hmm/decoder_grid.csv`, `results/hmm/decoder_tuning.json` |
| `scripts/08_eval_hmm.py` | `results/hmm/student_test_eval.csv`, `results/hmm/student_test_summary.json`, `figures/hmm/`, `figures/maps/student_test_00000076.html` |
| `scripts/09_error_analysis.py` | `results/hmm/error_breakdown.csv`, `results/hmm/error_breakdown_per_trip.csv`, `results/hmm/reverse_error_headings.csv` |
| `scripts/10_feature_analysis.py` | `results/features/`, `figures/features/` |

`results/run_logs/` holds the console output of steps 04–10 from this run (`04_train_compare.log`, `05.log` … `10.log`).
The folder isn't named `logs/` because `.gitignore` excludes every `logs/` folder.

Only 4 of the HTML maps are committed: the three EDA maps and trip 00000076. The other student-test maps are about 2–4 MB each, and `08_eval_hmm.py` writes them again.

## Notes

- `results/optuna/` and `results/hmm/` come from the tuned LightGBM trained in this run: tuned student-test top-1 0.9041, HMM pointwise accuracy 0.9238.
  The committed `artifacts/model.joblib` was trained separately on Ayush's laptop. If you run `08_eval_hmm.py` with it, the numbers can differ slightly.
- Raw data (`data/`), feature tables (`cache/`) and the virtual environment aren't committed. See the main `README.md` to rebuild them.
