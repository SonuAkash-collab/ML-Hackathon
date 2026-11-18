# Sonu (SonuAkash-collab): logreg and tree

`results.csv` holds the `logreg` and `tree` rows, copied unchanged from
`results/models/comparison.csv` of a run on Sonu's own laptop.

- **Command:** `.venv\Scripts\python.exe scripts\04_train_compare.py` (all 8 models in one run;
  only the logreg and tree rows are committed here, the other six belong to teammates)
- **Features:** built beforehand with `python scripts/02_build_features.py`
- **CV setting:** full 5-fold GroupKFold by trip on the 80 training trips (not `--no-cv`),
  then refit on all training trips and scored once on the 10 student-test trips
- **Date:** 2026-11-18
- **Machine:** Sonu's laptop (Windows 11, Python 3.13.3)
- **Log:** `logs/04_train_compare.log` on that laptop (logs/ is not committed)

| model | cv_top1 | test_top1 |
|---|---|---|
| tree | 0.8885 | 0.8932 |
| logreg | 0.8872 | 0.8833 |
