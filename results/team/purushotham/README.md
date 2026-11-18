# Purushotham (Purushothama449): lgbm and hgb

`results.csv` holds the `lgbm` and `hgb` rows, copied unchanged from
`results/models/comparison.csv` of a run on Purushotham's own laptop.

- **Command:** `python scripts/04_train_compare.py lgbm hgb`
- **Features:** built beforehand with `python scripts/02_build_features.py`
- **CV setting:** full 5-fold GroupKFold by trip on the 80 training trips (no `--no-cv`),
  then refit on all training trips and scored once on the 10 student-test trips
- **Date:** 2026-11-18
- **Machine:** Purushotham's laptop (Windows 11, Python 3.13.2)
- **Log:** `logs/04_train_compare.log` on that laptop (logs/ is not committed)

| model | cv_top1 | test_top1 |
|---|---|---|
| lgbm | 0.8945 | 0.9042 |
| hgb | 0.8919 | 0.8957 |
