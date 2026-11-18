"""Run the whole training pipeline in order (Laptop 1).

    python scripts/run_all.py            # everything
    python scripts/run_all.py --from 6   # resume from step 6

Steps
  1  EDA                      scripts/01_eda.py
  2  candidates + features    scripts/02_build_features.py
  3  EDA folium maps          scripts/03_eda_maps.py
  4  model comparison         scripts/04_train_compare.py
  5  comparison plots/table   scripts/05_plot_comparison.py
  6  Bayesian optimisation    scripts/06_bayes_opt.py      -> artifacts/model.joblib
  7  decoder tuning           scripts/07_tune_decoder.py   -> artifacts/metadata.json
  8  HMM evaluation + maps    scripts/08_eval_hmm.py
  9  error analysis           scripts/09_error_analysis.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_eda.py", "02_build_features.py", "03_eda_maps.py", "04_train_compare.py",
         "05_plot_comparison.py", "06_bayes_opt.py", "07_tune_decoder.py", "08_eval_hmm.py",
         "09_error_analysis.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=len(STEPS))
    args = ap.parse_args()
    for i in range(args.start, args.end + 1):
        script = STEPS[i - 1]
        print(f"\n{'=' * 70}\nstep {i}: {script}\n{'=' * 70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / script)])
        if r.returncode != 0:
            sys.exit(f"step {i} ({script}) failed with exit code {r.returncode}")
        print(f"step {i} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
