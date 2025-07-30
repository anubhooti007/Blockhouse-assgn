# Assignment Submission


## Contents
Task 1

  * `h.py` — Numba‑accelerated slippage calculator: processes raw market‑by‑price snapshots into enhanced slippage datasets.
  * `I.PY` — Model comparison pipeline: aggregates enhanced data, engineers features, and evaluates candidate regression models via cross‑validation.
Task 1 submission PDF
Task 2 submission PDF

Task 1: Slippage Calculation & Model Comparison

1. Process raw snapshots
   Convert raw MBP‑10 CSVs into enhanced slippage records:
   ```bash
   python h.py
   ```
   * Input: files matching `*_YYYY-MM-DD*.csv`
   * Output: `*_enhanced_slippage.csv` files

2. Compare predictive models
   Evaluate models on the enhanced datasets:
   ```bash
   python I.PY
   ```
   * Reads `*_enhanced_slippage.csv`, aggregates by size buckets, and runs cross‑validation
   * Prints MSE and R² for each candidate model

