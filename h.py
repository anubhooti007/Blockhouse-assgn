import glob
import os
import pandas as pd
import numpy as np
import logging
from numba import njit

# ----------------------------------------------------------------------------
# Configure logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

# ----------------------------------------------------------------------------
# 1) Numba‐accelerated slippage calculator for entire batch
# ----------------------------------------------------------------------------
@njit
def compute_slippages_numba_all(asks_px_arr, asks_sz_arr, mids, x_grid):
    n_snap, _ = asks_px_arr.shape
    n_x = x_grid.shape[0]
    sl = np.empty((n_snap, n_x), dtype=np.float64)
    for i in range(n_snap):
        mid_price = mids[i]
        for j in range(n_x):
            remaining = x_grid[j]
            cost = 0.0
            for lvl in range(asks_px_arr.shape[1]):
                price = asks_px_arr[i, lvl]
                size = asks_sz_arr[i, lvl]
                take = remaining if size >= remaining else size
                cost += take * price
                remaining -= take
                if remaining <= 0:
                    break
            exec_price = cost / x_grid[j] if x_grid[j] > 0 else mid_price
            sl[i, j] = exec_price - mid_price
    return sl

# Warm up JIT
logging.info("Warming up Numba JIT for slippage calculator...")
_dummy_px = np.zeros((1, 10), dtype=np.float64)
_dummy_sz = np.zeros((1, 10), dtype=np.float64)
_dummy_mid = np.zeros(1, dtype=np.float64)
_dummy_x = np.arange(100, 200, 100, dtype=np.float64)
_ = compute_slippages_numba_all(_dummy_px, _dummy_sz, _dummy_mid, _dummy_x)

# ----------------------------------------------------------------------------
# 2) Main script
# ----------------------------------------------------------------------------
x_grid = np.arange(100, 2100, 100, dtype=np.float64)

# Find raw MBP-10 files
raw_files = [
    f for f in glob.glob("*_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*.csv")
    if not f.endswith("_enhanced_slippage.csv")
]
logging.info(f"Found {len(raw_files)} raw files to process.")

for file_path in raw_files:
    base = os.path.basename(file_path)
    logging.info(f"Processing file: {base}")

    # parse token & date
    token, rest = base.split("_", 1)
    date_str = rest[:-4]  # drop ".csv"

    # load raw snapshot
    df_snap = pd.read_csv(file_path)
    # Handle timestamp parsing
    if 'ts_event' in df_snap.columns:
        df_snap['ts_event'] = pd.to_datetime(df_snap['ts_event'], format='mixed')
    else:
        logging.warning(f"No ts_event column found in {base}, skipping...")
        continue
    n_snapshots = len(df_snap)
    logging.info(f"Loaded {n_snapshots} snapshots from {base}")

    # core metrics
    df_snap["mid_price"] = 0.5 * (df_snap["bid_px_00"] + df_snap["ask_px_00"])
    df_snap["spread"]    = df_snap["ask_px_00"] - df_snap["bid_px_00"]
    bid_cols = [f"bid_sz_{i:02d}" for i in range(10)]
    ask_px_cols = [f"ask_px_{i:02d}" for i in range(10)]
    ask_sz_cols = [f"ask_sz_{i:02d}" for i in range(10)]
    df_snap["depth"]     = df_snap[bid_cols + ask_sz_cols].sum(axis=1)
    df_snap["imbalance"] = (
        df_snap["bid_sz_00"] - df_snap["ask_sz_00"]
    ) / (df_snap["bid_sz_00"] + df_snap["ask_sz_00"] + 1e-9)
    df_snap["hour"]      = df_snap["ts_event"].dt.hour
    df_snap["mid_return"] = df_snap["mid_price"].pct_change()
    df_snap["volatility"] = df_snap["mid_return"].rolling(window=60, min_periods=1).std()

    # extract arrays for Numba
    asks_px_arr = df_snap[ask_px_cols].to_numpy(dtype=np.float64)
    asks_sz_arr = df_snap[ask_sz_cols].to_numpy(dtype=np.float64)
    mids        = df_snap["mid_price"].to_numpy(dtype=np.float64)

    logging.info("Computing slippage matrix via Numba...")
    slippages = compute_slippages_numba_all(asks_px_arr, asks_sz_arr, mids, x_grid)
    logging.info(f"Computed slippages matrix with shape {slippages.shape}")

    # build enhanced records
    records = []
    for i in range(slippages.shape[0]):
        row = df_snap.iloc[i]
        for j, x in enumerate(x_grid):
            records.append({
                "timestamp":   row["ts_event"],
                "size":        float(x),
                "slippage":    float(slippages[i, j]),
                "vol_ratio":   float(x / row["depth"]) if row["depth"] > 0 else np.nan,
                "spread":      float(row["spread"]),
                "depth":       float(row["depth"]),
                "imbalance":   float(row["imbalance"]),
                "volatility":  float(row["volatility"]),
                "hour_of_day": int(row["hour"]),
            })
    logging.info(f"Built {len(records)} enhanced slippage records")

    # write enhanced file
    enhanced_df = pd.DataFrame(records)
    out_name = f"{token}_{date_str}_enhanced_slippage.csv"
    enhanced_df.to_csv(out_name, index=False)
    logging.info(f"Written enhanced file: {out_name}")

logging.info("All files processed.")

