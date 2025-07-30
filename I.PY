
import glob
import os
import logging

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# Custom Power-Law regressor: g = alpha * (x_over_V)^beta
# -----------------------------------------------------------------------------
class PowerLawRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, maxfev=10000, p0=(1.0, 0.5)):
        self.maxfev = maxfev
        self.p0 = p0

    def fit(self, X, y):
        # Convert X to numpy array if it's a DataFrame
        X_arr = X.values if hasattr(X, "values") else X
        r = X_arr[:, 0]
        mask = (r > 0) & np.isfinite(y)
        try:
            popt, _ = curve_fit(
                lambda r_, a, b: a * np.power(r_, b),
                r[mask],
                y[mask],
                p0=self.p0,
                maxfev=self.maxfev
            )
            self.alpha_, self.beta_ = popt
        except Exception:
            self.alpha_, self.beta_ = np.nan, np.nan
        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        r = X_arr[:, 0]
        return self.alpha_ * np.power(r, self.beta_)

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logging.info("Starting extended compare_models_pipeline")

# -----------------------------------------------------------------------------
# 1) Discover enhanced slippage files
# -----------------------------------------------------------------------------
files = glob.glob("*_enhanced_slippage.csv")
if not files:
    logging.error("No enhanced slippage CSVs found.")
    raise FileNotFoundError("No enhanced slippage CSVs found.")
logging.info(f"Found {len(files)} enhanced slippage files.")

# -----------------------------------------------------------------------------
# 2) Per-file aggregation
# -----------------------------------------------------------------------------
usecols = ["size", "slippage", "depth", "spread", "imbalance", "volatility", "hour_of_day"]
per_file_aggs = []
for file_id, fp in enumerate(files):
    logging.info(f"[{file_id+1}/{len(files)}] Reading {os.path.basename(fp)}")
    df = pd.read_csv(fp, usecols=usecols, low_memory=True)
    logging.info(f"  -> {len(df):,} rows loaded")
    agg = df.groupby("size").agg({
        "slippage":   "mean",
        "depth":      "mean",
        "spread":     "mean",
        "imbalance":  "mean",
        "volatility": "mean",
        "hour_of_day":"mean"
    }).reset_index()
    logging.info(f"  -> aggregated to {len(agg)} size buckets")
    agg["file_id"] = file_id
    per_file_aggs.append(agg)

# -----------------------------------------------------------------------------
# 3) Combine per-file aggregates
# -----------------------------------------------------------------------------
combined = pd.concat(per_file_aggs, ignore_index=True)
logging.info(f"Combined dataset: {combined.shape[0]:,} rows")

# -----------------------------------------------------------------------------
# 4) Feature engineering
# -----------------------------------------------------------------------------
before = len(combined)
combined = combined[combined["depth"] > 0].copy()
logging.info(f"Dropped {before - len(combined):,} rows with zero depth")
combined["sqrt_xV"]  = np.sqrt(combined["size"] / combined["depth"])
combined["x_over_V"] = combined["size"] / combined["depth"]
combined["log_x"]    = np.log(combined["size"])
logging.info("Feature engineering completed")

feature_cols = [
    "sqrt_xV", "x_over_V", "log_x",
    "spread", "depth", "imbalance",
    "volatility", "hour_of_day"
]
X = combined[feature_cols]
y = combined["slippage"].values
groups = combined["file_id"].values

# -----------------------------------------------------------------------------
# 5) Define models (including PowerLaw and poly(x/V)^2)
# -----------------------------------------------------------------------------
models = {
    'Square-root':   (LinearRegression(fit_intercept=False), ['sqrt_xV']),
    'Linear x/V':    (LinearRegression(fit_intercept=False), ['x_over_V']),
    'Logarithmic':   (LinearRegression(),                   ['log_x']),
    'Quadratic':     (make_pipeline(PolynomialFeatures(2, include_bias=False),
                                    LinearRegression()),       ['size']),
    'PowerLaw x/V':  (PowerLawRegressor(),                  ['x_over_V']),
    'Poly x/V^2':    (make_pipeline(PolynomialFeatures(2, include_bias=False),
                                    LinearRegression()),       ['x_over_V']),
    'ElasticNet':    (ElasticNet(random_state=0),           feature_cols),
    'Ridge':         (Ridge(),                              feature_cols),
    'RandomForest':  (RandomForestRegressor(n_estimators=100, random_state=0),
                                                            feature_cols),
    'GradientBoost': (GradientBoostingRegressor(n_estimators=100, random_state=0),
                                                            feature_cols),
    'KNeighbors':    (KNeighborsRegressor(),                feature_cols),
    'SVR':           (SVR(),                               feature_cols),
    'XGBoost':       (XGBRegressor(n_estimators=100, random_state=0, verbosity=0),
                                                            feature_cols)
}
logging.info(f"Prepared {len(models)} candidate models: {list(models.keys())}")

# -----------------------------------------------------------------------------
# 6) Cross-validated evaluation with GroupKFold
# -----------------------------------------------------------------------------
gkf = GroupKFold(n_splits=5)
scoring = {
    'mse': make_scorer(mean_squared_error, greater_is_better=False),
    'r2': 'r2'
}
results = []

for name, (model, feats) in models.items():
    logging.info(f"Evaluating model: {name}")
    cv_res = cross_validate(
        model,
        combined[feats],
        y,
        groups=groups,
        cv=gkf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    mean_mse = -np.mean(cv_res['test_mse'])
    std_mse  = np.std(cv_res['test_mse'])
    mean_r2  = np.mean(cv_res['test_r2'])
    std_r2   = np.std(cv_res['test_r2'])
    logging.info(f"{name}: MSE={mean_mse:.4f}±{std_mse:.4f}, R2={mean_r2:.3f}±{std_r2:.3f}")
    results.append({
        'model':         name,
        'mean_test_mse': mean_mse,
        'std_test_mse':  std_mse,
        'mean_test_r2':  mean_r2,
        'std_test_r2':   std_r2
    })

# -----------------------------------------------------------------------------
# 7) Report results
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values('mean_test_mse')
print("\nCross-validated Model Performance:")
print(results_df.to_string(index=False))


