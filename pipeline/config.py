"""
config.py — Single source of truth for all paths and hyperparameters.

All other pipeline stages import from here. Change a path or param once,
and it takes effect everywhere.
"""

import os

# ── Root directories ──────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_ROOT = os.path.join(PROJECT_ROOT, "src", "data")
OUTPUTS_ROOT  = os.path.join(PROJECT_ROOT, "outputs")

# ── Raw input paths (never modified by the pipeline) ─────────────────────────
SNAP_FILE         = os.path.join(RAW_DATA_ROOT, "SNAPApps", "SNAPData.csv")
COUNTY_METRO_FILE = os.path.join(RAW_DATA_ROOT, "county_to_metro.csv")
POP_FILE          = os.path.join(RAW_DATA_ROOT, "popData.csv")
INCOME_FILE       = os.path.join(RAW_DATA_ROOT, "MedianIncome.csv")
TRENDS_DIR        = os.path.join(RAW_DATA_ROOT, "trends")
PREDICTION_DIR    = os.path.join(RAW_DATA_ROOT, "prediction")

# ── Generated output paths ────────────────────────────────────────────────────
TRAINING_DATA_CSV      = os.path.join(OUTPUTS_ROOT, "data", "training_data.csv")
FEATURES_CSV            = os.path.join(OUTPUTS_ROOT, "data", "features.csv")
FEATURE_REGISTRY_CSV    = os.path.join(OUTPUTS_ROOT, "data", "feature_registry.csv")
FEATURE_DICTIONARY_CSV  = os.path.join(OUTPUTS_ROOT, "data", "feature_dictionary.csv")
SCALING_PARAMS_JSON    = os.path.join(OUTPUTS_ROOT, "data", "trend_scaling_params.json")
MODEL_PKL              = os.path.join(OUTPUTS_ROOT, "models", "xgboost_tuned.pkl")
INSAMPLE_METRICS_JSON  = os.path.join(OUTPUTS_ROOT, "metrics", "insample_metrics.json")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUTS_ROOT, "metrics", "feature_importance.csv")
WF_OVERALL_JSON        = os.path.join(OUTPUTS_ROOT, "metrics", "walkforward_overall.json")
WF_PER_MONTH_CSV       = os.path.join(OUTPUTS_ROOT, "metrics", "walkforward_per_month.csv")
PREDICTIONS_CSV        = os.path.join(OUTPUTS_ROOT, "predictions", "finalPrediction.csv")
PAPER_SUMMARY_JSON     = os.path.join(OUTPUTS_ROOT, "metrics", "paper_summary.json")
FIGURES_DIR            = os.path.join(OUTPUTS_ROOT, "figures")

# ── Model features and target ─────────────────────────────────────────────────
KEYWORDS = ["CalFresh", "FoodBank"]

# Base features used at prediction time (no lags — future data unavailable)
# Also used as fallback if features.csv has not been built yet.
BASE_FEATURE_COLS = [
    "Population",
    "Median_Income",
    "monthly_average_CalFresh",
    "monthly_average_FoodBank",
    "month",
]

# Engineered feature list used by stages 3 and 4 (train + evaluate).
# Validated by walk-forward: R²=0.91, MAE=0.000381 vs R²=0.77, MAE=0.000683 base.
# Year-over-year (lag-12) was tested and excluded — halved dataset for +0.003 R².
FEATURE_COLS = [
    # Base
    "Population", "Median_Income",
    "monthly_average_CalFresh", "monthly_average_FoodBank", "month",
    # Lags — SNAP rate
    "rate_lag1", "rate_lag2", "rate_lag3",
    # Lags — Google Trends
    "calfresh_lag1", "calfresh_lag2",
    "foodbank_lag1", "foodbank_lag2",
    # Rolling windows
    "rate_roll3_mean", "rate_roll3_std",
    "calfresh_roll3", "foodbank_roll3",
    # Momentum
    "calfresh_momentum", "foodbank_momentum",
    # Seasonality
    "month_sin", "month_cos", "quarter",
    # Transforms
    "log_population", "log_income", "income_quintile",
]

# Which CSV stages 3 and 4 read from
MODELLING_CSV = FEATURES_CSV  # set to TRAINING_DATA_CSV to use base features only

TARGET_COL = "SNAP_Application_Rate"  # = SNAP_Applications / Population

# ── XGBoost tuned hyperparameters ─────────────────────────────────────────────
# Determined via RandomizedSearch + GridSearch in experiments/walk_forward_production.py.
# Production walk-forward results: R²=0.338, MAE=0.000877, sMAPE=13.46%
# (vs default XGBoost R²=-0.037 — tuning is essential for this dataset)
XGBOOST_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.01,
    min_child_weight=6,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=4,
    reg_alpha=0,
    random_state=42,
    n_jobs=-1,
)

# ── Walk-forward validation settings ─────────────────────────────────────────
# Minimum months of history required before the first test window opens.
# With monthly SNAP data, 12 months gives the model a full seasonal cycle to learn from.
WALK_FORWARD_MIN_MONTHS = 12

# ── Data cleaning ─────────────────────────────────────────────────────────────
# Rows where SNAP_Applications > OUTLIER_THRESHOLD × county median are dropped.
# Catches data-entry spikes (e.g. Madera Jan 2023: 11,090 vs typical ~1,000).
OUTLIER_THRESHOLD = 3.0

# ── Convenience: ensure all output directories exist ─────────────────────────
def ensure_output_dirs():
    for d in [
        os.path.join(OUTPUTS_ROOT, "data"),
        os.path.join(OUTPUTS_ROOT, "models"),
        os.path.join(OUTPUTS_ROOT, "metrics"),
        os.path.join(OUTPUTS_ROOT, "predictions"),
        FIGURES_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
