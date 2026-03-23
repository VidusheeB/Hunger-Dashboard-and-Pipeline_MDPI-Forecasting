"""
benchmark_models.py — Compare all candidate models under identical walk-forward validation.

Reads the engineered feature dataset (outputs/data/features.csv) produced by the
pipeline and evaluates every model under the same temporal walk-forward loop used
in stage4_evaluate.py.  No information from the test month is ever used during
training for any model.

Models evaluated
----------------
1.  Naive (last-month)       — county's SNAP rate from T-1; no training at all
2.  Linear Regression        — sklearn LinearRegression on all FEATURE_COLS
3.  Random Forest            — RandomForestRegressor(n_estimators=100)
4.  Gradient Boosting        — GradientBoostingRegressor(n_estimators=200)
5.  XGBoost (default)        — XGBRegressor() with no hyperparameter tuning
6.  XGBoost (tuned)          — config.XGBOOST_PARAMS (production model settings)
7.  ARIMA(1,1,1)             — per-county univariate time-series (statsmodels)
8.  SARIMAX(1,1,1)×(1,0,1,12)— seasonal ARIMA with Trends as exogenous regressors

ARIMA/SARIMAX are optional: if statsmodels is not installed they are skipped with
a clear warning rather than crashing the whole benchmark.

Outputs
-------
  outputs/metrics/benchmark_comparison.csv    — one row per model, all metrics
  outputs/metrics/benchmark_per_month.csv     — one row per (model, month)

Run
---
    python experiments/benchmark_models.py                # all models
    python experiments/benchmark_models.py --no-arima     # skip ARIMA/SARIMAX
    python experiments/benchmark_models.py --models naive,lr,rf  # subset
"""

import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Optional statsmodels ──────────────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning(
        "statsmodels not installed — ARIMA and SARIMAX will be skipped.\n"
        "  Install with: pip install statsmodels"
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², RMSE, MAE, sMAPE — same formula as stage4_evaluate.py."""
    y_pred = np.clip(y_pred, 0, None)
    denom  = np.abs(y_true) + np.abs(y_pred)
    smape  = float(np.mean(2 * np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)) * 100)
    return {
        "r2":    float(r2_score(y_true, y_pred)),
        "rmse":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "smape": smape,
    }


# ── Model registry ────────────────────────────────────────────────────────────

def get_models(include_arima: bool = True) -> dict:
    """
    Return an ordered dict of {model_key: (display_name, model_object_or_None)}.

    None means the model is handled specially in the walk-forward loop
    (Naive, ARIMA, SARIMAX).
    """
    models = {
        "naive": ("Naive (last-month)", None),
        "lr":    ("Linear Regression",  LinearRegression()),
        "rf":    ("Random Forest",       RandomForestRegressor(
                      n_estimators=100, random_state=42, n_jobs=-1)),
        "gb":    ("Gradient Boosting",   GradientBoostingRegressor(
                      n_estimators=200, max_depth=4, learning_rate=0.05,
                      random_state=42)),
        "xgb_default": ("XGBoost (default)", xgb.XGBRegressor(
                      random_state=42, n_jobs=-1)),
        "xgb_tuned":   ("XGBoost (tuned)",   xgb.XGBRegressor(**config.XGBOOST_PARAMS)),
    }
    if include_arima and HAS_STATSMODELS:
        models["arima"]   = ("ARIMA(1,1,1)",                  None)
        models["sarimax"] = ("SARIMAX(1,1,1)×(1,0,1,12)",    None)
    return models


# ── Naive model ───────────────────────────────────────────────────────────────

def build_naive_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute a county × month lookup of T-1 SNAP rates.

    For test month T and county C, the naive prediction is the SNAP rate
    observed for county C in the most recent prior month available in the
    training data.
    """
    df = df.copy().sort_values(["county", "date"])
    df["naive_pred"] = df.groupby("county")[config.TARGET_COL].shift(1)
    return df.set_index(["county", "date"])["naive_pred"]


# ── ARIMA / SARIMAX helpers ───────────────────────────────────────────────────

SARIMAX_EXOG_COLS = ["monthly_average_CalFresh", "monthly_average_FoodBank"]

def _fit_arima(rate_series: pd.Series) -> float:
    """
    Fit ARIMA(1,1,1) on a county's SNAP rate history and return a 1-step forecast.
    Returns NaN on failure (e.g. too few observations, convergence error).
    """
    if len(rate_series) < 6:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMA(rate_series.values, order=(1, 1, 1)).fit()
            return float(np.clip(m.forecast(steps=1)[0], 0, None))
    except Exception:
        return float("nan")


def _fit_sarimax(
    rate_series: pd.Series,
    exog_train: pd.DataFrame,
    exog_test: pd.Series,
) -> float:
    """
    Fit SARIMAX(1,1,1)(1,0,1,12) with Trends as exogenous regressors.
    Falls back to ARIMA if seasonal order causes convergence issues.
    """
    if len(rate_series) < 13:
        return _fit_arima(rate_series)
    try:
        # Align exog to the rate series index
        exog_train = exog_train.reindex(rate_series.index).fillna(method="ffill").fillna(0)
        if exog_train.isna().any().any():
            return _fit_arima(rate_series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(
                rate_series.values,
                exog=exog_train.values,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        exog_fc = np.array(exog_test).reshape(1, -1)
        return float(np.clip(m.forecast(steps=1, exog=exog_fc)[0], 0, None))
    except Exception:
        return _fit_arima(rate_series)


# ── Main walk-forward loop ────────────────────────────────────────────────────

def run_benchmark(df: pd.DataFrame, models: dict) -> tuple:
    """
    Run all models under identical walk-forward validation.

    For each test month T (after MIN_MONTHS of history):
      - ML models:   fit on all rows with date < T, predict rows with date == T
      - Naive:       look up county's most recent prior rate from training data
      - ARIMA/SARIMAX: fit per county on rate history before T, forecast T

    Returns:
      overall_df   — DataFrame with one row per model (headline metrics)
      per_month_df — DataFrame with one row per (model, month)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    feature_cols = [f for f in config.FEATURE_COLS if f in df.columns]
    logger.info(f"  Features used by ML models: {len(feature_cols)}")
    missing = set(config.FEATURE_COLS) - set(feature_cols)
    if missing:
        logger.warning(f"  Missing features (will be excluded): {missing}")

    # Naive lookup over the full dataset
    naive_lookup = build_naive_lookup(df)

    # Per-model accumulators: {key: {"true": [...], "pred": [...], "months": [...], "counties": [...]}}
    accum = {k: {"true": [], "pred": [], "months": [], "counties": []} for k in models}
    per_month_rows = []

    n_test_months = len(dates) - config.WALK_FORWARD_MIN_MONTHS
    logger.info(f"  Walk-forward: {n_test_months} test months, "
                f"starting from month {config.WALK_FORWARD_MIN_MONTHS + 1}")

    for i, test_date in enumerate(dates[config.WALK_FORWARD_MIN_MONTHS:], 1):
        month_str = pd.Timestamp(test_date).strftime("%Y-%m")
        if i % 5 == 0 or i == 1:
            logger.info(f"  [{i}/{n_test_months}] Testing {month_str} ...")

        train_mask = df["date"] < test_date
        test_mask  = df["date"] == test_date

        train_df = df[train_mask]
        test_df  = df[test_mask]

        # Clean NaN for ML models
        ml_tr_ok = train_df[feature_cols].notna().all(axis=1) & train_df[config.TARGET_COL].notna()
        ml_te_ok = test_df[feature_cols].notna().all(axis=1)  & test_df[config.TARGET_COL].notna()

        X_tr = train_df.loc[ml_tr_ok, feature_cols]
        y_tr = train_df.loc[ml_tr_ok, config.TARGET_COL].clip(lower=0)
        X_te = test_df.loc[ml_te_ok, feature_cols]
        y_te = test_df.loc[ml_te_ok, config.TARGET_COL].values

        if len(X_tr) < 10 or len(X_te) == 0:
            continue

        for key, (name, model_obj) in models.items():
            try:
                if key == "naive":
                    preds, actuals, counties_seen = [], [], []
                    for _, row in test_df[ml_te_ok].iterrows():
                        naive_val = naive_lookup.get((row["county"], test_date))
                        if pd.isna(naive_val):
                            continue
                        preds.append(float(np.clip(naive_val, 0, None)))
                        actuals.append(row[config.TARGET_COL])
                        counties_seen.append(row["county"])
                    if not preds:
                        continue
                    y_pred      = np.array(preds)
                    y_actual    = np.array(actuals)
                    row_counties = counties_seen

                elif key in ("arima", "sarimax"):
                    preds, actuals, counties_seen = [], [], []
                    for county, county_test in test_df[ml_te_ok].groupby("county"):
                        county_hist = train_df[train_df["county"] == county].sort_values("date")
                        rate_series = county_hist.set_index("date")[config.TARGET_COL].dropna()
                        if len(rate_series) < 6:
                            continue
                        actual_val = county_test[config.TARGET_COL].values[0]

                        if key == "arima":
                            pred_val = _fit_arima(rate_series)
                        else:
                            exog_cols_avail = [c for c in SARIMAX_EXOG_COLS if c in county_hist.columns]
                            if not exog_cols_avail:
                                pred_val = _fit_arima(rate_series)
                            else:
                                exog_train = county_hist.set_index("date")[exog_cols_avail].reindex(rate_series.index)
                                exog_test_row = county_test[exog_cols_avail].fillna(0)
                                if exog_test_row.empty:
                                    pred_val = _fit_arima(rate_series)
                                else:
                                    pred_val = _fit_sarimax(rate_series, exog_train, exog_test_row.values[0])

                        if not np.isnan(pred_val):
                            preds.append(pred_val)
                            actuals.append(actual_val)
                            counties_seen.append(county)

                    if not preds:
                        continue
                    y_pred       = np.array(preds)
                    y_actual     = np.array(actuals)
                    row_counties = counties_seen

                else:
                    # Standard sklearn / XGBoost ML model
                    model_obj.fit(X_tr, y_tr)
                    y_pred       = np.clip(model_obj.predict(X_te), 0, None)
                    y_actual     = y_te
                    row_counties = test_df.loc[ml_te_ok, "county"].tolist()

                if len(y_actual) == 0:
                    continue

                m = calc_metrics(y_actual, y_pred)
                accum[key]["true"].extend(y_actual.tolist())
                accum[key]["pred"].extend(y_pred.tolist())
                accum[key]["months"].extend([month_str] * len(y_actual))
                accum[key]["counties"].extend(row_counties)

                per_month_rows.append({
                    "model":      key,
                    "model_name": name,
                    "month":      month_str,
                    "n":          len(y_actual),
                    **m,
                })

            except Exception as e:
                logger.warning(f"  [{key}] failed for {month_str}: {e}")
                continue

    # ── Aggregate overall metrics per model ───────────────────────────────────
    overall_rows = []
    for key, (name, _) in models.items():
        data = accum[key]
        if not data["true"]:
            logger.warning(f"  {name}: no predictions generated — skipping")
            continue
        y_true_all = np.array(data["true"])
        y_pred_all = np.array(data["pred"])
        m = calc_metrics(y_true_all, y_pred_all)
        m["months_tested"] = len(set(data["months"]))  # unique months
        m["n_predictions"] = len(y_true_all)
        overall_rows.append({"model": key, "model_name": name, **m})
        logger.info(
            f"  {name:<35s}  R²={m['r2']:+.4f}  MAE={m['mae']:.6f}  "
            f"sMAPE={m['smape']:.2f}%  n={m['n_predictions']:,}"
        )

    overall_df   = pd.DataFrame(overall_rows)
    per_month_df = pd.DataFrame(per_month_rows)
    return overall_df, per_month_df, accum


# ── Uncertainty estimation ────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> dict:
    """
    Non-parametric bootstrap 95% CIs for R², RMSE, MAE, and sMAPE.

    Resamples the (y_true, y_pred) pairs with replacement ``n_bootstrap`` times
    and computes each metric on each resample.  The CI is the (alpha/2, 1-alpha/2)
    percentile interval — no normality assumption required.

    This is the right tool here because:
    - The walk-forward predictions are NOT i.i.d. (temporal + cross-county
      correlation), so formula-based CIs (±1.96·SE) would be wrong.
    - Bootstrap CIs are still approximate (they don't account for temporal
      dependence), but they are the standard approach for ML metric uncertainty
      and are much better than reporting a point estimate alone.

    Returns: {metric: {"lower": float, "upper": float, "width": float}}
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    alpha = (100.0 - ci) / 2.0

    boot_r2, boot_rmse, boot_mae, boot_smape = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        m = calc_metrics(yt, yp)
        boot_r2.append(m["r2"])
        boot_rmse.append(m["rmse"])
        boot_mae.append(m["mae"])
        boot_smape.append(m["smape"])

    def _ci(samples):
        lo = float(np.percentile(samples, alpha))
        hi = float(np.percentile(samples, 100.0 - alpha))
        return {"lower": lo, "upper": hi, "width": hi - lo}

    return {
        "r2":    _ci(boot_r2),
        "rmse":  _ci(boot_rmse),
        "mae":   _ci(boot_mae),
        "smape": _ci(boot_smape),
    }


def fold_variability_ci(per_month_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fold-to-fold (month-to-month) variability CIs for MAE and R².

    Each walk-forward test month is one "fold."  We treat the per-month metric
    values as a sample and compute:
        95% CI = mean ± t_{n-1, 0.025} × (std / √n)

    This uses the t-distribution because n_months is small (~20), where the
    normal approximation is inappropriate.  The resulting CI answers:
    "If we ran this on a new set of months, where would the mean metric land?"

    Returns a DataFrame with one row per model.
    """
    from scipy import stats

    rows = []
    for (model, name), grp in per_month_df.groupby(["model", "model_name"]):
        for metric in ("mae", "r2", "smape"):
            vals = grp[metric].dropna().values
            n = len(vals)
            if n < 2:
                continue
            mean = float(vals.mean())
            se   = float(vals.std(ddof=1) / np.sqrt(n))
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            rows.append({
                "model":      model,
                "model_name": name,
                "metric":     metric,
                "n_folds":    n,
                "mean":       round(mean, 8),
                "std":        round(float(vals.std(ddof=1)), 8),
                "ci_lower":   round(mean - t_crit * se, 8),
                "ci_upper":   round(mean + t_crit * se, 8),
                "ci_width":   round(2 * t_crit * se, 8),
            })
    return pd.DataFrame(rows)


def county_residual_stats(accum: dict, models: dict) -> pd.DataFrame:
    """
    Compute per-county residual distribution for each model.

    For each (model, county) pair:
      - bias  = mean(pred - true)      → systematic over/under-prediction
      - MAE   = mean |pred - true|     → absolute error magnitude
      - std   = std(pred - true)       → consistency across months
      - n     = number of predictions  → how many months this county was tested

    Counties with large bias are systematically mis-predicted — worth
    flagging for further inspection (data quality, unusual demographics, etc.).
    """
    rows = []
    for key, (name, _) in models.items():
        data = accum[key]
        if not data["true"]:
            continue
        df = pd.DataFrame({
            "county": data["counties"],
            "true":   data["true"],
            "pred":   data["pred"],
        })
        df["residual"] = df["pred"] - df["true"]
        df["abs_err"]  = df["residual"].abs()

        for county, grp in df.groupby("county"):
            rows.append({
                "model":      key,
                "model_name": name,
                "county":     county,
                "n":          len(grp),
                "bias":       round(float(grp["residual"].mean()), 8),
                "mae":        round(float(grp["abs_err"].mean()), 8),
                "std":        round(float(grp["residual"].std(ddof=1)) if len(grp) > 1 else 0.0, 8),
                "max_abs_err":round(float(grp["abs_err"].max()), 8),
            })
    return pd.DataFrame(rows)


def compute_all_uncertainty(
    accum: dict,
    models: dict,
    per_month_df: pd.DataFrame,
    n_bootstrap: int = 2000,
) -> tuple:
    """
    Run all three uncertainty analyses and return (bootstrap_df, fold_ci_df, county_df).

    bootstrap_df  — one row per (model, metric) with 95% bootstrap CI bounds
    fold_ci_df    — one row per (model, metric) with t-distribution fold CI
    county_df     — one row per (model, county) with bias/MAE/std
    """
    logger.info("  Computing bootstrap CIs (2000 resamples per model)...")
    boot_rows = []
    for key, (name, _) in models.items():
        data = accum[key]
        if len(data["true"]) < 30:
            logger.warning(f"  {name}: too few predictions for bootstrap ({len(data['true'])}) — skipping")
            continue
        y_true = np.array(data["true"])
        y_pred = np.array(data["pred"])
        cis = bootstrap_ci(y_true, y_pred, n_bootstrap=n_bootstrap)
        for metric, bounds in cis.items():
            point = calc_metrics(y_true, y_pred)[metric]
            boot_rows.append({
                "model":      key,
                "model_name": name,
                "metric":     metric,
                "point":      round(point, 8),
                "ci_lower":   round(bounds["lower"], 8),
                "ci_upper":   round(bounds["upper"], 8),
                "ci_width":   round(bounds["width"], 8),
            })
        logger.info(f"    {name}: done")

    logger.info("  Computing fold-to-fold variability CIs...")
    fold_ci_df = fold_variability_ci(per_month_df)

    logger.info("  Computing county residual distributions...")
    county_df = county_residual_stats(accum, models)

    return pd.DataFrame(boot_rows), fold_ci_df, county_df


def print_uncertainty_table(bootstrap_df: pd.DataFrame, fold_ci_df: pd.DataFrame) -> None:
    """Print bootstrap and fold CIs side-by-side for MAE and R²."""
    print("\n" + "=" * 95)
    print("  UNCERTAINTY ESTIMATES — 95% Confidence Intervals")
    print("  Bootstrap CI: resampling predictions | Fold CI: t-distribution over monthly folds")
    print("=" * 95)
    print(f"  {'Model':<33} {'Bootstrap MAE CI':^28} {'Fold MAE CI':^28}")
    print(f"  {'':33} {'lower':>8} {'upper':>8} {'width':>8}  {'lower':>8} {'upper':>8} {'width':>8}")
    print("-" * 95)

    boot_mae  = bootstrap_df[bootstrap_df["metric"] == "mae"].set_index("model")
    fold_mae  = fold_ci_df[fold_ci_df["metric"] == "mae"].set_index("model")
    all_models = boot_mae.index.union(fold_mae.index)

    for key in all_models:
        name = (boot_mae.loc[key, "model_name"] if key in boot_mae.index
                else fold_mae.loc[key, "model_name"])
        b = boot_mae.loc[key] if key in boot_mae.index else None
        f = fold_mae.loc[key] if key in fold_mae.index else None

        b_str = (f"{b['ci_lower']:>8.6f} {b['ci_upper']:>8.6f} {b['ci_width']:>8.6f}"
                 if b is not None else f"{'—':>8} {'—':>8} {'—':>8}")
        f_str = (f"{f['ci_lower']:>8.6f} {f['ci_upper']:>8.6f} {f['ci_width']:>8.6f}"
                 if f is not None else f"{'—':>8} {'—':>8} {'—':>8}")

        print(f"  {name:<33} {b_str}  {f_str}")

    print("=" * 95 + "\n")


# ── 5-fold CV (for comparison / optimism-bias illustration) ──────────────────

def run_kfold_cv(df: pd.DataFrame, models: dict, n_folds: int = 5) -> pd.DataFrame:
    """
    Standard 5-fold cross-validation on the full feature matrix.

    WARNING — this is INCORRECT for time-series data.  It randomly mixes
    past and future rows across folds, so a model can effectively "train on
    the future."  The lag features (rate_lag1, rate_lag2, …) in the test fold
    were computed using values that appear in the training fold, inflating R².

    This function exists solely to QUANTIFY that optimism bias: the gap
    between k-fold metrics and walk-forward metrics shows exactly how much
    k-fold overstates real-world performance.  ARIMA/SARIMAX and Naive are
    excluded because they have no meaningful k-fold formulation.

    Returns a DataFrame with one row per (ML) model.
    """
    from sklearn.model_selection import KFold

    feature_cols = [f for f in config.FEATURE_COLS if f in df.columns]
    mask = df[feature_cols].notna().all(axis=1) & df[config.TARGET_COL].notna()
    df_clean = df[mask].copy()

    X = df_clean[feature_cols].values
    y = df_clean[config.TARGET_COL].clip(lower=0).values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Only ML models — skip naive/arima/sarimax
    ml_models = {k: v for k, v in models.items() if k not in ("naive", "arima", "sarimax")}

    accum = {k: {"true": [], "pred": []} for k in ml_models}

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        for key, (name, model_obj) in ml_models.items():
            try:
                model_obj.fit(X_tr, y_tr)
                preds = np.clip(model_obj.predict(X_te), 0, None)
                accum[key]["true"].extend(y_te.tolist())
                accum[key]["pred"].extend(preds.tolist())
            except Exception as e:
                logger.warning(f"  [kfold/{key}/fold{fold_i}] failed: {e}")

    rows = []
    for key, (name, _) in ml_models.items():
        data = accum[key]
        if not data["true"]:
            continue
        m = calc_metrics(np.array(data["true"]), np.array(data["pred"]))
        m["n_predictions"] = len(data["true"])
        rows.append({"model": key, "model_name": name, **m})

    return pd.DataFrame(rows)


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(overall_df: pd.DataFrame, title: str = "Walk-Forward Validation") -> None:
    """Print a formatted comparison table."""
    cols = ["model_name", "r2", "mae", "rmse", "smape", "n_predictions"]
    df = overall_df[[c for c in cols if c in overall_df.columns]].copy()
    df = df.sort_values("mae")

    print("\n" + "=" * 80)
    print(f"  MODEL COMPARISON — {title}")
    print("=" * 80)
    header = f"{'Model':<35} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'sMAPE':>8} {'N':>7}"
    print(header)
    print("-" * 80)
    for _, row in df.iterrows():
        print(
            f"  {row['model_name']:<33} "
            f"{row['r2']:>+8.4f} "
            f"{row['mae']:>10.6f} "
            f"{row['rmse']:>10.6f} "
            f"{row['smape']:>7.2f}% "
            f"{int(row['n_predictions']):>7,}"
        )
    print("=" * 80 + "\n")


def print_bias_table(wf_df: pd.DataFrame, cv_df: pd.DataFrame) -> None:
    """
    Print a side-by-side table showing the optimism bias of k-fold vs walk-forward.
    Only includes ML models present in both results.
    """
    wf = wf_df[~wf_df["model"].isin(["naive", "arima", "sarimax"])].set_index("model")
    cv = cv_df.set_index("model")
    common = wf.index.intersection(cv.index)

    if common.empty:
        return

    print("\n" + "=" * 90)
    print("  OPTIMISM BIAS: k-fold CV vs Walk-Forward  (positive = k-fold overstates performance)")
    print("=" * 90)
    print(f"  {'Model':<33} {'WF R²':>8} {'CV R²':>8} {'ΔR²':>8}  "
          f"{'WF MAE':>10} {'CV MAE':>10} {'ΔMAE':>10}")
    print("-" * 90)
    for key in common:
        wf_r2  = wf.loc[key, "r2"];   cv_r2  = cv.loc[key, "r2"]
        wf_mae = wf.loc[key, "mae"];  cv_mae = cv.loc[key, "mae"]
        name   = wf.loc[key, "model_name"]
        print(
            f"  {name:<33} "
            f"{wf_r2:>+8.4f} {cv_r2:>+8.4f} {cv_r2 - wf_r2:>+8.4f}  "
            f"{wf_mae:>10.6f} {cv_mae:>10.6f} {wf_mae - cv_mae:>+10.6f}"
        )
    print("=" * 90)
    print("  ΔR²  > 0 means k-fold overstates R² (expected for time-series data)")
    print("  ΔMAE > 0 means k-fold understates MAE (expected for time-series data)")
    print("=" * 90 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark all models under walk-forward CV")
    p.add_argument("--no-arima",  action="store_true",
                   help="Skip ARIMA and SARIMAX (much faster)")
    p.add_argument("--no-cv",     action="store_true",
                   help="Skip 5-fold CV (run walk-forward only)")
    p.add_argument("--no-uncertainty", action="store_true",
                   help="Skip uncertainty estimation (bootstrap + fold CIs)")
    p.add_argument("--models",    type=str, default=None,
                   help="Comma-separated model keys to run, e.g. 'naive,lr,xgb_tuned'")
    p.add_argument("--data",      type=str, default=None,
                   help="Path to features CSV (default: outputs/data/features.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    config.ensure_output_dirs()

    data_path = args.data or config.FEATURES_CSV
    if not os.path.exists(data_path):
        # Fall back to training_data.csv (base features only)
        data_path = config.TRAINING_DATA_CSV
        logger.warning(
            f"  features.csv not found; falling back to training_data.csv "
            f"(base features only — engineered features unavailable)"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No data file found at {data_path}. Run the pipeline (stages 2 + 25) first."
        )

    logger.info(f"  Loading: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"  Shape: {df.shape}")

    include_arima = (not args.no_arima) and HAS_STATSMODELS
    models = get_models(include_arima=include_arima)

    if args.models:
        requested = {k.strip() for k in args.models.split(",")}
        models = {k: v for k, v in models.items() if k in requested}
        unknown = requested - set(models)
        if unknown:
            logger.warning(f"  Unknown model keys (ignored): {unknown}")

    logger.info(f"  Models to benchmark: {[v[0] for v in models.values()]}")

    logger.info("\n" + "=" * 60)
    logger.info("  BENCHMARK: WALK-FORWARD VALIDATION")
    logger.info("=" * 60)

    overall_df, per_month_df, accum = run_benchmark(df, models)

    if overall_df.empty:
        logger.error("  No results produced — check that features.csv has sufficient data")
        return

    # Save outputs
    out_overall    = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_comparison.csv")
    out_per_month  = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_per_month.csv")
    out_json       = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_comparison.json")

    overall_df.to_csv(out_overall, index=False)
    per_month_df.to_csv(out_per_month, index=False)

    # JSON for paper / stage 6
    summary = overall_df.set_index("model").to_dict(orient="index")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Results → {out_overall}")
    logger.info(f"  Per-month → {out_per_month}")
    logger.info(f"  JSON → {out_json}")

    print_table(overall_df, title="Walk-Forward Validation (time-series correct)")

    # ── Uncertainty estimates ─────────────────────────────────────────────────
    if not args.no_uncertainty:
        logger.info("\n" + "=" * 60)
        logger.info("  UNCERTAINTY ESTIMATION")
        logger.info("=" * 60)

        # scipy needed for fold CI t-distribution
        try:
            bootstrap_df, fold_ci_df, county_df = compute_all_uncertainty(
                accum, models, per_month_df
            )

            out_boot    = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_bootstrap_ci.csv")
            out_fold_ci = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_fold_ci.csv")
            out_county  = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_county_residuals.csv")

            bootstrap_df.to_csv(out_boot, index=False)
            fold_ci_df.to_csv(out_fold_ci, index=False)
            county_df.to_csv(out_county, index=False)

            logger.info(f"  Bootstrap CIs → {out_boot}")
            logger.info(f"  Fold CIs      → {out_fold_ci}")
            logger.info(f"  County resids → {out_county}")

            print_uncertainty_table(bootstrap_df, fold_ci_df)

            # Top-10 hardest counties by MAE (tuned XGBoost)
            xgb_county = county_df[county_df["model"] == "xgb_tuned"].sort_values("mae", ascending=False)
            if not xgb_county.empty:
                print("  Hardest counties to predict (XGBoost tuned, by MAE):")
                print(f"  {'County':<20} {'MAE':>10} {'Bias':>10} {'Std':>10} {'N':>4}")
                print("  " + "-" * 56)
                for _, row in xgb_county.head(10).iterrows():
                    print(f"  {row['county']:<20} {row['mae']:>10.6f} {row['bias']:>+10.6f} "
                          f"{row['std']:>10.6f} {int(row['n']):>4}")
                print()

        except ImportError:
            logger.warning("  scipy not installed — fold CIs skipped. "
                           "Install with: pip install scipy")
            bootstrap_df, _, county_df = compute_all_uncertainty(
                accum, models, per_month_df
            )
            bootstrap_df.to_csv(
                os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_bootstrap_ci.csv"),
                index=False
            )

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    if not args.no_cv:
        logger.info("\n" + "=" * 60)
        logger.info("  5-FOLD CV  ⚠️  INCORRECT FOR TIME-SERIES — FOR COMPARISON ONLY")
        logger.info("=" * 60)
        cv_df = run_kfold_cv(df, models)

        if not cv_df.empty:
            out_cv = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_kfold_cv.csv")
            cv_df.to_csv(out_cv, index=False)
            logger.info(f"  k-fold results → {out_cv}")

            print_table(cv_df, title="5-Fold CV  ⚠️  time-series-INCORRECT — optimism bias expected")
            print_bias_table(overall_df, cv_df)


if __name__ == "__main__":
    main()
