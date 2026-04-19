"""
stage5_predict.py — Generate forward predictions for all counties.

For each county, this stage:
  1. Identifies the target prediction month from the latest prediction CSVs
  2. Scales current-month Google Trends to the training reference frame
  3. Assembles the full 24-feature vector from prediction trends + historical data
  4. Runs the model and converts the predicted rate to application count
  5. Builds a confidence interval using walkforward_mae from the model bundle
  6. Runs the early-warning alert layer (alert_layer.py) to compute warning signals

Feature assembly strategy:
  The model is trained on 24 features including lags and rolling windows.
  At prediction time the upcoming month's SNAP data doesn't exist yet, so lag
  features are computed from the most recent rows in features.csv:

    rate_lag1         ← most recent known SNAP_Application_Rate (last row in features.csv)
    rate_lag2         ← last_row["rate_lag1"]  (shifting one step back)
    rate_lag3         ← last_row["rate_lag2"]
    rate_roll3_mean   ← mean of last 3 known rates
    rate_roll3_std    ← std  of last 3 known rates
    calfresh_lag1     ← last_row["monthly_average_CalFresh"]
    calfresh_lag2     ← last_row["calfresh_lag1"]
    calfresh_roll3    ← rolling mean of last 3 known CalFresh values
    calfresh_momentum ← scaled_current_CalFresh − calfresh_lag1
    foodbank_lag*     ← same pattern as CalFresh

  If a county is absent from features.csv (dropped during feature engineering due
  to FoodBank NaN coverage), it is skipped with a warning.
  If a DMA has no FoodBank scaling params (SanFranciscoOaklandSanJose), the last
  known historical FoodBank value from features.csv is used instead.

Scaling formula (per DMA, per keyword):
    scaled = latest_month_avg × (train_avg / pred_window_avg)

Outputs:
  outputs/predictions/finalPrediction.csv
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.alert_layer import compute_warning_signals
from pipeline.stage1_load_raw import (
    load_prediction_trends, load_population, load_income, load_county_metro
)

logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_bundle() -> dict:
    """Load the serialized model bundle from outputs/models/."""
    if not os.path.exists(config.MODEL_PKL):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PKL}. Run stage 3 first."
        )
    with open(config.MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    logger.info(
        f"  Loaded model: {bundle.get('type','unknown')}, "
        f"features: {bundle['features']}, "
        f"walkforward_mae: {bundle.get('walkforward_mae', '?'):.6f}"
    )
    return bundle


def load_scaling_params() -> dict:
    """Load per-DMA training averages saved by stage 2."""
    if not os.path.exists(config.SCALING_PARAMS_JSON):
        raise FileNotFoundError(
            f"Scaling params not found at {config.SCALING_PARAMS_JSON}. Run stage 2 first."
        )
    with open(config.SCALING_PARAMS_JSON) as f:
        params = json.load(f)
    logger.info(
        "  Loaded scaling params: "
        + ", ".join(f"{kw} ({len(params[kw])} DMAs)" for kw in params)
    )
    return params


# ── Target month detection ────────────────────────────────────────────────────

def detect_target_month() -> pd.Timestamp:
    """
    Determine the prediction target month from the latest date in the
    Bakersfield prediction CSV.  Falls back to current month.
    """
    for kw in config.KEYWORDS:
        sample = os.path.join(config.PREDICTION_DIR, kw, "Bakersfield.csv")
        if os.path.exists(sample):
            from pipeline.stage1_load_raw import _read_prediction_csv
            df = _read_prediction_csv(sample)
            if not df.empty:
                target = df["date"].max().replace(day=1)
                logger.info(f"  Detected prediction month: {target.strftime('%B %Y')}")
                return target

    target = pd.Timestamp(datetime.now().replace(day=1))
    logger.warning(f"  Could not detect prediction month; using current: {target.strftime('%B %Y')}")
    return target


# ── Trend scaling ─────────────────────────────────────────────────────────────

def scale_prediction_trends(
    metro_area: str,
    keyword: str,
    pred_df: pd.DataFrame,
    scaling_params: dict,
) -> Optional[float]:
    """
    Scale the latest month's Google Trends value to the training reference frame.

    Formula: scaled = latest_month_avg × (train_avg / pred_window_avg)

    If pred_window_avg == 0 (no search activity), substitute train_avg directly.
    Returns None if training params are missing for this DMA/keyword.
    """
    train_avg = scaling_params.get(keyword, {}).get(metro_area)
    if train_avg is None:
        return None

    pred_df = pred_df.copy()
    pred_df["ym"] = pred_df["date"].dt.to_period("M")
    monthly = pred_df.groupby("ym")["value"].mean()

    if monthly.empty:
        return None

    latest_month_avg = float(monthly.iloc[-1])
    pred_window_avg  = float(monthly.mean())

    if pred_window_avg == 0:
        return float(train_avg)

    scaled = latest_month_avg * (train_avg / pred_window_avg)
    assert scaled >= 0, f"Negative scaled trend for {metro_area}/{keyword}: {scaled}"
    return scaled


# ── Historical feature lookup ─────────────────────────────────────────────────

def _get_county_history(county: str, features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Return the county's rows from features.csv, sorted by date.
    Returns None if the county is absent (was dropped during feature engineering).
    """
    hist = features_df[features_df["county"] == county].sort_values("date")
    if hist.empty:
        return None
    return hist


def _last(series: pd.Series, default=np.nan):
    """Return last non-NaN value in series, or default."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else default


# ── Feature assembly ──────────────────────────────────────────────────────────

def get_features_for_county(
    county: str,
    metro_area: str,
    target_month: pd.Timestamp,
    prediction_trends: dict,
    scaling_params: dict,
    pop_df: pd.DataFrame,
    income_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Build the full 24-feature vector for one county for the target month.

    Uses scaled prediction-window Trends for current-month values, and
    features.csv for all lag/rolling/momentum features (shifted one period
    forward so the most recent known month becomes lag-1).

    Returns a dict matching config.FEATURE_COLS, or None if required data
    is missing.
    """
    features = {}

    # ── Population ────────────────────────────────────────────────────────────
    pop_row = pop_df[pop_df["county"] == county]
    if pop_row.empty or pd.isna(pop_row["Population"].values[0]):
        logger.warning(f"  No population for {county} — skipping")
        return None
    population = float(pop_row["Population"].values[0])
    features["Population"] = population

    # ── Income ────────────────────────────────────────────────────────────────
    county_key = county.replace(" ", "")
    inc_match = income_df[income_df["county_key"] == county_key]
    if inc_match.empty or pd.isna(inc_match["Median_Income"].values[0]):
        logger.warning(f"  No income for {county} — using fallback $60,000")
        median_income = 60000.0
    else:
        median_income = float(inc_match["Median_Income"].values[0])
    features["Median_Income"] = median_income

    # ── Historical features from features.csv ────────────────────────────────
    hist = _get_county_history(county, features_df)
    if hist is None:
        logger.warning(f"  {county} absent from features.csv (dropped during engineering) — skipping")
        return None
    last = hist.iloc[-1]

    # SNAP rate lags: shift forward one step
    # last row = December 2025 row → its SNAP rate IS our rate_lag1 for Feb 2026
    features["rate_lag1"]       = _last(hist["SNAP_Application_Rate"])
    features["rate_lag2"]       = float(last["rate_lag1"])  if not pd.isna(last.get("rate_lag1", np.nan)) else np.nan
    features["rate_lag3"]       = float(last["rate_lag2"])  if not pd.isna(last.get("rate_lag2", np.nan)) else np.nan
    recent_rates = hist["SNAP_Application_Rate"].tail(3).dropna().values
    features["rate_roll3_mean"] = float(np.mean(recent_rates)) if len(recent_rates) >= 2 else features["rate_lag1"]
    features["rate_roll3_std"]  = float(np.std(recent_rates))  if len(recent_rates) >= 2 else 0.0

    # CalFresh lags: shift forward one step
    features["calfresh_lag1"]   = _last(hist["monthly_average_CalFresh"])
    features["calfresh_lag2"]   = float(last["calfresh_lag1"]) if not pd.isna(last.get("calfresh_lag1", np.nan)) else np.nan
    recent_cf = hist["monthly_average_CalFresh"].tail(3).dropna().values
    features["calfresh_roll3"]  = float(np.mean(recent_cf)) if len(recent_cf) > 0 else np.nan

    # FoodBank lags: shift forward one step
    features["foodbank_lag1"]   = _last(hist["monthly_average_FoodBank"])
    features["foodbank_lag2"]   = float(last["foodbank_lag1"]) if not pd.isna(last.get("foodbank_lag1", np.nan)) else np.nan
    recent_fb = hist["monthly_average_FoodBank"].tail(3).dropna().values
    features["foodbank_roll3"]  = float(np.mean(recent_fb)) if len(recent_fb) > 0 else np.nan

    # ── Current-month Trends (scaled from prediction folder) ──────────────────
    for kw in config.KEYWORDS:
        col = f"monthly_average_{kw}"
        pred_df = prediction_trends.get(kw, {}).get(metro_area)

        if pred_df is None or pred_df.empty:
            # Fallback: use most recent historical value for this DMA
            hist_val = _last(hist[col]) if col in hist.columns else np.nan
            if pd.isna(hist_val):
                logger.warning(f"  No prediction trends and no history for {metro_area}/{kw} — skipping {county}")
                return None
            logger.info(f"  {county}: no prediction trends for {kw} — using last known historical value {hist_val:.1f}")
            features[col] = hist_val
            continue

        scaled = scale_prediction_trends(metro_area, kw, pred_df, scaling_params)
        if scaled is None:
            # Fallback: scaling params missing (e.g. SanFrancisco/FoodBank) → use last known
            hist_val = _last(hist[col]) if col in hist.columns else np.nan
            if pd.isna(hist_val):
                logger.warning(f"  No scaling params and no history for {metro_area}/{kw} — skipping {county}")
                return None
            logger.info(f"  {county}: no scaling params for {kw} — using last known historical value {hist_val:.1f}")
            features[col] = hist_val
        else:
            features[col] = scaled

    # ── Momentum (current prediction vs last known) ───────────────────────────
    features["calfresh_momentum"] = features["monthly_average_CalFresh"] - features.get("calfresh_lag1", features["monthly_average_CalFresh"])
    features["foodbank_momentum"] = features["monthly_average_FoodBank"] - features.get("foodbank_lag1", features["monthly_average_FoodBank"])

    # ── Seasonality ───────────────────────────────────────────────────────────
    m = int(target_month.month)
    features["month"]     = m
    features["month_sin"] = float(np.sin(2 * np.pi * m / 12))
    features["month_cos"] = float(np.cos(2 * np.pi * m / 12))
    features["quarter"]   = int((m - 1) // 3 + 1)

    # ── Log transforms ────────────────────────────────────────────────────────
    features["log_population"] = float(np.log10(max(1, population)))
    features["log_income"]     = float(np.log10(max(1, median_income)))
    # income_quintile removed from FEATURE_COLS (zero XGBoost importance)

    return features



# ── Main prediction loop ──────────────────────────────────────────────────────

def predict_all_counties(target_month: pd.Timestamp) -> pd.DataFrame:
    """
    Generate predictions for every county that has all required data.
    """
    bundle         = load_model_bundle()
    scaling_params = load_scaling_params()
    model          = bundle["model"]
    feature_cols   = bundle["features"]
    wf_mae         = bundle.get("walkforward_mae", 0.000877)

    pop_df    = load_population()
    income_df = load_income()
    cm_df     = load_county_metro()

    # Load the engineered feature history for lag lookups
    if not os.path.exists(config.FEATURES_CSV):
        raise FileNotFoundError(
            f"features.csv not found at {config.FEATURES_CSV}. Run feature engineering first."
        )
    features_df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    logger.info(
        f"  Historical features: {features_df['county'].nunique()} counties, "
        f"up to {features_df['date'].max().date()}"
    )

    # Load prediction trend data: {keyword: {metro: df}}
    prediction_trends = {}
    for kw in config.KEYWORDS:
        kw_data = {}
        trends_df = load_prediction_trends(kw)
        if not trends_df.empty:
            for metro, grp in trends_df.groupby("metro_area"):
                kw_data[metro] = grp[["date", "value"]].reset_index(drop=True)
        prediction_trends[kw] = kw_data

    rows = []
    skipped = 0
    counties = cm_df["county"].unique()

    for county in sorted(counties):
        metro_row = cm_df[cm_df["county"] == county]
        if metro_row.empty:
            skipped += 1
            continue
        metro_area = metro_row["metro_area"].values[0]

        features = get_features_for_county(
            county, metro_area, target_month,
            prediction_trends, scaling_params,
            pop_df, income_df, features_df,
        )
        if features is None:
            skipped += 1
            continue

        # Build feature row in exact order the model expects
        X_row = pd.DataFrame([{f: features.get(f, np.nan) for f in feature_cols}])
        nan_cols = [c for c in feature_cols if pd.isna(X_row[c].values[0])]
        if nan_cols:
            logger.warning(f"  NaN in features for {county}: {nan_cols} — skipping")
            skipped += 1
            continue

        predicted_rate = float(np.clip(model.predict(X_row)[0], 0, None))
        population     = features["Population"]
        predicted_apps = round(predicted_rate * population)

        lower_rate = max(0.0, predicted_rate - wf_mae)
        upper_rate = predicted_rate + wf_mae
        lower_apps = round(lower_rate * population)
        upper_apps = round(upper_rate * population)

        # ── Early-warning alert layer ─────────────────────────────────────────
        # Pull this county's historical series from features_df (already loaded)
        hist = features_df[features_df["county"] == county].sort_values("date")
        signals = compute_warning_signals(
            county            = county,
            predicted_rate    = predicted_rate,
            hist_rate_series  = hist["SNAP_Application_Rate"],
            scaled_calfresh   = features.get("monthly_average_CalFresh"),
            scaled_foodbank   = features.get("monthly_average_FoodBank"),
            calfresh_hist_series = hist["monthly_average_CalFresh"],
            foodbank_hist_series = hist["monthly_average_FoodBank"],
        )

        rows.append({
            "date":                      target_month.strftime("%Y-%m-%d"),
            "county":                    county,
            "metro_area":                metro_area,
            "predicted_rate":            round(predicted_rate, 6),
            "predicted_applications":    int(predicted_apps),
            "lower_bound":               int(lower_apps),
            "upper_bound":               int(upper_apps),
            "Population":                int(population),
            # Alert layer fields
            "rolling_mean_rate":         signals["rolling_mean_rate"],
            "rolling_std_rate":          signals["rolling_std_rate"],
            "prediction_zscore_recent":  signals["prediction_zscore_recent"],
            "calfresh_trend_zscore":     signals["calfresh_trend_zscore"],
            "foodbank_trend_zscore":     signals["foodbank_trend_zscore"],
            "combined_trend_anomaly":    signals["combined_trend_anomaly"],
            "warning_score":             signals["warning_score"],
            "warning_flag":              signals["warning_flag"],
        })

    predictions_df = pd.DataFrame(rows)
    logger.info(
        f"  Predicted {len(predictions_df)} counties "
        f"({skipped} skipped) for {target_month.strftime('%B %Y')}"
    )
    return predictions_df


# ── Main entry point ──────────────────────────────────────────────────────────

def predict() -> pd.DataFrame:
    """Full prediction stage. Returns the predictions DataFrame."""
    logger.info("=== STAGE 5: PREDICT ===")

    target_month  = detect_target_month()

    predictions_df = predict_all_counties(target_month)

    if predictions_df.empty:
        logger.error("  No predictions generated — check prediction data and model files")
        return predictions_df

    os.makedirs(os.path.dirname(config.PREDICTIONS_CSV), exist_ok=True)
    predictions_df.to_csv(config.PREDICTIONS_CSV, index=False)
    logger.info(f"  Predictions → {config.PREDICTIONS_CSV}")

    flag_counts = predictions_df["warning_flag"].value_counts()
    logger.info(f"  Warning flag distribution: {flag_counts.to_dict()}")

    return predictions_df
