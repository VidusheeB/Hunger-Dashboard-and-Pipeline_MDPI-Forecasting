"""
stage5_predict.py — Generate forward predictions for all counties.

For each county, this stage:
  1. Identifies the target SNAP month as one month after the latest Trends CSV
  2. Scales the latest available Google Trends month to the training frame
  3. Assembles the deployable feature vector from prediction trends + historical data
  4. Runs the model and converts the predicted rate to application count
  5. Builds a confidence interval using walkforward_mae from the model bundle

Feature assembly strategy:
  The model is trained on deployable features only: demographics, Google Trends
  lags/rolling/momentum, BLS unemployment, and seasonality.  SNAP-derived lag
  and rolling features are intentionally not assembled or used.

    calfresh_lag1     ← scaled CalFresh Trends from t-1
    calfresh_lag2     ← scaled/historical CalFresh Trends from t-2
    calfresh_roll3    ← mean CalFresh Trends from t-1, t-2, and t-3
    calfresh_momentum ← calfresh_lag1 − calfresh_lag2
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
    Determine the SNAP target month from the latest date in the Bakersfield
    prediction CSV. If the latest Trends month is February, the target SNAP
    month is March, so every model feature is from t-1 or earlier.
    """
    for kw in config.KEYWORDS:
        sample = os.path.join(config.PREDICTION_DIR, kw, "Bakersfield.csv")
        if os.path.exists(sample):
            from pipeline.stage1_load_raw import _read_prediction_csv
            df = _read_prediction_csv(sample)
            if not df.empty:
                latest_trends = df["date"].max().replace(day=1)
                target = latest_trends + pd.DateOffset(months=1)
                logger.info(
                    f"  Latest prediction Trends month: {latest_trends.strftime('%B %Y')} "
                    f"→ target SNAP month: {target.strftime('%B %Y')}"
                )
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


def scale_prediction_trend_series(
    metro_area: str,
    keyword: str,
    pred_df: pd.DataFrame,
    scaling_params: dict,
) -> Optional[pd.Series]:
    """
    Scale all available prediction-window monthly Trends values.

    The returned Series is indexed by actual Trends month. For target SNAP
    month t, the last value becomes lag1 (t-1), the previous value lag2 (t-2),
    and the last three values form the trailing roll3.
    """
    train_avg = scaling_params.get(keyword, {}).get(metro_area)
    if train_avg is None or pred_df is None or pred_df.empty:
        return None

    pred_df = pred_df.copy()
    pred_df["month"] = pred_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = pred_df.groupby("month")["value"].mean().sort_index()
    if monthly.empty:
        return None

    pred_window_avg = float(monthly.mean())
    if pred_window_avg == 0:
        return pd.Series(float(train_avg), index=monthly.index)

    scaled = monthly * (float(train_avg) / pred_window_avg)
    return scaled.clip(lower=0)


def load_laus_history() -> pd.DataFrame:
    """Load raw LAUS unemployment history for publication-safe Stage 5 lookup."""
    if not os.path.exists(config.LAUS_FILE):
        logger.warning(f"LAUS unemployment file not found: {config.LAUS_FILE}")
        return pd.DataFrame(columns=["county", "date", "unemployment_rate"])

    laus = pd.read_csv(config.LAUS_FILE, parse_dates=["date"])
    laus = laus[["county", "date", "unemployment_rate"]].copy()
    laus["date"] = laus["date"].dt.to_period("M").dt.to_timestamp()
    return laus.sort_values(["county", "date"]).reset_index(drop=True)


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
    laus_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Build the deployable feature vector for one county for the target month.

    Uses scaled prediction-window Trends through target month t-1, plus
    historical Trends from features.csv when prediction-window data is missing.
    SNAP-derived lag and rolling features are intentionally excluded.

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

    # Trend lags and rolling means. Stage 2 stores monthly_average_* on the
    # SNAP month row, but the value itself comes from the prior Trends month.
    latest_allowed_trend_month = target_month - pd.DateOffset(months=1)
    for prefix, keyword, col in [
        ("calfresh",   "CalFresh",    "monthly_average_CalFresh"),
        ("foodbank",   "FoodBank",    "monthly_average_FoodBank"),
        ("foodstamps", "FoodStamps",  "monthly_average_FoodStamps"),
        ("snaptopic",  "SNAPTopic",   "monthly_average_SNAPTopic"),
    ]:
        pred_df = prediction_trends.get(keyword, {}).get(metro_area)
        pred_series = scale_prediction_trend_series(metro_area, keyword, pred_df, scaling_params)

        historical_series = pd.Series(dtype=float)
        if col in hist.columns:
            historical = hist[["date", col]].dropna().copy()
            historical["trend_month"] = (
                pd.to_datetime(historical["date"]).dt.to_period("M").dt.to_timestamp()
                - pd.DateOffset(months=1)
            )
            historical_series = historical.set_index("trend_month")[col].astype(float)

        pieces = [s for s in [historical_series, pred_series] if s is not None and not s.empty]
        combined = pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=float)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined[combined.index <= latest_allowed_trend_month].dropna()

        if combined.empty:
            logger.warning(f"  No usable Trends history for {metro_area}/{keyword} — skipping {county}")
            return None

        recent = combined.tail(3)
        features[f"{prefix}_lag1"] = float(recent.iloc[-1])
        features[f"{prefix}_lag2"] = float(recent.iloc[-2]) if len(recent) >= 2 else np.nan
        features[f"{prefix}_roll3"] = float(recent.mean()) if len(recent) >= 2 else np.nan
        features[f"{prefix}_momentum"] = (
            features[f"{prefix}_lag1"] - features[f"{prefix}_lag2"]
            if not pd.isna(features[f"{prefix}_lag2"])
            else np.nan
        )

    # Preserve raw latest Trends values in case downstream diagnostics inspect them.
    for prefix, col in [
        ("calfresh",   "monthly_average_CalFresh"),
        ("foodbank",   "monthly_average_FoodBank"),
        ("foodstamps", "monthly_average_FoodStamps"),
        ("snaptopic",  "monthly_average_SNAPTopic"),
    ]:
        features[col] = features.get(f"{prefix}_lag1", np.nan)

    # ── Unemployment ─────────────────────────────────────────────────────────
    county_laus = laus_df[laus_df["county"] == county].sort_values("date")
    if county_laus.empty:
        features["unemployment_rate"] = _last(hist["unemployment_rate"]) if "unemployment_rate" in hist.columns else np.nan
        features["unemployment_rate_lag1"] = _last(hist["unemployment_rate_lag1"]) if "unemployment_rate_lag1" in hist.columns else np.nan
    else:
        u_t1 = county_laus[county_laus["date"] <= target_month - pd.DateOffset(months=1)]
        u_t2 = county_laus[county_laus["date"] <= target_month - pd.DateOffset(months=2)]
        features["unemployment_rate"] = _last(u_t1["unemployment_rate"])
        features["unemployment_rate_lag1"] = _last(u_t2["unemployment_rate"])

    # ── Seasonality ───────────────────────────────────────────────────────────
    m = int(target_month.month)
    features["month"]     = m
    features["month_sin"] = float(np.sin(2 * np.pi * m / 12))
    features["month_cos"] = float(np.cos(2 * np.pi * m / 12))
    features["quarter"]   = int((m - 1) // 3 + 1)

    # ── Log transforms ────────────────────────────────────────────────────────
    features["log_population"] = float(np.log10(max(1, population)))
    features["log_income"]     = float(np.log10(max(1, median_income)))

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
    laus_df = load_laus_history()
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
            pop_df, income_df, features_df, laus_df,
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

        rows.append({
            "date":                      target_month.strftime("%Y-%m-%d"),
            "county":                    county,
            "metro_area":                metro_area,
            "predicted_rate":            round(predicted_rate, 6),
            "predicted_applications":    int(predicted_apps),
            "lower_bound":               int(lower_apps),
            "upper_bound":               int(upper_apps),
            "Population":                int(population),
            "flag":                      "Gray",
            "warning_flag":              "Gray",
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
    logger.info(f"  Warning flag distribution: {flag_counts.to_dict()} (unscored)")

    return predictions_df
