"""
stage5_predict.py — Generate forward predictions for all counties.

For each county, this stage:
  1. Identifies the target prediction month from the latest prediction CSVs
  2. Scales current-month Google Trends to the training reference frame
  3. Assembles the feature vector (trends + population + income + month)
  4. Runs the model and converts the predicted rate to application count
  5. Builds a confidence interval using walkforward_mae from the model bundle
  6. Assigns a risk flag based on z-score vs historical SNAP rates

Scaling formula (per DMA, per keyword):
    scaled = latest_month_avg × (train_avg / pred_window_avg)
This preserves spike signals — if this month's searches are elevated relative
to the prediction window, the scaled value is proportionally above the training
average the model learned from.

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
from pipeline.stage1_load_raw import load_prediction_trends, load_population, load_income, load_county_metro

logger = logging.getLogger(__name__)

# Risk flag thresholds (z-score of predicted SNAP rate vs county historical mean)
FLAG_RED    =  1.0   # z-score ≥ 1.0 → significantly elevated
FLAG_YELLOW =  0.5   # z-score ≥ 0.5 → moderately elevated
FLAG_GREEN  = -0.5   # z-score > -0.5 → near or below historical average


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
        f"walkforward_mae: {bundle.get('walkforward_mae','?'):.6f}"
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

    assert isinstance(params, dict) and len(params) > 0, (
        f"Scaling params file is empty or malformed: {config.SCALING_PARAMS_JSON}"
    )
    for kw in config.KEYWORDS:
        assert kw in params, (
            f"Missing keyword '{kw}' in scaling params — re-run stage 2"
        )
        assert len(params[kw]) > 0, (
            f"No DMA entries for keyword '{kw}' in scaling params"
        )
    logger.info(
        f"  Loaded scaling params: "
        + ", ".join(f"{kw} ({len(params[kw])} DMAs)" for kw in params)
    )
    return params


# ── Target month detection ────────────────────────────────────────────────────

def detect_target_month() -> pd.Timestamp:
    """
    Determine the prediction target month by reading the latest date in the
    Bakersfield prediction CSV (used as a reference DMA).

    The target month is the month of the most recent data point — we are
    predicting applications *for* that month using trends from that month,
    since trends are uploaded for the current month before SNAP data arrives.
    Falls back to current month if no prediction data is available.
    """
    for kw in config.KEYWORDS:
        sample = os.path.join(config.PREDICTION_DIR, kw, "Bakersfield.csv")
        if os.path.exists(sample):
            from pipeline.stage1_load_raw import _read_prediction_csv
            df = _read_prediction_csv(sample)
            if not df.empty:
                last_date = df["date"].max()
                target = last_date.replace(day=1)
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

    Why this works:
    - The model was trained on trends in the 0-100 absolute scale
    - Prediction CSVs only cover a short window (e.g. 3 months), so their
      absolute level may differ from the training period's absolute level
    - This ratio correction aligns the two windows

    Edge case: if pred_window_avg == 0 (all-zero prediction window, meaning
    no search activity was recorded), we substitute train_avg directly as a
    neutral 'no signal' value rather than producing 0 or NaN.

    Returns scaled float, or None if training params are missing for this DMA.
    """
    train_avg = scaling_params.get(keyword, {}).get(metro_area)
    if train_avg is None:
        logger.warning(f"  No training avg for {metro_area}/{keyword}")
        return None

    # Monthly-aggregate the prediction window
    pred_df = pred_df.copy()
    pred_df["ym"] = pred_df["date"].dt.to_period("M")
    monthly = pred_df.groupby("ym")["value"].mean()

    if monthly.empty:
        return None

    latest_month_avg = float(monthly.iloc[-1])
    pred_window_avg  = float(monthly.mean())

    if pred_window_avg == 0:
        logger.info(f"  {metro_area}/{keyword}: all-zero window → using train_avg={train_avg:.2f}")
        return float(train_avg)

    scale_factor = train_avg / pred_window_avg
    scaled = latest_month_avg * scale_factor

    # ── Assertions ────────────────────────────────────────────────────────────
    assert scaled >= 0, (
        f"Negative scaled trend for {metro_area}/{keyword}: "
        f"latest={latest_month_avg:.2f}, factor={scale_factor:.4f} → {scaled:.4f}"
    )
    assert train_avg > 0, (
        f"Training average is zero for {metro_area}/{keyword} — "
        f"check that training trend data was loaded correctly"
    )
    if scaled > 500:
        logger.warning(
            f"  {metro_area}/{keyword}: scaled={scaled:.1f} > 500 — "
            f"unusually large; latest={latest_month_avg:.1f}, "
            f"pred_avg={pred_window_avg:.1f}, train_avg={train_avg:.2f}, factor={scale_factor:.4f}"
        )

    logger.debug(
        f"  {metro_area}/{keyword}: latest={latest_month_avg:.1f}, "
        f"pred_avg={pred_window_avg:.1f}, train_avg={train_avg:.2f}, "
        f"factor={scale_factor:.4f}, scaled={scaled:.2f}"
    )
    return scaled


# ── Feature assembly ──────────────────────────────────────────────────────────

def get_features_for_county(
    county: str,
    metro_area: str,
    target_month: pd.Timestamp,
    prediction_trends: dict,
    scaling_params: dict,
    pop_df: pd.DataFrame,
    income_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Build the feature vector for one county for the target month.

    Returns a dict matching config.FEATURE_COLS, or None if required data
    is missing (county is skipped from predictions with a warning).
    """
    features = {}

    # Population
    pop_row = pop_df[pop_df["county"] == county]
    if pop_row.empty or pd.isna(pop_row["Population"].values[0]):
        logger.warning(f"  No population for {county} — skipping")
        return None
    features["Population"] = float(pop_row["Population"].values[0])

    # Income
    county_key = county.replace(" ", "")
    income_df["county_key"] = income_df["county_key"] if "county_key" in income_df.columns else income_df.index
    inc_match = income_df[income_df["county_key"] == county_key]
    if inc_match.empty or pd.isna(inc_match["Median_Income"].values[0]):
        logger.warning(f"  No income for {county} — using national fallback $60,000")
        features["Median_Income"] = 60000.0
    else:
        features["Median_Income"] = float(inc_match["Median_Income"].values[0])

    # Month feature
    features["month"] = int(target_month.month)

    # Trend features — scale prediction window to training reference frame
    for kw in config.KEYWORDS:
        col = f"monthly_average_{kw}"
        pred_df = prediction_trends.get(kw, {}).get(metro_area)
        if pred_df is None or pred_df.empty:
            logger.warning(f"  No prediction trends for {metro_area}/{kw} — skipping {county}")
            return None
        scaled = scale_prediction_trends(metro_area, kw, pred_df, scaling_params)
        if scaled is None:
            return None
        features[col] = scaled

    return features


# ── Risk flagging ─────────────────────────────────────────────────────────────

def assign_risk_flag(predicted_rate: float, county: str, historical_df: pd.DataFrame) -> str:
    """
    Assign a Green/Yellow/Red risk flag based on z-score.

    Z-score measures how many standard deviations the predicted rate is above
    the county's historical mean SNAP application rate. Higher z-score means
    more unusual / higher demand relative to that county's baseline.

    Returns 'Gray' if there is insufficient historical data for the county.
    """
    county_hist = historical_df[historical_df["county"] == county][config.TARGET_COL].dropna()
    if len(county_hist) < 3:
        return "Gray"

    mean, std = county_hist.mean(), county_hist.std()
    if std == 0 or pd.isna(std):
        return "Gray"

    z = (predicted_rate - mean) / std
    if z >= FLAG_RED:
        return "Red"
    elif z >= FLAG_YELLOW:
        return "Yellow"
    else:
        return "Green"


# ── Main prediction loop ──────────────────────────────────────────────────────

def predict_all_counties(target_month: pd.Timestamp, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for every county that has all required data.

    Confidence intervals use the walk-forward MAE embedded in the model bundle:
        lower = max(0, predicted_rate - walkforward_mae)
        upper = predicted_rate + walkforward_mae
    Then multiply by population to get application-count bounds.
    """
    bundle         = load_model_bundle()
    scaling_params = load_scaling_params()
    model          = bundle["model"]
    feature_cols   = bundle["features"]
    wf_mae         = bundle.get("walkforward_mae", 0.000877)

    pop_df    = load_population()
    income_df = load_income()
    cm_df     = load_county_metro()

    # Load all prediction trend data into a nested dict: {keyword: {metro: df}}
    prediction_trends = {}
    for kw in config.KEYWORDS:
        kw_data = {}
        trends_df = load_prediction_trends(kw)
        if not trends_df.empty:
            for metro, grp in trends_df.groupby("metro_area"):
                kw_data[metro] = grp[["date", "value"]].reset_index(drop=True)
        prediction_trends[kw] = kw_data

    # Build predictions for each county
    rows = []
    counties = cm_df["county"].unique()
    skipped = 0

    for county in sorted(counties):
        metro_row = cm_df[cm_df["county"] == county]
        if metro_row.empty:
            skipped += 1
            continue
        metro_area = metro_row["metro_area"].values[0]

        features = get_features_for_county(
            county, metro_area, target_month,
            prediction_trends, scaling_params,
            pop_df, income_df,
        )
        if features is None:
            skipped += 1
            continue

        # Build feature row in the exact order the model expects
        X_row = pd.DataFrame([{f: features.get(f, np.nan) for f in feature_cols}])
        if X_row.isna().any(axis=1).values[0]:
            logger.warning(f"  NaN in feature vector for {county} — skipping")
            skipped += 1
            continue

        predicted_rate = float(np.clip(model.predict(X_row)[0], 0, None))
        population     = features["Population"]
        predicted_apps = round(predicted_rate * population)

        lower_rate = max(0.0, predicted_rate - wf_mae)
        upper_rate = predicted_rate + wf_mae
        lower_apps = round(lower_rate * population)
        upper_apps = round(upper_rate * population)

        flag = assign_risk_flag(predicted_rate, county, historical_df)

        rows.append({
            "date":                   target_month.strftime("%Y-%m-%d"),
            "county":                 county,
            "metro_area":             metro_area,
            "predicted_rate":         round(predicted_rate, 6),
            "predicted_applications": int(predicted_apps),
            "lower_bound":            int(lower_apps),
            "upper_bound":            int(upper_apps),
            "Population":             int(population),
            "flag":                   flag,
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

    historical_df = pd.read_csv(config.TRAINING_DATA_CSV)
    target_month  = detect_target_month()

    predictions_df = predict_all_counties(target_month, historical_df)

    if predictions_df.empty:
        logger.error("  No predictions generated — check prediction data and model files")
        return predictions_df

    predictions_df.to_csv(config.PREDICTIONS_CSV, index=False)
    logger.info(f"  Predictions → {config.PREDICTIONS_CSV}")

    # Flag summary
    flag_counts = predictions_df["flag"].value_counts()
    logger.info(f"  Risk flag distribution: {flag_counts.to_dict()}")

    return predictions_df
