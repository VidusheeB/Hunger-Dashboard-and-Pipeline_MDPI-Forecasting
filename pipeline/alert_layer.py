"""
alert_layer.py — Early-warning classification layer on top of the baseline prediction.

Purpose
-------
The baseline XGBoost model predicts smooth, conservative SNAP application rates
dominated by lagged features (rate_roll3_mean alone accounts for ~50% of importance).
Comparing predicted rates against the county's full historical mean/std produces
z-scores that are almost always near zero because the model rarely predicts far
from the historical average.

This module replaces that classification logic with signals that are sensitive to
*recent* deviations rather than long-run baselines:

  1. prediction_zscore_recent
       How much does the predicted rate deviate from the county's own recent rolling
       mean?  Uses a rolling window of W months (config.ALERT_ROLLING_WINDOW_W) rather
       than the full history.  Detects counties where the model is predicting above
       their own recent normal even if the county is not extreme statewide.

  2. trend_anomaly_score (per DMA, per keyword, then combined)
       How elevated are the current Trends signals relative to their own recent rolling
       baseline?  A CalFresh or FoodBank z-score that is high means search activity is
       unusual recently — a leading-indicator signal independent of the model output.

  3. warning_score (composite)
       warning_score = alpha * prediction_zscore_recent
                     + (1 - alpha) * combined_trend_anomaly
       Alpha and combination method are set in config.py.

  4. warning_flag
       Green / Yellow / Red / Gray based on configurable thresholds in config.py.
       All thresholds are documented and centralised — none are hidden in logic.

All configuration lives in config.py under the "Early-warning alert layer" section.
Tune thresholds by running:
    python experiments/evaluate_alerts.py

Public API
----------
    compute_warning_signals(
        county, predicted_rate, target_month, hist_rate_series,
        scaled_calfresh, scaled_foodbank,
        calfresh_hist_series, foodbank_hist_series,
    ) -> dict

Returns a dict of all signal components plus warning_score and warning_flag.
Designed to be called once per county inside stage5_predict.py's prediction loop.
"""

import numpy as np
import pandas as pd
from typing import Optional

from pipeline import config


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_zscore(value: float, mean: float, std: float) -> Optional[float]:
    """
    Compute (value - mean) / std safely.

    Returns None if std is zero, NaN, or either input is NaN.
    Caller handles None as "insufficient data".
    """
    if pd.isna(value) or pd.isna(mean) or pd.isna(std):
        return None
    if std == 0 or std < 1e-10:
        return None
    return float((value - mean) / std)


def _rolling_stats(series: pd.Series, window: int, min_obs: int) -> tuple:
    """
    Return (mean, std) of the last `window` non-NaN values in `series`.

    Uses the most recent `window` observations regardless of date gaps.
    Returns (NaN, NaN) if fewer than `min_obs` valid observations are available.
    """
    valid = series.dropna().values
    if len(valid) < min_obs:
        return float("nan"), float("nan")
    recent = valid[-window:]   # take the last `window` observations
    return float(np.mean(recent)), float(np.std(recent, ddof=1) if len(recent) > 1 else 0.0)


def _combine_trend_zscores(cf_z: Optional[float], fb_z: Optional[float]) -> Optional[float]:
    """
    Combine CalFresh and FoodBank z-scores into one trend anomaly score.

    Method is controlled by config.ALERT_TREND_COMBINE:
      "max"  → max of available z-scores (flag if either keyword spikes)
      "mean" → mean of available z-scores (both must move to flag)

    Returns None only if no z-score is available at all.
    """
    available = [z for z in [cf_z, fb_z] if z is not None]
    if not available:
        return None
    method = config.ALERT_TREND_COMBINE
    if method == "max":
        return float(max(available))
    else:  # "mean"
        return float(np.mean(available))


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def compute_warning_signals(
    county: str,
    predicted_rate: float,
    hist_rate_series: pd.Series,
    scaled_calfresh: Optional[float],
    scaled_foodbank: Optional[float],
    calfresh_hist_series: pd.Series,
    foodbank_hist_series: pd.Series,
) -> dict:
    """
    Compute all early-warning signal components for one county.

    Parameters
    ----------
    county : str
        County name (used only for logging).
    predicted_rate : float
        The baseline model's predicted SNAP_Application_Rate for the target month.
        This value is never modified — we only classify around it.
    hist_rate_series : pd.Series
        Historical SNAP_Application_Rate values for this county, sorted by date,
        up to (but not including) the target month.
        Typically: features_df[features_df["county"] == county]["SNAP_Application_Rate"]
    scaled_calfresh : float or None
        The prediction-window CalFresh Trends value, already scaled to the training
        reference frame by stage5_predict.scale_prediction_trends().
        None if no prediction data is available for this DMA.
    scaled_foodbank : float or None
        Same for FoodBank Trends.
    calfresh_hist_series : pd.Series
        Historical monthly_average_CalFresh for this county/DMA, sorted by date.
        Used to compute the Trends rolling baseline.
    foodbank_hist_series : pd.Series
        Same for FoodBank.

    Returns
    -------
    dict with keys:
        rolling_mean_rate         — recent county rate baseline (or NaN)
        rolling_std_rate          — recent county rate std (or NaN)
        prediction_zscore_recent  — (predicted_rate - rolling_mean) / rolling_std (or NaN)
        calfresh_trend_zscore     — CalFresh Trends anomaly z-score (or NaN)
        foodbank_trend_zscore     — FoodBank Trends anomaly z-score (or NaN)
        combined_trend_anomaly    — combined Trends anomaly score (or NaN)
        warning_score             — composite alert score (or NaN)
        warning_flag              — "Green" / "Yellow" / "Red" / "Gray"
    """
    W       = config.ALERT_ROLLING_WINDOW_W
    min_obs = config.ALERT_MIN_HISTORY
    alpha   = config.ALERT_ALPHA

    # ── 1. Prediction z-score relative to recent county baseline ──────────────
    #
    # Formula:
    #   rolling_mean_rate = mean of last W actual SNAP rates for this county
    #   rolling_std_rate  = std  of last W actual SNAP rates
    #   prediction_zscore_recent = (predicted_rate - rolling_mean_rate) / rolling_std_rate
    #
    # Interpretation: a z-score of 1.0 means the model predicts a rate that is
    # one standard deviation above this county's own recent normal — regardless of
    # whether the county is high or low statewide.

    roll_mean_rate, roll_std_rate = _rolling_stats(hist_rate_series, W, min_obs)
    pred_z = _safe_zscore(predicted_rate, roll_mean_rate, roll_std_rate)

    # ── 2. Trends anomaly z-scores ────────────────────────────────────────────
    #
    # Formula (per keyword):
    #   rolling_mean_trend = mean of last W monthly_average_{kw} values for this DMA
    #   rolling_std_trend  = std  of last W monthly_average_{kw} values
    #   trend_zscore = (current_scaled_value - rolling_mean_trend) / rolling_std_trend
    #
    # Interpretation: are people searching for CalFresh/FoodBank more than they
    # normally do in recent months?  This signal is independent of the model output.

    cf_mean, cf_std = _rolling_stats(calfresh_hist_series, W, min_obs)
    fb_mean, fb_std = _rolling_stats(foodbank_hist_series, W, min_obs)

    cf_z = _safe_zscore(scaled_calfresh, cf_mean, cf_std)
    fb_z = _safe_zscore(scaled_foodbank, fb_mean, fb_std)

    combined_trend = _combine_trend_zscores(cf_z, fb_z)

    # ── 3. Composite warning score ────────────────────────────────────────────
    #
    # Formula:
    #   warning_score = alpha * prediction_zscore_recent
    #                 + (1 - alpha) * combined_trend_anomaly
    #
    # If one component is missing (None/NaN), fall back to the available one.
    # If both are missing, warning_score is NaN → flag becomes Gray.

    if pred_z is not None and combined_trend is not None:
        warning_score = alpha * pred_z + (1 - alpha) * combined_trend
    elif pred_z is not None:
        warning_score = float(pred_z)          # Trends unavailable: prediction-only
    elif combined_trend is not None:
        warning_score = float(combined_trend)  # Rate history thin: Trends-only
    else:
        warning_score = float("nan")

    # ── 4. Warning flag ───────────────────────────────────────────────────────
    #
    # Thresholds from config.py — tune via experiments/evaluate_alerts.py.
    # Gray = insufficient data to compute a meaningful score.

    if pd.isna(warning_score):
        warning_flag = "Gray"
    elif warning_score >= config.ALERT_RED_THRESHOLD:
        warning_flag = "Red"
    elif warning_score >= config.ALERT_YELLOW_THRESHOLD:
        warning_flag = "Yellow"
    else:
        warning_flag = "Green"

    return {
        "rolling_mean_rate":        round(roll_mean_rate, 7) if not pd.isna(roll_mean_rate) else float("nan"),
        "rolling_std_rate":         round(roll_std_rate,  7) if not pd.isna(roll_std_rate)  else float("nan"),
        "prediction_zscore_recent": round(pred_z,         4) if pred_z is not None          else float("nan"),
        "calfresh_trend_zscore":    round(cf_z,           4) if cf_z is not None            else float("nan"),
        "foodbank_trend_zscore":    round(fb_z,           4) if fb_z is not None            else float("nan"),
        "combined_trend_anomaly":   round(combined_trend, 4) if combined_trend is not None  else float("nan"),
        "warning_score":            round(warning_score,  4) if not pd.isna(warning_score)  else float("nan"),
        "warning_flag":             warning_flag,
    }
