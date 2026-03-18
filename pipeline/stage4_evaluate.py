"""
stage4_evaluate.py — Walk-forward validation for time-series model evaluation.

Standard k-fold cross-validation is inappropriate for time-series data because
it randomly mixes past and future — the model would see future data during
training. Walk-forward validation respects temporal order:

  For each month T (starting after WALK_FORWARD_MIN_MONTHS of history):
    - Train on all data before T
    - Predict on month T
    - Record metrics

This simulates exactly how the model would perform in production.

Outputs:
  outputs/metrics/walkforward_overall.json   — headline R², RMSE, MAE, sMAPE
  outputs/metrics/walkforward_per_month.csv  — per-month validation table
"""

import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline import config

logger = logging.getLogger(__name__)


# ── Metric computation ────────────────────────────────────────────────────────

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute R², RMSE, MAE, and sMAPE.

    sMAPE (symmetric mean absolute percentage error) is used instead of MAPE
    because MAPE is undefined when actuals are zero or near-zero — which occurs
    for small counties with very low SNAP application rates.

    sMAPE formula: mean(2|y - ŷ| / (|y| + |ŷ|)) × 100
    """
    y_pred = np.clip(y_pred, 0, None)

    # Guard against all-zero predictions (degenerate model output)
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(
        np.mean(2 * np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)) * 100
    )

    return {
        "r2":    float(r2_score(y_true, y_pred)),
        "rmse":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "smape": smape,
    }


# ── Walk-forward loop ─────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Run walk-forward validation across all available months.

    Each iteration trains a fresh model on all historical data before the
    test month and evaluates on the test month. The model is retrained from
    scratch each time — no information from the test period leaks into training.

    Returns: (overall_metrics_dict, per_month_DataFrame)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    per_month_rows = []
    all_true, all_pred = [], []

    n_skipped = 0
    for test_date in dates[config.WALK_FORWARD_MIN_MONTHS:]:
        train_mask = df["date"] < test_date
        test_mask  = df["date"] == test_date

        X_tr = df.loc[train_mask, feature_cols]
        y_tr = df.loc[train_mask, config.TARGET_COL].clip(lower=0)
        X_te = df.loc[test_mask,  feature_cols]
        y_te = df.loc[test_mask,  config.TARGET_COL]

        # Drop NaN rows within each window
        tr_ok = X_tr.notna().all(axis=1) & y_tr.notna()
        te_ok = X_te.notna().all(axis=1) & y_te.notna()

        if tr_ok.sum() < 10 or te_ok.sum() == 0:
            n_skipped += 1
            continue

        m = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        m.fit(X_tr[tr_ok], y_tr[tr_ok])
        preds = np.clip(m.predict(X_te[te_ok]), 0, None)
        actuals = y_te[te_ok].values

        month_metrics = calc_metrics(actuals, preds)
        per_month_rows.append({
            "month":       pd.Timestamp(test_date).strftime("%Y-%m"),
            "train_size":  int(tr_ok.sum()),
            "test_size":   int(te_ok.sum()),
            **month_metrics,
        })

        all_true.extend(actuals)
        all_pred.extend(preds)

    if n_skipped:
        logger.info(f"  Skipped {n_skipped} months (insufficient data)")

    if not all_true:
        logger.error("  No walk-forward predictions generated — check data coverage")
        return {}, pd.DataFrame()

    overall = calc_metrics(np.array(all_true), np.array(all_pred))
    overall["months_tested"]      = len(per_month_rows)
    overall["total_predictions"]  = len(all_true)

    # Per-month stats for the paper
    per_month_df = pd.DataFrame(per_month_rows)
    r2_vals = per_month_df["r2"].values
    overall["r2_mean"] = float(np.mean(r2_vals))
    overall["r2_std"]  = float(np.std(r2_vals))

    logger.info(
        f"\n  Walk-forward results ({overall['months_tested']} months, "
        f"{overall['total_predictions']:,} predictions):"
    )
    logger.info(f"    R²:    {overall['r2']:.4f}  (mean per-month: {overall['r2_mean']:.4f} ± {overall['r2_std']:.4f})")
    logger.info(f"    RMSE:  {overall['rmse']:.6f}")
    logger.info(f"    MAE:   {overall['mae']:.6f}")
    logger.info(f"    sMAPE: {overall['smape']:.2f}%")

    return overall, per_month_df


# ── Main entry point ──────────────────────────────────────────────────────────

def evaluate() -> dict:
    """
    Full evaluation stage: load training data, run walk-forward, save outputs.
    Returns the overall metrics dict.
    """
    logger.info("=== STAGE 4: EVALUATE (WALK-FORWARD VALIDATION) ===")

    df = pd.read_csv(config.MODELLING_CSV)
    logger.info(f"  Loaded: {config.MODELLING_CSV}  {df.shape}")

    feature_cols = [f for f in config.FEATURE_COLS if f in df.columns]
    missing = set(config.FEATURE_COLS) - set(feature_cols)
    if missing:
        logger.warning(f"  Missing features: {missing}")

    # Drop rows missing any feature or target before validation
    mask = df[feature_cols].notna().all(axis=1) & df[config.TARGET_COL].notna()
    df_clean = df[mask].copy()
    logger.info(f"  Rows after NaN drop: {len(df_clean):,} (dropped {(~mask).sum()})")

    overall, per_month_df = run_walk_forward(df_clean, feature_cols)

    if not overall:
        return {}

    # Save overall metrics
    with open(config.WF_OVERALL_JSON, "w") as f:
        json.dump(overall, f, indent=2)
    logger.info(f"  Overall metrics → {config.WF_OVERALL_JSON}")

    # Save per-month table
    per_month_df.to_csv(config.WF_PER_MONTH_CSV, index=False)
    logger.info(f"  Per-month metrics → {config.WF_PER_MONTH_CSV}")

    return overall
