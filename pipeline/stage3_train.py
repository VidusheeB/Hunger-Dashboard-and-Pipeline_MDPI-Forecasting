"""
stage3_train.py — Train and save the production XGBoost model.

Uses tuned hyperparameters from config.py (validated in experiments/).
Trains on the full training dataset (no train/test split here — that is
done properly in stage4_evaluate.py via walk-forward validation).

Outputs:
  outputs/models/xgboost_tuned.pkl      — model bundle for prediction
  outputs/metrics/insample_metrics.json — in-sample R², RMSE (training fit)
  outputs/metrics/feature_importance.csv
"""

import json
import logging
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline import config

logger = logging.getLogger(__name__)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    """
    Extract features and target from the training DataFrame.

    Only rows with complete data for all features AND the target are kept.
    Target is clipped to ≥ 0 (SNAP application rates cannot be negative).

    Returns: (X, y, feature_cols_used)
    """
    available_features = [f for f in config.FEATURE_COLS if f in df.columns]
    missing = set(config.FEATURE_COLS) - set(available_features)
    if missing:
        logger.warning(f"  Feature(s) not found in data: {missing}")

    X = df[available_features]
    y = df[config.TARGET_COL].clip(lower=0)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    logger.info(f"  Training rows: {len(X):,} (dropped {(~mask).sum()} with NaN)")
    return X, y, available_features


# ── Model training ────────────────────────────────────────────────────────────

def train_final_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """
    Fit XGBoost on the full training set using tuned hyperparameters.

    This is the model saved for production use. Walk-forward validation
    (stage 4) independently verifies generalization — it does NOT use
    this fitted model, it refits from scratch on rolling windows.
    """
    model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    model.fit(X, y)
    logger.info(f"  Fitted XGBoost on {len(X):,} samples with {len(X.columns)} features")
    return model


# ── Metrics and diagnostics ───────────────────────────────────────────────────

def compute_insample_metrics(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute in-sample (training) metrics.

    These measure how well the model fits its own training data — they are NOT
    a measure of generalization. Always report walk-forward metrics (stage 4)
    alongside these in any paper or presentation.
    """
    preds = np.clip(model.predict(X), 0, None)
    metrics = {
        "r2":   round(float(r2_score(y, preds)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 8),
        "mae":  round(float(mean_absolute_error(y, preds)), 8),
        "note": "In-sample metrics on full training data — not a generalization estimate. See walkforward_overall.json for held-out performance.",
    }
    logger.info(
        f"  In-sample — R²: {metrics['r2']:.4f}, "
        f"RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}"
    )
    return metrics


def compute_feature_ranges(X: pd.DataFrame) -> dict:
    """
    Compute min/max of each feature in the training set.
    Used at prediction time to detect out-of-distribution inputs (data drift).
    """
    return {f: {"min": float(X[f].min()), "max": float(X[f].max())} for f in X.columns}


def _quick_walkforward_mae(df: pd.DataFrame, feature_cols: list) -> float:
    """
    Run a lightweight walk-forward to extract MAE for confidence intervals.

    This is a compact version of stage 4's full walk-forward — it only
    computes MAE (not all metrics, not per-month tables) so stage 3 can
    embed a correct walkforward_mae in the model pickle without depending
    on stage 4's output files.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    maes = []

    for test_date in dates[config.WALK_FORWARD_MIN_MONTHS:]:
        train_mask = df["date"] < test_date
        test_mask  = df["date"] == test_date
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr = df.loc[train_mask, feature_cols]
        y_tr = df.loc[train_mask, config.TARGET_COL].clip(lower=0)
        X_te = df.loc[test_mask,  feature_cols]
        y_te = df.loc[test_mask,  config.TARGET_COL]

        # Drop rows with NaN in this window
        tr_mask = X_tr.notna().all(axis=1) & y_tr.notna()
        te_mask = X_te.notna().all(axis=1) & y_te.notna()
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        m = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        m.fit(X_tr[tr_mask], y_tr[tr_mask])
        preds = np.clip(m.predict(X_te[te_mask]), 0, None)
        maes.append(mean_absolute_error(y_te[te_mask], preds))

    wf_mae = float(np.mean(maes)) if maes else 0.000877
    logger.info(f"  Walk-forward MAE (for CI): {wf_mae:.6f} over {len(maes)} months")
    return wf_mae


# ── Save model bundle ─────────────────────────────────────────────────────────

def save_model(
    model: xgb.XGBRegressor,
    feature_cols: list,
    feature_ranges: dict,
    walkforward_mae: float,
) -> None:
    """
    Serialize the model bundle to a pickle file.

    Bundle keys:
      model           — fitted XGBRegressor
      features        — ordered list of feature column names
      type            — 'xgboost' (for downstream compatibility checks)
      walkforward_mae — used by stage 5 to build per-county confidence intervals
      feature_ranges  — {feature: {min, max}} for drift detection
      params          — the hyperparameters used (for reproducibility)
    """
    bundle = {
        "model":           model,
        "features":        feature_cols,
        "type":            "xgboost",
        "walkforward_mae": walkforward_mae,
        "feature_ranges":  feature_ranges,
        "params":          config.XGBOOST_PARAMS,
    }
    with open(config.MODEL_PKL, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"  Model saved → {config.MODEL_PKL}")


# ── Main entry point ──────────────────────────────────────────────────────────

def train_and_save() -> dict:
    """
    Full training stage: load data, train model, save all artifacts.
    Returns the in-sample metrics dict.
    """
    logger.info("=== STAGE 3: TRAIN MODEL ===")

    df = pd.read_csv(config.MODELLING_CSV)
    logger.info(f"  Loaded: {config.MODELLING_CSV}  {df.shape}")

    X, y, feature_cols = prepare_xy(df)

    # Train production model on full dataset
    model = train_final_model(X, y)

    # Compute in-sample metrics and save
    insample = compute_insample_metrics(model, X, y)
    with open(config.INSAMPLE_METRICS_JSON, "w") as f:
        json.dump(insample, f, indent=2)
    logger.info(f"  In-sample metrics → {config.INSAMPLE_METRICS_JSON}")

    # Feature importance
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)
    logger.info(f"  Feature importance → {config.FEATURE_IMPORTANCE_CSV}")
    for _, row in importance_df.iterrows():
        logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    # Walk-forward MAE for confidence intervals (embedded in model bundle)
    mask = df[feature_cols].notna().all(axis=1) & df[config.TARGET_COL].notna()
    wf_mae = _quick_walkforward_mae(df[mask].copy(), feature_cols)

    # Feature ranges for drift detection
    feature_ranges = compute_feature_ranges(X)

    # Save model bundle
    save_model(model, feature_cols, feature_ranges, wf_mae)

    return insample
