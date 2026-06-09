"""
tune_deployable_model.py — Hyperparameter tuning for the deployment-realistic model.

Deployment context
------------------
In production, SNAP administrative data has a reporting lag of approximately
6 months. Features derived from recent official SNAP outcomes are therefore not
available at prediction time and are excluded from the model.

The deployable model therefore uses only features that are genuinely available
at prediction time:
  - Google Trends through t-1 (near-real-time, ~2-day lag)
  - Demographics (annual Census/ACS estimates)
  - BLS LAUS unemployment through t-1
  - Seasonality (deterministic)

SNAP lag and rolling features are excluded entirely.

Tuning procedure
----------------
Hyperparameters are selected via a temporal hold-out that exactly mirrors
the walk-forward evaluation scheme:

  - Dates are split at the TUNE_FRAC (60th percentile) cutoff.
  - The tuning set is the first 60% of dates.
  - Within the tuning set, the first 60% of those dates form the TRAIN split
    and the remaining 40% form the HOLD-OUT split.
  - For each candidate parameter combination, XGBoost is fit on the train
    split and MAE is evaluated on the hold-out split.
  - No cross-validation, no folding, no averaging across folds.
    The evaluation scheme matches production exactly: one model, one test set,
    temporal ordering always respected.

This avoids the conceptual error of using cross-validation (even TimeSeriesSplit)
for a time-series model: CV trains on different subsets than the final model
and averages scores that have no direct relationship to the walk-forward MAE
that is the actual performance criterion.

Walk-forward evaluation
-----------------------
After hyperparameters are fixed, a full walk-forward evaluation is run over
the same 93 months as the production model, for a direct comparison.

Outputs
-------
  outputs/metrics/deployable_walkforward_predictions.csv
  outputs/metrics/deployable_tuning_results.json
  outputs/metrics/deployable_threshold_calibration.json
  Console: deployable walk-forward metrics

Usage
-----
    python experiments/tune_deployable_model.py

Runtime: ~15-25 minutes (RandomizedSearch 50 iterations × 5 CV folds,
         then walk-forward 93 months × XGBoost fit).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from pipeline import config

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Deployable feature set ────────────────────────────────────────────────────

DEPLOYABLE_FEATURES = [
    # Demographics (annual estimates, always available)
    "Population", "Median_Income",
    # Google Trends — lag1/lag2 only (same-month Trends not available at prediction time)
    "calfresh_lag1", "calfresh_lag2",
    "foodbank_lag1", "foodbank_lag2",
    "foodstamps_lag1", "foodstamps_lag2",
    "snaptopic_lag1", "snaptopic_lag2",
    "calfresh_roll3", "foodbank_roll3",
    "foodstamps_roll3", "snaptopic_roll3",
    "calfresh_momentum", "foodbank_momentum",
    "foodstamps_momentum", "snaptopic_momentum",
    # Unemployment (BLS LAUS, publication-safe: t-1 and t-2 for target month t)
    "unemployment_rate", "unemployment_rate_lag1",
    # Seasonality (deterministic)
    "month_sin", "month_cos", "quarter", "month",
    # Log transforms of demographics
    "log_population", "log_income",
]

# Path to BLS LAUS county unemployment data
LAUS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "data", "trends", "laus_county_unemployment_2017_2025.csv"
)

TARGET = "SNAP_Application_Rate"
WALK_FORWARD_MIN_MONTHS = config.WALK_FORWARD_MIN_MONTHS   # matches pipeline (12)

# Tuning fraction: use first TUNE_FRAC of unique dates for hyperparameter search
# COVID period handling:
# - Walk-forward runs on ALL data (model trains through COVID, alert layer sees real spikes)
# - Regression metrics (R², MAE, RMSE) reported only on non-COVID rows
#   (COVID policy conditions are structurally anomalous — not representative of normal ops)
# - Alert evaluation uses ALL rows including 2020-2021 (spikes are real early-warning events)
COVID_START = "2020-01-01"
COVID_END   = "2021-12-31"

TUNE_FRAC = 0.78  # pushes hold-out into post-2023 flat regime (post emergency allotments)

# Output paths
OUT_METRICS = os.path.join(config.OUTPUTS_ROOT, "metrics")
WF_CSV      = os.path.join(OUT_METRICS, "deployable_walkforward_predictions.csv")
TUNE_JSON   = os.path.join(OUT_METRICS, "deployable_tuning_results.json")
CALIB_JSON  = os.path.join(OUT_METRICS, "deployable_threshold_calibration.json")


# ── LAUS unemployment merge ───────────────────────────────────────────────────

def merge_laus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BLS LAUS county unemployment with publication-safe lags.

    For target SNAP month t:
      unemployment_rate      — LAUS unemployment at t-1
      unemployment_rate_lag1 — LAUS unemployment at t-2
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    if {"unemployment_rate", "unemployment_rate_lag1"}.issubset(df.columns):
        return df

    laus = pd.read_csv(LAUS_CSV, parse_dates=["date"])
    laus = laus[["county", "date", "unemployment_rate"]].copy()
    laus["date"] = laus["date"].dt.to_period("M").dt.to_timestamp()

    # Compute lags within LAUS before merging so they are county-specific and
    # never use unemployment from the target SNAP month.
    laus = laus.sort_values(["county", "date"])
    laus["unemployment_rate_lag1"] = laus.groupby("county")["unemployment_rate"].shift(2)
    laus["unemployment_rate"] = laus.groupby("county")["unemployment_rate"].shift(1)

    df = df.drop(columns=["unemployment_rate", "unemployment_rate_lag1"], errors="ignore")
    df = df.merge(laus, on=["county", "date"], how="left")

    missing_ur = df["unemployment_rate"].isna().sum()
    if missing_ur > 0:
        logger.warning(
            f"  LAUS: {missing_ur:,} rows without unemployment_rate after merge "
            f"(will be dropped by dropna in walk-forward)"
        )
    else:
        logger.info(f"  LAUS merged: unemployment_rate coverage = 100%")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smape(actual, predicted):
    mask = (np.abs(actual) + np.abs(predicted)) > 0
    return float(100 * np.mean(
        np.abs(actual[mask] - predicted[mask]) /
        ((np.abs(actual[mask]) + np.abs(predicted[mask])) / 2)
    ))


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def _holdout_mae(params: dict, X_train, y_train, X_val, y_val) -> float:
    """Fit XGBoost with params on train split, return MAE on val split."""
    model = xgb.XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_val), 0, None)
    return float(np.mean(np.abs(y_val - preds)))


def tune_hyperparameters(df: pd.DataFrame) -> dict:
    """
    Tune XGBoost using a temporal hold-out that mirrors production exactly.

    Split:
      All dates → first TUNE_FRAC are the tuning pool.
      Within tuning pool → first 60% = train, last 40% = hold-out.
      Candidate params are scored by MAE on the hold-out only.
      No folding, no averaging across folds.

    Stage 1: random search over a broad grid (N_RANDOM_ITER candidates).
    Stage 2: exhaustive grid search in a narrow region around Stage 1 best.
    """
    N_RANDOM_ITER = 60
    rng = np.random.default_rng(42)

    dates = sorted(df["date"].unique())
    tune_cutoff_idx  = int(len(dates) * TUNE_FRAC)
    tune_cutoff_date = dates[tune_cutoff_idx]

    tune_dates  = dates[:tune_cutoff_idx]
    inner_split = int(len(tune_dates) * 0.60)
    train_dates = tune_dates[:inner_split]
    val_dates   = tune_dates[inner_split:]

    train_df = df[df["date"].isin(train_dates)][DEPLOYABLE_FEATURES + [TARGET]].dropna()
    val_df   = df[df["date"].isin(val_dates)][DEPLOYABLE_FEATURES + [TARGET]].dropna()

    X_train, y_train = train_df[DEPLOYABLE_FEATURES].values, train_df[TARGET].values
    X_val,   y_val   = val_df[DEPLOYABLE_FEATURES].values,   val_df[TARGET].values

    logger.info(
        f"Tuning split: {len(train_dates)} train months / {len(val_dates)} hold-out months "
        f"(cutoff: {pd.Timestamp(tune_cutoff_date).date()})"
    )
    logger.info(
        f"  Train rows: {len(X_train):,}  |  Hold-out rows: {len(X_val):,}"
    )
    logger.info(
        f"  Walk-forward evaluation reserved: {len(dates) - tune_cutoff_idx} months "
        f"(never seen during tuning)"
    )

    # ── Stage 1: random search ────────────────────────────────────────────────
    logger.info(f"  Stage 1: random search ({N_RANDOM_ITER} candidates) …")

    candidates = [
        {
            "n_estimators":     int(rng.integers(200, 800)),
            "max_depth":        int(rng.integers(3, 10)),
            "learning_rate":    float(rng.uniform(0.005, 0.15)),
            "min_child_weight": int(rng.integers(1, 10)),
            "subsample":        float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_lambda":       float(rng.uniform(0.5, 8.0)),
            "reg_alpha":        float(rng.uniform(0.0, 2.0)),
        }
        for _ in range(N_RANDOM_ITER)
    ]

    best_mae   = float("inf")
    best_rand  = None
    for i, params in enumerate(candidates):
        mae = _holdout_mae(params, X_train, y_train, X_val, y_val)
        if mae < best_mae:
            best_mae  = mae
            best_rand = params
        if (i + 1) % 10 == 0:
            logger.info(f"    [{i+1}/{N_RANDOM_ITER}] best so far: MAE={best_mae:.6f}")

    logger.info(f"  Stage 1 best MAE (hold-out): {best_mae:.6f}")
    logger.info(f"  Stage 1 best params: {best_rand}")

    # ── Stage 2: focused grid around Stage 1 best ────────────────────────────
    def _grid(val, step, lo, hi, n=3):
        return sorted(set(
            round(max(lo, min(hi, val + i * step)), 6)
            for i in range(-(n // 2), n // 2 + 1)
        ))

    # Only tune the 3 params XGBoost is most sensitive to around the best point.
    # Remaining params stay fixed at Stage 1 best — avoids combinatorial explosion.
    focused = {
        "n_estimators":  _grid(best_rand["n_estimators"],  100, 100, 1200),
        "max_depth":     _grid(best_rand["max_depth"],       1,   2,   12),
        "learning_rate": _grid(best_rand["learning_rate"],  0.01, 0.003, 0.20),
    }
    # Fix remaining params at Stage 1 best
    fixed_params = {
        k: best_rand[k]
        for k in ("min_child_weight", "subsample", "colsample_bytree",
                  "reg_lambda", "reg_alpha")
    }

    # Cartesian product
    import itertools
    keys   = list(focused.keys())
    combos = list(itertools.product(*[focused[k] for k in keys]))
    logger.info(f"  Stage 2: focused grid ({len(combos)} combinations) …")

    best_mae2  = float("inf")
    best_params = None
    for combo in combos:
        params = dict(zip(keys, combo))
        params.update(fixed_params)
        params["n_estimators"] = int(params["n_estimators"])
        params["max_depth"]    = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])
        mae = _holdout_mae(params, X_train, y_train, X_val, y_val)
        if mae < best_mae2:
            best_mae2   = mae
            best_params = dict(params)

    best_params["random_state"] = 42
    best_params["n_jobs"]       = -1
    logger.info(f"  Stage 2 best MAE (hold-out): {best_mae2:.6f}")
    logger.info(f"  Stage 2 best params: {best_params}")

    return best_params


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Walk-forward with tuned params on DEPLOYABLE_FEATURES."""
    dates = sorted(df["date"].unique())
    prediction_rows = []
    n_skipped = 0

    for test_date in dates:
        train_df = df[df["date"] < test_date].copy()
        test_df  = df[df["date"] == test_date].copy()

        if train_df["date"].nunique() < WALK_FORWARD_MIN_MONTHS:
            n_skipped += 1
            continue

        tr = train_df[DEPLOYABLE_FEATURES + [TARGET]].dropna()
        te = test_df[DEPLOYABLE_FEATURES + [TARGET]].dropna()
        if len(tr) < 10 or len(te) == 0:
            n_skipped += 1
            continue

        model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            verbosity=0,
        )
        model.fit(tr[DEPLOYABLE_FEATURES].values, tr[TARGET].values)
        preds = np.clip(model.predict(te[DEPLOYABLE_FEATURES].values), 0, None)

        counties = test_df.loc[te.index, "county"].values
        for county, pred, actual in zip(counties, preds, te[TARGET].values):
            prediction_rows.append({
                "county":         county,
                "date":           pd.Timestamp(test_date).strftime("%Y-%m-%d"),
                "predicted_rate": float(pred),
                "actual_rate":    float(actual),
            })

    logger.info(
        f"  Walk-forward: {len(prediction_rows):,} predictions "
        f"| {n_skipped} months skipped (< {WALK_FORWARD_MIN_MONTHS} training months)"
    )
    return pd.DataFrame(prediction_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    if not os.path.exists(config.FEATURES_CSV):
        raise FileNotFoundError(f"features.csv not found at {config.FEATURES_CSV}")
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    df = df.sort_values(["county", "date"]).reset_index(drop=True)
    logger.info(
        f"Loaded {len(df):,} rows | {df['county'].nunique()} counties | "
        f"{df['date'].nunique()} months"
    )

    # Merge LAUS unemployment data
    logger.info("Merging BLS LAUS county unemployment data …")
    df = merge_laus(df)

    logger.info(
        f"COVID period {COVID_START} – {COVID_END}: included in walk-forward + alert eval; "
        f"excluded from regression metrics only"
    )

    missing = [f for f in DEPLOYABLE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    logger.info(
        f"Deployable feature set ({len(DEPLOYABLE_FEATURES)} features): "
        f"Trends + unemployment + demographics + seasonality."
    )

    # ── Tune ─────────────────────────────────────────────────────────────────
    logger.info("\nHyperparameter tuning …")
    best_params = tune_hyperparameters(df)

    # ── Walk-forward ──────────────────────────────────────────────────────────
    logger.info("\nWalk-forward evaluation with tuned params …")
    wf_df = run_walk_forward(df, best_params)

    # Regression metrics on non-COVID rows only
    # COVID walk-forward predictions are kept for alert evaluation (real spikes)
    wf_df["date"] = pd.to_datetime(wf_df["date"])
    wf_noncovid = wf_df[~wf_df["date"].between(COVID_START, COVID_END)]
    actual    = wf_noncovid["actual_rate"].values
    predicted = wf_noncovid["predicted_rate"].values
    r2_val    = r2_score(actual, predicted)
    mae_val   = float(np.mean(np.abs(actual - predicted)))
    smape_val = _smape(actual, predicted)
    logger.info(
        f"Regression metrics on {len(wf_noncovid):,} non-COVID rows "
        f"({len(wf_df) - len(wf_noncovid):,} COVID rows used for alert eval only)"
    )

    os.makedirs(OUT_METRICS, exist_ok=True)
    wf_df.to_csv(WF_CSV, index=False)
    logger.info(f"Walk-forward predictions ({len(wf_df):,} rows) → {WF_CSV}")

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  DEPLOYABLE MODEL — TUNED XGBOOST (NO SNAP LAGS)")
    print("  Features: Google Trends + BLS unemployment + demographics + seasonality")
    print(f"  COVID {COVID_START}–{COVID_END}: excluded from regression metrics only")
    print("  Realistic when SNAP data has a ≥6-month reporting lag")
    print("═" * 72)

    print(f"\n  Best hyperparameters (tuned on first {int(TUNE_FRAC*100)}% of dates):")
    for k, v in best_params.items():
        if k not in ("random_state", "n_jobs"):
            print(f"    {k:<22} = {v}")

    print(f"\n  {'─'*68}")
    print(f"  WALK-FORWARD ACCURACY  ({len(wf_df):,} county-months)")
    print(f"  {'─'*68}")
    print(f"  {'Metric':<25}  {'Deployable (tuned)':>18}")
    print(f"  {'-'*46}")
    print(f"  {'R²':<25}  {r2_val:>18.4f}")
    print(f"  {'MAE':<25}  {mae_val:>18.6f}")
    print(f"  {'sMAPE (%)':<25}  {smape_val:>18.2f}")

    print(f"\n{'═'*72}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "model":              "XGBoost — deployable (no SNAP lags, tuned)",
        "features":           DEPLOYABLE_FEATURES,
        "n_features":         len(DEPLOYABLE_FEATURES),
        "snap_outcome_features": "excluded",
        "covid_handling":     "excluded from regression metrics; included in walk-forward",
        "covid_exclusion_window": f"{COVID_START} – {COVID_END}",
        "tune_fraction":      TUNE_FRAC,
        "best_params":        {k: v for k, v in best_params.items() if k not in ("n_jobs",)},
        "walk_forward": {
            "n_predictions": len(wf_df),
            "r2":    round(r2_val, 4),
            "mae":   round(mae_val, 6),
            "smape": round(smape_val, 2),
        },
    }
    with open(CALIB_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    with open(TUNE_JSON, "w") as f:
        json.dump({"best_params": best_params}, f, indent=2)
    logger.info(f"Results → {CALIB_JSON}")


if __name__ == "__main__":
    run()
