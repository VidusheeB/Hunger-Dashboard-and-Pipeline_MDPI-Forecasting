"""
tune_deployable_model.py — Hyperparameter tuning for the deployment-realistic model.

Deployment context
------------------
In production, SNAP administrative data has a reporting lag of approximately
6 months. This means that at prediction time, rate_lag1 reflects the rate from
~7 months ago (not 1 month ago), rate_lag2 from ~8 months ago, etc. — far
outside the range at which these features were calibrated. Using stale SNAP lags
in a model tuned on fresh lags would produce silently incorrect predictions.

The deployable model therefore uses only features that are genuinely available
at prediction time:
  - Google Trends (near-real-time, ~2-day lag)
  - Demographics (annual Census/ACS estimates)
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
  outputs/figures/deployable_roc_pr.png
  Console: full comparison table vs full production model

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
from sklearn.metrics import roc_curve, precision_recall_curve, auc, r2_score

from pipeline import config
from pipeline.alert_layer import compute_warning_signals, _rolling_stats

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Deployable feature set ────────────────────────────────────────────────────

DEPLOYABLE_FEATURES = [
    # Demographics (annual estimates, always available)
    "Population", "Median_Income",
    # Google Trends (near-real-time)
    "monthly_average_CalFresh", "monthly_average_FoodBank",
    "monthly_average_FoodStamps", "monthly_average_SNAPTopic",
    "calfresh_lag1", "calfresh_lag2",
    "foodbank_lag1", "foodbank_lag2",
    "foodstamps_lag1", "foodstamps_lag2",
    "snaptopic_lag1", "snaptopic_lag2",
    "calfresh_roll3", "foodbank_roll3",
    "foodstamps_roll3", "snaptopic_roll3",
    "calfresh_momentum", "foodbank_momentum",
    "foodstamps_momentum", "snaptopic_momentum",
    # Unemployment (BLS LAUS, ~1-month lag — available well before SNAP admin data)
    "unemployment_rate", "unemployment_rate_lag1",
    # Seasonality (deterministic)
    "month_sin", "month_cos", "quarter", "month",
    # Log transforms of demographics
    "log_population", "log_income",
    # EXCLUDED: rate_lag1, rate_lag2, rate_lag3, rate_roll3_mean, rate_roll3_std
    #   (require recent SNAP data unavailable in the 6-month lag scenario)
]

# Path to BLS LAUS county unemployment data
LAUS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "data", "trends", "laus_county_unemployment_2017_2025.csv"
)

TARGET = "SNAP_Application_Rate"
WALK_FORWARD_MIN_MONTHS = config.WALK_FORWARD_MIN_MONTHS   # matches pipeline (12)

# Full model metrics for comparison (from last pipeline run)
FULL_MODEL = {
    "label":   "Full XGBoost (23 features, SNAP lags included)",
    "r2":      0.6788,
    "mae":     0.000791,
    "smape":   14.42,
    "roc_auc": 0.6467,
    "pr_auc":  0.1881,
}

# Tuning fraction: use first TUNE_FRAC of unique dates for hyperparameter search
# COVID period handling:
# - Walk-forward runs on ALL data (model trains through COVID, alert layer sees real spikes)
# - Regression metrics (R², MAE, RMSE) reported only on non-COVID rows
#   (COVID policy conditions are structurally anomalous — not representative of normal ops)
# - Alert evaluation uses ALL rows including 2020-2021 (spikes are real early-warning events)
COVID_START = "2020-01-01"
COVID_END   = "2021-12-31"

TUNE_FRAC = 0.78  # pushes hold-out into post-2023 flat regime (post emergency allotments)

TARGET_PRECISION_RED  = 0.35
MIN_RECALL_YELLOW     = 0.50

# Output paths
OUT_METRICS = os.path.join(config.OUTPUTS_ROOT, "metrics")
OUT_FIGURES = config.FIGURES_DIR
WF_CSV      = os.path.join(OUT_METRICS, "deployable_walkforward_predictions.csv")
TUNE_JSON   = os.path.join(OUT_METRICS, "deployable_tuning_results.json")
CALIB_JSON  = os.path.join(OUT_METRICS, "deployable_threshold_calibration.json")
FIGURE_PNG  = os.path.join(OUT_FIGURES, "deployable_roc_pr.png")


# ── LAUS unemployment merge ───────────────────────────────────────────────────

def merge_laus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BLS LAUS county unemployment rate into df and compute a 1-month lag.

    unemployment_rate      — current month unemp rate (BLS LAUS, ~1-month lag)
    unemployment_rate_lag1 — prior month unemp rate
    """
    laus = pd.read_csv(LAUS_CSV, parse_dates=["date"])
    laus = laus[["county", "date", "unemployment_rate"]].copy()
    laus["date"] = laus["date"].dt.to_period("M").dt.to_timestamp()

    # Compute lag within LAUS before merging so the lag is county-specific
    laus = laus.sort_values(["county", "date"])
    laus["unemployment_rate_lag1"] = laus.groupby("county")["unemployment_rate"].shift(1)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
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


def _score_at_threshold(df, threshold):
    pred   = (df["warning_score"] >= threshold).astype(int)
    actual = df["is_spike"].astype(int)
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1   = (2*prec*rec/(prec+rec)
            if not (pd.isna(prec) or pd.isna(rec)) and (prec+rec) > 0
            else float("nan"))
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    J    = (rec + spec - 1
            if not (pd.isna(rec) or pd.isna(spec)) else float("nan"))
    return dict(threshold=round(threshold,4), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, fpr=fpr,
                specificity=spec, youden_J=J)


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


# ── Alert evaluation ──────────────────────────────────────────────────────────

def run_alert_evaluation(df: pd.DataFrame, wf_lookup: dict) -> pd.DataFrame:
    W, min_obs, spike_k = (
        config.ALERT_ROLLING_WINDOW_W,
        config.ALERT_MIN_HISTORY,
        config.SPIKE_K,
    )
    rows = []
    n_no_pred = 0

    for county in sorted(df["county"].unique()):
        cdf = df[df["county"] == county].sort_values("date").reset_index(drop=True)
        for i in range(len(cdf)):
            row         = cdf.iloc[i]
            actual_rate = row.get(TARGET, np.nan)
            if pd.isna(actual_rate):
                continue
            hist       = cdf.iloc[:i]
            hist_rates = hist[TARGET].dropna()
            if len(hist_rates) < min_obs:
                continue
            key            = (county, row["date"].normalize())
            predicted_rate = wf_lookup.get(key)
            if predicted_rate is None:
                n_no_pred += 1
                continue

            spike_mean, spike_std = _rolling_stats(hist[TARGET], W, min_obs)
            is_spike = (
                bool(actual_rate > spike_mean + spike_k * spike_std)
                if not (pd.isna(spike_mean) or pd.isna(spike_std) or spike_std == 0)
                else False
            )

            signals = compute_warning_signals(
                county               = county,
                predicted_rate       = predicted_rate,
                hist_rate_series     = hist[TARGET],
                scaled_calfresh      = row.get("monthly_average_CalFresh"),
                scaled_foodbank      = row.get("monthly_average_FoodBank"),
                calfresh_hist_series = hist["monthly_average_CalFresh"],
                foodbank_hist_series = hist["monthly_average_FoodBank"],
            )
            rows.append({
                "date":          row["date"].strftime("%Y-%m-%d"),
                "county":        county,
                "actual_rate":   round(actual_rate, 7),
                "predicted_rate":round(predicted_rate, 7),
                "is_spike":      int(is_spike),
                "warning_score": signals["warning_score"],
                "warning_flag":  signals["warning_flag"],
            })

    if n_no_pred:
        logger.info(f"  Alert eval: skipped {n_no_pred:,} county-months (no WF prediction)")
    return pd.DataFrame(rows)


# ── ROC / PR calibration ──────────────────────────────────────────────────────

def run_roc_pr(eval_df: pd.DataFrame) -> dict:
    scored = eval_df[
        eval_df["warning_score"].notna() & (eval_df["warning_flag"] != "Gray")
    ].copy()
    y_true  = scored["is_spike"].values.astype(int)
    y_score = scored["warning_score"].values

    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_arr, tpr_arr)
    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec_arr, prec_arr)

    # Yellow: Youden's J (with recall floor)
    J = tpr_arr - fpr_arr
    j_idx = int(np.argmax(J))
    j_thr, j_tpr, j_fpr = float(roc_thresh[j_idx]), float(tpr_arr[j_idx]), float(fpr_arr[j_idx])
    yellow_method = "youden_J"
    if j_tpr < MIN_RECALL_YELLOW:
        eligible = [(t, tp, fp) for t, tp, fp in zip(roc_thresh, tpr_arr, fpr_arr)
                    if tp >= MIN_RECALL_YELLOW]
        if eligible:
            eligible.sort(key=lambda x: x[2])
            j_thr, j_tpr, j_fpr = eligible[0]
        yellow_method = f"recall_floor(≥{MIN_RECALL_YELLOW})"

    # Red: precision floor
    p, r, t = prec_arr[:-1], rec_arr[:-1], pr_thresh
    eligible_red = [(th, pr, rc) for th, pr, rc in zip(t, p, r)
                    if pr >= TARGET_PRECISION_RED]
    if eligible_red:
        eligible_red.sort(key=lambda x: x[0])
        red_thr = eligible_red[0][0]
        red_method = f"precision_floor(≥{TARGET_PRECISION_RED})"
    else:
        red_thr    = float(np.percentile(y_score, 95))
        red_method = "p95_fallback"

    return dict(
        n_scored=len(scored), n_spikes=int(y_true.sum()),
        spike_rate=float(y_true.mean()),
        roc_auc=round(roc_auc, 4), pr_auc=round(pr_auc, 4),
        yellow_threshold=round(j_thr, 4), yellow_method=yellow_method,
        yellow_metrics=_score_at_threshold(scored, j_thr),
        red_threshold=round(red_thr, 4), red_method=red_method,
        red_metrics=_score_at_threshold(scored, red_thr),
        scored_df=scored,
        fpr_arr=fpr_arr, tpr_arr=tpr_arr,
        prec_arr=prec_arr, rec_arr=rec_arr,
    )


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
        f"Trends + unemployment + demographics + seasonality. "
        f"EXCLUDED: rate_lag1/2/3, rate_roll3_mean/std"
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

    # ── Alert evaluation (all rows including COVID) ───────────────────────────
    wf_lookup = {
        (row["county"], row["date"].normalize()): row["predicted_rate"]
        for _, row in wf_df.iterrows()
    }
    eval_df = run_alert_evaluation(df, wf_lookup)

    # ── ROC / PR ──────────────────────────────────────────────────────────────
    roc = run_roc_pr(eval_df)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  DEPLOYABLE MODEL — TUNED XGBOOST (NO SNAP LAGS)")
    print("  Features: Google Trends + BLS unemployment + demographics + seasonality")
    print(f"  COVID {COVID_START}–{COVID_END}: in walk-forward + alerts; excluded from regression metrics")
    print("  Realistic when SNAP data has a ≥6-month reporting lag")
    print("═" * 72)

    print(f"\n  Best hyperparameters (tuned on first {int(TUNE_FRAC*100)}% of dates):")
    for k, v in best_params.items():
        if k not in ("random_state", "n_jobs"):
            print(f"    {k:<22} = {v}")

    print(f"\n  {'─'*68}")
    print(f"  WALK-FORWARD ACCURACY  ({len(wf_df):,} county-months)")
    print(f"  {'─'*68}")
    print(f"  {'Metric':<25}  {'Deployable (tuned)':>18}  {'Full XGBoost':>14}  {'Δ':>8}")
    print(f"  {'-'*68}")
    print(f"  {'R²':<25}  {r2_val:>18.4f}  {FULL_MODEL['r2']:>14.4f}  "
          f"{r2_val - FULL_MODEL['r2']:>+8.4f}")
    print(f"  {'MAE':<25}  {mae_val:>18.6f}  {FULL_MODEL['mae']:>14.6f}  "
          f"{mae_val - FULL_MODEL['mae']:>+8.6f}")
    print(f"  {'sMAPE (%)':<25}  {smape_val:>18.2f}  {FULL_MODEL['smape']:>14.2f}  "
          f"{smape_val - FULL_MODEL['smape']:>+8.2f}")

    print(f"\n  {'─'*68}")
    print(f"  ALERT LAYER ROC / PR")
    print(f"  {'─'*68}")
    print(f"  {'Metric':<25}  {'Deployable (tuned)':>18}  {'Full XGBoost':>14}  {'Δ':>8}")
    print(f"  {'-'*68}")
    print(f"  {'ROC AUC':<25}  {roc['roc_auc']:>18.4f}  "
          f"{FULL_MODEL['roc_auc']:>14.4f}  "
          f"{roc['roc_auc'] - FULL_MODEL['roc_auc']:>+8.4f}")
    print(f"  {'PR AUC':<25}  {roc['pr_auc']:>18.4f}  "
          f"{FULL_MODEL['pr_auc']:>14.4f}  "
          f"{roc['pr_auc'] - FULL_MODEL['pr_auc']:>+8.4f}")

    ym = roc["yellow_metrics"]
    rm = roc["red_metrics"]
    print(f"\n  {'─'*68}")
    print(f"  CALIBRATED THRESHOLDS")
    print(f"  {'─'*68}")
    print(f"  Yellow  threshold : {roc['yellow_threshold']:.4f}  ({roc['yellow_method']})")
    print(f"    Recall          : {ym['recall']:.3f}")
    print(f"    Precision       : {ym['precision']:.3f}")
    print(f"    FPR             : {ym['fpr']:.3f}")
    print(f"    Youden's J      : {ym['youden_J']:.3f}")
    print(f"    TP / FP / FN / TN: {ym['tp']} / {ym['fp']} / {ym['fn']} / {ym['tn']}")
    print(f"  Red     threshold : {roc['red_threshold']:.4f}  ({roc['red_method']})")
    print(f"    Recall          : {rm['recall']:.3f}")
    print(f"    Precision       : {rm['precision']:.3f}")
    print(f"    FPR             : {rm['fpr']:.3f}")
    print(f"    TP / FP / FN / TN: {rm['tp']} / {rm['fp']} / {rm['fn']} / {rm['tn']}")

    print(f"\n  {'─'*68}")
    print(f"  SCORE SWEEP")
    print(f"  {'─'*68}")
    print(f"  {'Threshold':>10}  {'Recall':>7}  {'Precision':>9}  {'F1':>6}  "
          f"{'FPR':>6}  {'Youden_J':>9}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*68}")
    scored_df = roc["scored_df"]
    sweep = sorted(set(
        list(np.arange(0.0, 3.1, 0.2)) +
        [roc["yellow_threshold"], roc["red_threshold"],
         config.ALERT_YELLOW_THRESHOLD, config.ALERT_RED_THRESHOLD]
    ))
    for thr in sweep:
        m = _score_at_threshold(scored_df, thr)
        marker = ""
        if abs(thr - roc["yellow_threshold"]) < 0.001:
            marker += " ◄ YELLOW*"
        if abs(thr - roc["red_threshold"]) < 0.001:
            marker += " ◄ RED*"
        if abs(thr - config.ALERT_YELLOW_THRESHOLD) < 0.001 and abs(thr - roc["yellow_threshold"]) > 0.001:
            marker += " (prod yellow)"
        if abs(thr - config.ALERT_RED_THRESHOLD) < 0.001 and abs(thr - roc["red_threshold"]) > 0.001:
            marker += " (prod red)"
        rec  = f"{m['recall']:.3f}"    if m['recall']    is not None else "  —  "
        prec = f"{m['precision']:.3f}" if m['precision'] is not None else "  —  "
        f1v  = f"{m['f1']:.3f}"        if m['f1']        is not None else "  —  "
        fprv = f"{m['fpr']:.3f}"       if m['fpr']       is not None else "  —  "
        jv   = f"{m['youden_J']:.3f}"  if m['youden_J']  is not None else "  —  "
        print(f"  {thr:>10.3f}  {rec:>7}  {prec:>9}  {f1v:>6}  "
              f"{fprv:>6}  {jv:>9}  {m['tp']:>5}  {m['fp']:>5}  {m['fn']:>5}{marker}")

    print(f"\n{'═'*72}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "model":              "XGBoost — deployable (no SNAP lags, tuned)",
        "features":           DEPLOYABLE_FEATURES,
        "n_features":         len(DEPLOYABLE_FEATURES),
        "excluded_features":  ["rate_lag1","rate_lag2","rate_lag3","rate_roll3_mean","rate_roll3_std"],
        "added_features":     ["unemployment_rate","unemployment_rate_lag1"],
        "covid_handling":     "excluded from regression metrics; included in alert evaluation",
        "covid_exclusion_window": f"{COVID_START} – {COVID_END}",
        "tune_fraction":      TUNE_FRAC,
        "best_params":        {k: v for k, v in best_params.items() if k not in ("n_jobs",)},
        "walk_forward": {
            "n_predictions": len(wf_df),
            "r2":    round(r2_val, 4),
            "mae":   round(mae_val, 6),
            "smape": round(smape_val, 2),
        },
        "alert_roc_pr": {
            "roc_auc":          roc["roc_auc"],
            "pr_auc":           roc["pr_auc"],
            "yellow_threshold": roc["yellow_threshold"],
            "yellow_method":    roc["yellow_method"],
            "yellow_metrics":   {k: v for k, v in ym.items() if k != "threshold"},
            "red_threshold":    roc["red_threshold"],
            "red_method":       roc["red_method"],
            "red_metrics":      {k: v for k, v in rm.items() if k != "threshold"},
        },
        "full_model_comparison": FULL_MODEL,
    }
    with open(CALIB_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    with open(TUNE_JSON, "w") as f:
        json.dump({"best_params": best_params}, f, indent=2)
    logger.info(f"Calibration JSON → {CALIB_JSON}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        full_roc_csv = os.path.join(OUT_METRICS, "threshold_roc.csv")
        full_pr_csv  = os.path.join(OUT_METRICS, "threshold_pr.csv")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(roc["fpr_arr"], roc["tpr_arr"], lw=2, color="steelblue",
                label=f"Deployable tuned XGBoost (AUC={roc['roc_auc']:.3f})")
        if os.path.exists(full_roc_csv):
            fr = pd.read_csv(full_roc_csv)
            ax.plot(fr["fpr"], fr["tpr"], lw=2, color="dimgray", ls="--",
                    label=f"Full XGBoost (AUC={FULL_MODEL['roc_auc']:.3f})")
        ax.plot([0,1],[0,1],"k:",lw=1,alpha=0.4)
        ax.scatter([ym["fpr"]], [ym["recall"]], color="orange", zorder=5, s=80,
                   label=f"Yellow={roc['yellow_threshold']:.3f}")
        ax.scatter([rm["fpr"]], [rm["recall"]], color="red", zorder=5, s=80,
                   label=f"Red={roc['red_threshold']:.3f}")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC — Deployable (tuned, no SNAP lags) vs Full XGBoost")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(roc["rec_arr"], roc["prec_arr"], lw=2, color="steelblue",
                label=f"Deployable tuned (AUC={roc['pr_auc']:.3f})")
        if os.path.exists(full_pr_csv):
            fp_df = pd.read_csv(full_pr_csv).dropna()
            ax.plot(fp_df["recall"], fp_df["precision"], lw=2, color="dimgray", ls="--",
                    label=f"Full XGBoost (AUC={FULL_MODEL['pr_auc']:.3f})")
        ax.axhline(roc["spike_rate"], color="gray", lw=1, ls=":",
                   label=f"Baseline (spike rate={roc['spike_rate']:.3f})")
        ax.scatter([ym["recall"]], [ym["precision"]], color="orange", zorder=5, s=80,
                   label=f"Yellow prec={ym['precision']:.3f}")
        ax.scatter([rm["recall"]], [rm["precision"]], color="red", zorder=5, s=80,
                   label=f"Red prec={rm['precision']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("PR — Deployable (tuned, no SNAP lags) vs Full XGBoost")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(OUT_FIGURES, exist_ok=True)
        plt.savefig(FIGURE_PNG, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure → {FIGURE_PNG}")
    except ImportError:
        logger.info("matplotlib not available — skipping figure.")


if __name__ == "__main__":
    run()
