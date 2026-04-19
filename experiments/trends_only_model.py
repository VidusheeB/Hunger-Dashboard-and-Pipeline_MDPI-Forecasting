"""
trends_only_model.py — Trends-signal-only model as a diagnostic baseline.

Fits an XGBoost model on Google Trends features alone (no SNAP history,
no demographics) and evaluates how well Trends alone can predict CalFresh
application rate and drive the alert layer.

Purpose
-------
Separates two questions:
  1. How much predictive signal do Trends carry in isolation?
  2. Does the alert layer work even with only Trends-based predictions?

This is a diagnostic / ablation experiment.  Nothing in the pipeline is
modified; no pipeline outputs are overwritten.

Features used (full baseline minus SNAP rate lags/rolling)
-----------------------------------------------------------
  Same as the full pipeline model EXCEPT the lagged and rolling SNAP
  rate features are removed:
    Removed: rate_lag1, rate_lag2, rate_lag3, rate_roll3_mean, rate_roll3_std
    Kept: all demographics, Trends (raw + lags + rolling + momentum),
          seasonality, log transforms
  This isolates how much the SNAP rate autocorrelation contributes
  vs everything else.

Model: XGBoost with the same tuned hyperparameters as the full pipeline model
       (config.XGBOOST_PARAMS), but restricted to 10 Trends features only.
Walk-forward: same scheme as stage 4 — train on all months before T,
predict on T.  Minimum 24 training months required before first prediction
(matching stage 4's WALK_FORWARD_MIN_MONTHS).

Outputs (written to outputs/metrics/ — prefixed with trends_only_)
-------
  trends_only_walkforward_predictions.csv
  trends_only_alert_evaluation.csv
  trends_only_threshold_roc.csv
  trends_only_threshold_pr.csv
  trends_only_threshold_calibration.json
  trends_only_roc_pr.png     (in outputs/figures/)

Usage
-----
    python experiments/trends_only_model.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, precision_recall_curve, auc, r2_score

from pipeline import config
from pipeline.alert_layer import compute_warning_signals, _rolling_stats, _safe_zscore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TRENDS_FEATURES = [
    # Base demographics + Trends (same as full model)
    "Population", "Median_Income",
    "monthly_average_CalFresh", "monthly_average_FoodBank", "month",
    # Lags — Google Trends only
    "calfresh_lag1", "calfresh_lag2",
    "foodbank_lag1", "foodbank_lag2",
    # Rolling — Google Trends only
    "calfresh_roll3", "foodbank_roll3",
    # Momentum
    "calfresh_momentum", "foodbank_momentum",
    # Seasonality
    "month_sin", "month_cos", "quarter",
    # Transforms
    "log_population", "log_income",
    # Excluded: rate_lag1/2/3, rate_roll3_mean, rate_roll3_std
]

TARGET = "SNAP_Application_Rate"
WALK_FORWARD_MIN_MONTHS = 24   # match stage 4

# Full-model walk-forward metrics (from last pipeline run) for comparison
FULL_MODEL_STATS = {
    "r2":    0.6788,
    "mae":   0.000791,
    "smape": 14.42,
    "roc_auc": 0.6467,
    "pr_auc":  0.1881,
}

# Output paths
OUT_METRICS  = os.path.join(config.OUTPUTS_ROOT, "metrics")
OUT_FIGURES  = config.FIGURES_DIR
WF_PRED_CSV  = os.path.join(OUT_METRICS, "trends_only_walkforward_predictions.csv")
EVAL_CSV     = os.path.join(OUT_METRICS, "trends_only_alert_evaluation.csv")
ROC_CSV      = os.path.join(OUT_METRICS, "trends_only_threshold_roc.csv")
PR_CSV       = os.path.join(OUT_METRICS, "trends_only_threshold_pr.csv")
CALIB_JSON   = os.path.join(OUT_METRICS, "trends_only_threshold_calibration.json")
FIGURE_PNG   = os.path.join(OUT_FIGURES, "trends_only_roc_pr.png")

TARGET_PRECISION_RED = 0.35
MIN_RECALL_YELLOW    = 0.50


# ── Helpers (mirrored from calibrate_thresholds.py) ───────────────────────────

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
    precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall      = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if not (pd.isna(precision) or pd.isna(recall)) and (precision + recall) > 0
          else float("nan"))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    J   = (recall + specificity - 1
           if not (pd.isna(recall) or pd.isna(specificity))
           else float("nan"))
    return dict(
        threshold=round(threshold, 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, fpr=fpr,
        specificity=specificity, youden_J=J,
    )


def _youden_threshold(fpr_arr, tpr_arr, thresholds):
    J   = tpr_arr - fpr_arr
    idx = int(np.argmax(J))
    return float(thresholds[idx]), float(tpr_arr[idx]), float(fpr_arr[idx])


def _recall_floor_threshold(fpr_arr, tpr_arr, thresholds, min_recall):
    eligible = [(t, tp, fp) for t, tp, fp in zip(thresholds, tpr_arr, fpr_arr)
                if tp >= min_recall]
    if not eligible:
        idx = int(np.argmax(tpr_arr))
        return float(thresholds[idx]), float(tpr_arr[idx]), float(fpr_arr[idx])
    eligible.sort(key=lambda x: x[2])
    return eligible[0]


def _precision_floor_threshold(precisions, recalls, thresholds, target):
    p, r, t = precisions[:-1], recalls[:-1], thresholds
    eligible = [(th, pr, rc) for th, pr, rc in zip(t, p, r) if pr >= target]
    if not eligible:
        return None
    eligible.sort(key=lambda x: x[0])
    return eligible[0]


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward linear regression on trends-only features.
    Returns a DataFrame with columns: county, date, predicted_rate, actual_rate.
    """
    dates = sorted(df["date"].unique())
    prediction_rows = []
    n_skipped_min = 0
    n_skipped_nan = 0

    for test_date in dates:
        train_df = df[df["date"] < test_date].copy()
        test_df  = df[df["date"] == test_date].copy()

        # Need minimum training months (by unique date count, not rows)
        n_train_months = train_df["date"].nunique()
        if n_train_months < WALK_FORWARD_MIN_MONTHS:
            n_skipped_min += 1
            continue

        # Drop rows with NaN in features or target
        tr_clean = train_df[TRENDS_FEATURES + [TARGET]].dropna()
        te_clean = test_df[TRENDS_FEATURES + [TARGET]].dropna()

        if len(tr_clean) < 10 or len(te_clean) == 0:
            n_skipped_nan += 1
            continue

        X_train = tr_clean[TRENDS_FEATURES].values
        y_train = tr_clean[TARGET].values
        X_test  = te_clean[TRENDS_FEATURES].values
        y_test  = te_clean[TARGET].values

        model = xgb.XGBRegressor(
            **config.XGBOOST_PARAMS,
            objective="reg:squarederror",
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)

        te_counties = test_df.loc[te_clean.index, "county"].values
        for county, pred, actual in zip(te_counties, preds, y_test):
            prediction_rows.append({
                "county":         county,
                "date":           pd.Timestamp(test_date).strftime("%Y-%m-%d"),
                "predicted_rate": float(pred),
                "actual_rate":    float(actual),
            })

    logger.info(
        f"  Walk-forward: {len(prediction_rows):,} predictions "
        f"| {n_skipped_min} months skipped (< {WALK_FORWARD_MIN_MONTHS} training months) "
        f"| {n_skipped_nan} skipped (NaN)"
    )
    return pd.DataFrame(prediction_rows)


# ── Alert evaluation ──────────────────────────────────────────────────────────

def run_alert_evaluation(df: pd.DataFrame, wf_lookup: dict) -> pd.DataFrame:
    """
    Replicate evaluate_alerts.py logic using trends-only predicted_rate.
    Returns the evaluation DataFrame.
    """
    W       = config.ALERT_ROLLING_WINDOW_W
    min_obs = config.ALERT_MIN_HISTORY
    spike_k = config.SPIKE_K

    rows = []
    n_no_pred = 0

    for county in sorted(df["county"].unique()):
        cdf = df[df["county"] == county].sort_values("date").reset_index(drop=True)

        for i in range(len(cdf)):
            row         = cdf.iloc[i]
            actual_rate = row.get("SNAP_Application_Rate", np.nan)
            if pd.isna(actual_rate):
                continue

            hist      = cdf.iloc[:i]
            hist_rates = hist["SNAP_Application_Rate"].dropna()
            if len(hist_rates) < min_obs:
                continue

            key            = (county, row["date"].normalize())
            predicted_rate = wf_lookup.get(key)
            if predicted_rate is None:
                n_no_pred += 1
                continue

            # Spike label
            spike_mean, spike_std = _rolling_stats(hist["SNAP_Application_Rate"], W, min_obs)
            if pd.isna(spike_mean) or pd.isna(spike_std) or spike_std == 0:
                is_spike = False
            else:
                is_spike = bool(actual_rate > spike_mean + spike_k * spike_std)

            # Alert signals using trends-only predicted rate
            signals = compute_warning_signals(
                county               = county,
                predicted_rate       = predicted_rate,
                hist_rate_series     = hist["SNAP_Application_Rate"],
                scaled_calfresh      = row.get("monthly_average_CalFresh"),
                scaled_foodbank      = row.get("monthly_average_FoodBank"),
                calfresh_hist_series = hist["monthly_average_CalFresh"],
                foodbank_hist_series = hist["monthly_average_FoodBank"],
            )

            rows.append({
                "date":                     row["date"].strftime("%Y-%m-%d"),
                "county":                   county,
                "actual_rate":              round(actual_rate, 7),
                "predicted_rate":           round(predicted_rate, 7),
                "is_spike":                 int(is_spike),
                "warning_score":            signals["warning_score"],
                "warning_flag":             signals["warning_flag"],
                "prediction_zscore_recent": signals["prediction_zscore_recent"],
                "combined_trend_anomaly":   signals["combined_trend_anomaly"],
            })

    if n_no_pred > 0:
        logger.info(f"  Alert eval: skipped {n_no_pred:,} county-months with no WF prediction")

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

    # Save curves
    pd.DataFrame({"threshold": roc_thresh, "fpr": fpr_arr, "tpr": tpr_arr,
                  "youden_J": tpr_arr - fpr_arr}).to_csv(ROC_CSV, index=False)
    pd.DataFrame({"threshold": list(pr_thresh) + [float("nan")],
                  "precision": prec_arr, "recall": rec_arr}).to_csv(PR_CSV, index=False)

    # Yellow: Youden's J
    j_thr, j_tpr, j_fpr = _youden_threshold(fpr_arr, tpr_arr, roc_thresh)
    if j_tpr < MIN_RECALL_YELLOW:
        j_thr, j_tpr, j_fpr = _recall_floor_threshold(
            fpr_arr, tpr_arr, roc_thresh, MIN_RECALL_YELLOW
        )
        yellow_method = f"recall_floor(≥{MIN_RECALL_YELLOW})"
    else:
        yellow_method = "youden_J"

    yellow_metrics = _score_at_threshold(scored, j_thr)

    # Red: precision floor
    red_result = _precision_floor_threshold(prec_arr, rec_arr, pr_thresh, TARGET_PRECISION_RED)
    if red_result is None:
        red_thr    = float(np.percentile(y_score, 95))
        red_method = f"p95_fallback"
    else:
        red_thr, _, _ = red_result
        red_method = f"precision_floor(≥{TARGET_PRECISION_RED})"

    red_metrics = _score_at_threshold(scored, red_thr)

    return dict(
        n_scored=len(scored),
        n_spikes=int(y_true.sum()),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        yellow_threshold=round(j_thr, 4),
        yellow_method=yellow_method,
        yellow_metrics=yellow_metrics,
        red_threshold=round(red_thr, 4),
        red_method=red_method,
        red_metrics=red_metrics,
        scored_df=scored,
        fpr_arr=fpr_arr,
        tpr_arr=tpr_arr,
        prec_arr=prec_arr,
        rec_arr=rec_arr,
        spike_rate=float(y_true.mean()),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # Load features
    if not os.path.exists(config.FEATURES_CSV):
        raise FileNotFoundError(f"features.csv not found at {config.FEATURES_CSV}")
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    df = df.sort_values(["county", "date"]).reset_index(drop=True)
    logger.info(
        f"Loaded features.csv: {len(df):,} rows, {df['county'].nunique()} counties, "
        f"{df['date'].nunique()} months"
    )

    # Check all trends features present
    missing = [f for f in TRENDS_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing trends features in features.csv: {missing}")

    logger.info(f"Trends-only features ({len(TRENDS_FEATURES)}): {TRENDS_FEATURES}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    logger.info("Running walk-forward (linear regression, trends-only) …")
    wf_df = run_walk_forward(df)

    if wf_df.empty:
        logger.error("Walk-forward produced no predictions — check data.")
        return

    # Walk-forward metrics
    actual    = wf_df["actual_rate"].values
    predicted = wf_df["predicted_rate"].values
    r2   = r2_score(actual, predicted)
    mae  = float(np.mean(np.abs(actual - predicted)))
    smape_val = _smape(actual, predicted)

    os.makedirs(OUT_METRICS, exist_ok=True)
    wf_df.to_csv(WF_PRED_CSV, index=False)
    logger.info(f"Walk-forward predictions ({len(wf_df):,} rows) → {WF_PRED_CSV}")

    # ── Alert evaluation ──────────────────────────────────────────────────────
    wf_df["date"] = pd.to_datetime(wf_df["date"])
    wf_lookup = {
        (row["county"], row["date"].normalize()): row["predicted_rate"]
        for _, row in wf_df.iterrows()
    }

    logger.info("Running alert evaluation …")
    eval_df = run_alert_evaluation(df, wf_lookup)
    eval_df.to_csv(EVAL_CSV, index=False)
    logger.info(f"Alert evaluation ({len(eval_df):,} rows) → {EVAL_CSV}")

    # ── ROC / PR ──────────────────────────────────────────────────────────────
    logger.info("Computing ROC / PR curves …")
    roc_results = run_roc_pr(eval_df)
    logger.info(f"ROC → {ROC_CSV}  |  PR → {PR_CSV}")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  TRENDS-ONLY MODEL RESULTS")
    print("  XGBoost — full features minus SNAP rate lags/rolling (rate_lag1/2/3, rate_roll3_mean/std)")
    print("═" * 72)

    print(f"\n  {'─'*68}")
    print(f"  WALK-FORWARD PREDICTION ACCURACY  ({len(wf_df):,} county-months)")
    print(f"  {'─'*68}")
    print(f"  {'Metric':<25}  {'Trends-only':>14}  {'Full XGBoost':>14}  {'Δ':>8}")
    print(f"  {'-'*68}")
    print(f"  {'R²':<25}  {r2:>14.4f}  {FULL_MODEL_STATS['r2']:>14.4f}  "
          f"{r2 - FULL_MODEL_STATS['r2']:>+8.4f}")
    print(f"  {'MAE':<25}  {mae:>14.6f}  {FULL_MODEL_STATS['mae']:>14.6f}  "
          f"{mae - FULL_MODEL_STATS['mae']:>+8.6f}")
    print(f"  {'sMAPE (%)':<25}  {smape_val:>14.2f}  {FULL_MODEL_STATS['smape']:>14.2f}  "
          f"{smape_val - FULL_MODEL_STATS['smape']:>+8.2f}")

    print(f"\n  {'─'*68}")
    print(f"  ALERT LAYER ROC / PR PERFORMANCE")
    print(f"  {'─'*68}")
    print(f"  {'Metric':<25}  {'Trends-only':>14}  {'Full XGBoost':>14}  {'Δ':>8}")
    print(f"  {'-'*68}")
    print(f"  {'ROC AUC':<25}  {roc_results['roc_auc']:>14.4f}  "
          f"{FULL_MODEL_STATS['roc_auc']:>14.4f}  "
          f"{roc_results['roc_auc'] - FULL_MODEL_STATS['roc_auc']:>+8.4f}")
    print(f"  {'PR AUC':<25}  {roc_results['pr_auc']:>14.4f}  "
          f"{FULL_MODEL_STATS['pr_auc']:>14.4f}  "
          f"{roc_results['pr_auc'] - FULL_MODEL_STATS['pr_auc']:>+8.4f}")

    ym = roc_results["yellow_metrics"]
    rm = roc_results["red_metrics"]
    print(f"\n  {'─'*68}")
    print(f"  CALIBRATED THRESHOLDS  (same criteria as calibrate_thresholds.py)")
    print(f"  {'─'*68}")
    print(f"  Yellow threshold : {roc_results['yellow_threshold']:.4f}  "
          f"({roc_results['yellow_method']})")
    print(f"    Recall         : {ym['recall']:.3f}")
    print(f"    Precision      : {ym['precision']:.3f}")
    print(f"    FPR            : {ym['fpr']:.3f}")
    print(f"    Youden's J     : {ym['youden_J']:.3f}")
    print(f"  Red threshold    : {roc_results['red_threshold']:.4f}  "
          f"({roc_results['red_method']})")
    print(f"    Recall         : {rm['recall']:.3f}")
    print(f"    Precision      : {rm['precision']:.3f}")
    print(f"    FPR            : {rm['fpr']:.3f}")

    print(f"\n  {'─'*68}")
    print(f"  SCORE SWEEP (key thresholds)")
    print(f"  {'─'*68}")
    print(f"  {'Threshold':>10}  {'Recall':>7}  {'Precision':>9}  {'F1':>6}  "
          f"{'FPR':>6}  {'Youden_J':>9}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*68}")
    scored_df = roc_results["scored_df"]
    for thr in np.arange(0.0, 3.1, 0.2):
        m = _score_at_threshold(scored_df, thr)
        marker = ""
        if abs(thr - roc_results["yellow_threshold"]) < 0.11:
            marker += " ◄ ~YELLOW"
        if abs(thr - roc_results["red_threshold"]) < 0.11:
            marker += " ◄ ~RED"
        print(f"  {thr:>10.2f}  {m['recall']:>7.3f}  {m['precision']:>9.3f}  "
              f"{m['f1']:>6.3f}  {m['fpr']:>6.3f}  {m['youden_J']:>9.3f}  "
              f"{m['tp']:>5}  {m['fp']:>5}  {m['fn']:>5}{marker}")

    print(f"\n{'═'*72}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "model": "XGBoost — full features minus SNAP rate lags/rolling (rate_lag1/2/3, rate_roll3_mean/std)",
        "features": TRENDS_FEATURES,
        "walk_forward": {
            "n_predictions": len(wf_df),
            "r2": round(r2, 4),
            "mae": round(mae, 6),
            "smape": round(smape_val, 2),
        },
        "alert_roc_pr": {
            "roc_auc": roc_results["roc_auc"],
            "pr_auc":  roc_results["pr_auc"],
            "yellow_threshold": roc_results["yellow_threshold"],
            "yellow_method":    roc_results["yellow_method"],
            "yellow_metrics":   {k: v for k, v in roc_results["yellow_metrics"].items()
                                 if k not in ("threshold",)},
            "red_threshold": roc_results["red_threshold"],
            "red_method":    roc_results["red_method"],
            "red_metrics":   {k: v for k, v in roc_results["red_metrics"].items()
                              if k not in ("threshold",)},
        },
        "xgboost_params": config.XGBOOST_PARAMS,
    "full_model_comparison": FULL_MODEL_STATS,
    }
    with open(CALIB_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON → {CALIB_JSON}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr_full = None  # load from full model ROC CSV if available
        full_roc_csv = os.path.join(OUT_METRICS, "threshold_roc.csv")
        full_pr_csv  = os.path.join(OUT_METRICS, "threshold_pr.csv")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ROC
        ax = axes[0]
        ax.plot(roc_results["fpr_arr"], roc_results["tpr_arr"], lw=2, color="steelblue",
                label=f"Trends-only XGBoost (AUC={roc_results['roc_auc']:.3f})")
        if os.path.exists(full_roc_csv):
            full_roc = pd.read_csv(full_roc_csv)
            ax.plot(full_roc["fpr"], full_roc["tpr"], lw=2, color="dimgray", ls="--",
                    label=f"Full XGBoost (AUC={FULL_MODEL_STATS['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.4)
        ax.scatter([ym["fpr"]], [ym["recall"]], color="orange", zorder=5, s=80,
                   label=f"Yellow={roc_results['yellow_threshold']:.3f}")
        ax.scatter([rm["fpr"]], [rm["recall"]], color="red", zorder=5, s=80,
                   label=f"Red={roc_results['red_threshold']:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve — Trends-only vs Full XGBoost")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # PR
        ax = axes[1]
        ax.plot(roc_results["rec_arr"], roc_results["prec_arr"], lw=2, color="steelblue",
                label=f"Trends-only XGBoost (AUC={roc_results['pr_auc']:.3f})")
        if os.path.exists(full_pr_csv):
            full_pr = pd.read_csv(full_pr_csv).dropna()
            ax.plot(full_pr["recall"], full_pr["precision"], lw=2, color="dimgray", ls="--",
                    label=f"Full XGBoost (AUC={FULL_MODEL_STATS['pr_auc']:.3f})")
        ax.axhline(roc_results["spike_rate"], color="gray", lw=1, ls=":",
                   label=f"Baseline (spike rate {roc_results['spike_rate']:.3f})")
        ax.scatter([ym["recall"]], [ym["precision"]], color="orange", zorder=5, s=80,
                   label=f"Yellow prec={ym['precision']:.3f}")
        ax.scatter([rm["recall"]], [rm["precision"]], color="red", zorder=5, s=80,
                   label=f"Red prec={rm['precision']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall — Trends-only vs Full XGBoost")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(OUT_FIGURES, exist_ok=True)
        plt.savefig(FIGURE_PNG, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure → {FIGURE_PNG}")
    except ImportError:
        logger.info("matplotlib not available — skipping figure.")


if __name__ == "__main__":
    run()
