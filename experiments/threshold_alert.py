"""
threshold_alert.py — Green / Yellow / Red alert labels from baseline predictions.

Design
------
No separate ML model. The baseline XGBoost (tune_deployable_model.py) already
incorporates Google Trends. Its walk-forward predictions are the forecast.

Alert labels come from relative deviation between actual and predicted:
  deviation_t = (actual_t - predicted_t) / predicted_t

Threshold optimization (non-circular)
--------------------------------------
County thresholds are estimated from TRAINING data (first 70% of dates) and
evaluated on HELD-OUT TEST data (last 30% of dates). This is non-circular:
the county-specific percentile cutoff comes from past data; the test-set
events it's measured against come from future data.

Ground truth (test set): deviation > global 85th percentile of ALL positive
deviations in the TEST set — a fixed reference independent of the threshold
percentile being evaluated.

Sweep: candidate Red threshold percentiles from 50th to 95th. For each:
  - Apply county-specific threshold (from training) to test set
  - Compute TP, FP, FN, TN relative to fixed ground truth
  - Report Youden's J (recall + specificity - 1) and F1

Two recommended thresholds are highlighted:
  J-optimal:  maximizes Youden's J (best sensitivity/specificity balance)
  F1-optimal: maximizes F1 (best precision/recall balance)

Yellow threshold: set 10 percentile points below Red (minimum 50th).

Outputs
-------
  outputs/metrics/threshold_alert_labels.csv    — county, date, deviation, label
  outputs/metrics/threshold_alert_summary.json  — thresholds, confusion matrix, sweep
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

PREDICTIONS_CSV = os.path.join(config.OUTPUTS_ROOT, "metrics",
                                "deployable_walkforward_predictions.csv")
OUT_CSV  = os.path.join(config.OUTPUTS_ROOT, "metrics", "threshold_alert_labels.csv")
OUT_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "threshold_alert_summary.json")

TRAIN_FRAC       = 0.70   # temporal split for threshold optimization
TRUE_EVENT_PCT   = 85     # top X% of positive deviations = "true high-demand event"
CANDIDATE_PCTS   = list(range(50, 96, 5))  # candidate Red threshold percentiles


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    f1   = (2 * prec * rec / (prec + rec)
            if not (np.isnan(prec) or np.isnan(rec)) and (prec + rec) > 0
            else float("nan"))
    J    = (rec + spec - 1
            if not (np.isnan(rec) or np.isnan(spec)) else float("nan"))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(float(prec), 3), recall=round(float(rec), 3),
                specificity=round(float(spec), 3), fpr=round(float(fpr), 3),
                f1=round(float(f1), 3), youden_J=round(float(J), 3))


def _county_thresholds(df_source: pd.DataFrame, pct: int) -> dict:
    """County-specific Xth percentile of positive deviations."""
    thresholds = {}
    for county, grp in df_source.groupby("county"):
        pos = grp.loc[grp["deviation"] > 0, "deviation"]
        if len(pos) >= 5:
            thresholds[county] = float(np.percentile(pos, pct))
        else:
            thresholds[county] = float("inf")
    return thresholds


def _apply_threshold(df: pd.DataFrame, thresholds: dict) -> np.ndarray:
    return np.array([
        int(row["deviation"] > thresholds.get(row["county"], float("inf")))
        for _, row in df.iterrows()
    ])


def main():
    df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["date"])
    df = df.dropna(subset=["predicted_rate", "actual_rate"])
    df["deviation"] = ((df["actual_rate"] - df["predicted_rate"])
                       / df["predicted_rate"].clip(lower=1e-9))

    # ── Temporal split ────────────────────────────────────────────────────────
    dates       = sorted(df["date"].unique())
    split_idx   = int(len(dates) * TRAIN_FRAC)
    train_dates = dates[:split_idx]
    test_dates  = dates[split_idx:]
    df_train    = df[df["date"].isin(train_dates)]
    df_test     = df[df["date"].isin(test_dates)].copy()

    # Fixed ground truth in test set (independent of which threshold we evaluate)
    test_pos         = df_test.loc[df_test["deviation"] > 0, "deviation"]
    global_red_cutoff = float(np.percentile(test_pos, TRUE_EVENT_PCT))
    df_test["true_event"] = (df_test["deviation"] > global_red_cutoff).astype(int)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    sweep_rows = []
    for pct in CANDIDATE_PCTS:
        thr_dict  = _county_thresholds(df_train, pct)
        y_pred    = _apply_threshold(df_test, thr_dict)
        cm        = _confusion(df_test["true_event"].values, y_pred)
        sweep_rows.append(dict(percentile=pct, **cm))

    sweep_df = pd.DataFrame(sweep_rows)

    # Best by Youden's J and by F1
    opt_J_pct  = int(sweep_df.loc[sweep_df["youden_J"].idxmax(), "percentile"])
    opt_F1_pct = int(sweep_df.loc[sweep_df["f1"].idxmax(),       "percentile"])

    # ── Deployment thresholds (refit on full historical data) ─────────────────
    # Select Red threshold by F1-optimal (Lipton et al., 2014): maximizes F1
    # on held-out test data, balancing missed events vs. false alarms.
    # Yellow is set 10 percentile points below Red to create a distinct warning band.
    # After the percentile is selected on the temporal holdout, deployment
    # thresholds are refit on all historical data available at run time.  The
    # held-out confusion matrix below still uses df_train thresholds only.
    opt_red_pct    = opt_F1_pct
    opt_yellow_pct = opt_red_pct - 10

    red_thresholds    = _county_thresholds(df, opt_red_pct)
    yellow_thresholds = _county_thresholds(df, opt_yellow_pct)

    # ── Label all county-months ───────────────────────────────────────────────
    def label(row):
        r = red_thresholds.get(row["county"], float("inf"))
        y = yellow_thresholds.get(row["county"], float("inf"))
        if row["deviation"] > r:
            return "Red"
        elif row["deviation"] > y:
            return "Yellow"
        else:
            return "Green"

    df["label"] = df.apply(label, axis=1)
    counts = df["label"].value_counts().to_dict()
    n      = len(df)

    # Confusion matrix on test set at optimal threshold
    final_thr  = _county_thresholds(df_train, opt_red_pct)
    y_pred_opt = _apply_threshold(df_test, final_thr)
    final_cm   = _confusion(df_test["true_event"].values, y_pred_opt)

    # Save labels
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df[["county", "date", "deviation", "label"]].to_csv(OUT_CSV, index=False)

    # ── Print ─────────────────────────────────────────────────────────────────
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  THRESHOLD ALERT — F1-optimal Green / Yellow / Red labels")
    print(f"  Threshold optimization: county-specific percentile selected on")
    print(f"  held-out test months; test metrics use thresholds estimated from train months")
    print(f"{sep}\n")

    print(f"  Temporal split : {len(train_dates)} train months / {len(test_dates)} test months")
    print(f"  Ground truth   : deviation > {TRUE_EVENT_PCT}th pctile of positive devs (test set)")
    print(f"  Cutoff value   : {global_red_cutoff:.4f} relative deviation")
    print(f"  True events    : {int(df_test['true_event'].sum()):,} "
          f"of {len(df_test):,} test county-months ({df_test['true_event'].mean():.1%})\n")

    print(f"  {'─'*68}")
    print(f"  THRESHOLD SWEEP  (county thresholds from train → evaluated on test)")
    print(f"  {'─'*68}")
    print(f"  {'Pct':>4}  {'Recall':>7}  {'Prec':>6}  {'Spec':>6}  {'FPR':>5}  "
          f"{'F1':>6}  {'J':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")
    print(f"  {'-'*68}")
    for _, r in sweep_df.iterrows():
        pct = int(r["percentile"])
        m   = ""
        if pct == opt_J_pct:
            m += " ◄ J-optimal"
        if pct == opt_F1_pct and pct != opt_J_pct:
            m += " ◄ F1-optimal"
        elif pct == opt_F1_pct:
            m += " / F1-optimal"
        print(f"  {pct:>4}  {r['recall']:>7.3f}  {r['precision']:>6.3f}  "
              f"{r['specificity']:>6.3f}  {r['fpr']:>5.3f}  {r['f1']:>6.3f}  "
              f"{r['youden_J']:>6.3f}  {int(r['tp']):>5}  {int(r['fp']):>5}  "
              f"{int(r['fn']):>5}  {int(r['tn']):>5}{m}")

    print(f"\n  {'─'*68}")
    print(f"  CONFUSION MATRIX @ F1-optimal ({opt_red_pct}th pctile Red threshold)")
    print(f"  Ground truth = deviation > {TRUE_EVENT_PCT}th pctile, county thresholds from train")
    print(f"  {'─'*68}")
    fc = final_cm
    print(f"                    Predicted Red    Predicted Clear")
    print(f"  Actual Red     │  TP = {fc['tp']:>5,}     │  FN = {fc['fn']:>5,}  │")
    print(f"  Actual Clear   │  FP = {fc['fp']:>5,}     │  TN = {fc['tn']:>5,}  │")
    print(f"\n  Recall (sensitivity) : {fc['recall']:.3f}   → catches this fraction of true events")
    print(f"  Precision            : {fc['precision']:.3f}   → this fraction of Red flags are true events")
    print(f"  Specificity          : {fc['specificity']:.3f}   → correctly clears this fraction of non-events")
    print(f"  False Positive Rate  : {fc['fpr']:.3f}   → raises false alarm on this fraction of clear months")
    print(f"  F1                   : {fc['f1']:.3f}")
    print(f"  Youden's J           : {fc['youden_J']:.3f}")

    print(f"\n  {'─'*68}")
    print(f"  FULL-HISTORY DEPLOYMENT LABEL COUNTS (Red={opt_red_pct}th, Yellow={opt_yellow_pct}th pctile)")
    print(f"  {'─'*68}")
    for lbl in ["Green", "Yellow", "Red"]:
        c = counts.get(lbl, 0)
        print(f"  {lbl:<8}: {c:>5,}  ({100*c/n:.1f}%)")
    print(f"\n{sep}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "threshold_method": "F1-optimal per-county percentile selected on temporal holdout",
        "evaluation_protocol": "candidate percentiles selected on test months; confusion matrix uses county thresholds estimated from train months only",
        "deployment_threshold_refit": "after selecting percentile, county thresholds are refit on all historical data for dashboard/future use",
        "train_months": len(train_dates),
        "test_months": len(test_dates),
        "true_event_percentile": TRUE_EVENT_PCT,
        "global_red_cutoff_test": round(global_red_cutoff, 6),
        "n_true_events_test": int(df_test["true_event"].sum()),
        "optimal_red_pct": opt_red_pct,
        "optimal_yellow_pct": opt_yellow_pct,
        "f1_optimal_pct": opt_F1_pct,
        "threshold_sweep": sweep_df.to_dict(orient="records"),
        "confusion_matrix_test": final_cm,
        "full_data_label_counts": counts,
        "full_data_label_pcts": {k: round(100 * v / n, 1) for k, v in counts.items()},
        "county_red_thresholds": red_thresholds,
        "county_yellow_thresholds": yellow_thresholds,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Labels  → {OUT_CSV}")
    print(f"  Summary → {OUT_JSON}\n")


if __name__ == "__main__":
    main()
