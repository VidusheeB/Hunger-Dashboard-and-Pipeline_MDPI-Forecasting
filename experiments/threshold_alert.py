"""
threshold_alert.py — Green / Yellow / Red alert labels from baseline predictions.

Design
------
No separate ML model. The baseline XGBoost (tune_deployable_model.py) already
incorporates Google Trends. Its walk-forward predictions are the forecast.

Alert logic (county-specific thresholds):
  deviation_t = (actual_t - predicted_t) / predicted_t

  For each county, compute Yellow and Red cutoffs from its own history of
  positive deviations:
    Yellow threshold  = 75th percentile of that county's positive deviations
    Red    threshold  = 85th percentile of that county's positive deviations

  Label at time t:
    Green  — deviation_t <= yellow_threshold  (or deviation is non-positive)
    Yellow — yellow_threshold < deviation_t <= red_threshold
    Red    — deviation_t > red_threshold

Output
------
  outputs/metrics/threshold_alert_labels.csv   — county, date, deviation, label
  outputs/metrics/threshold_alert_summary.json — overall counts, per-county thresholds
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

PREDICTIONS_CSV = os.path.join(config.OUTPUTS_ROOT, "metrics", "deployable_walkforward_predictions.csv")
OUT_CSV  = os.path.join(config.OUTPUTS_ROOT, "metrics", "threshold_alert_labels.csv")
OUT_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "threshold_alert_summary.json")

YELLOW_PCT = 75
RED_PCT    = 85


def main():
    df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["date"])
    df = df.dropna(subset=["predicted_rate", "actual_rate"])

    # Relative deviation
    df["deviation"] = (df["actual_rate"] - df["predicted_rate"]) / df["predicted_rate"].clip(lower=1e-9)

    # County-specific thresholds from positive deviations
    thresholds = {}
    for county, grp in df.groupby("county"):
        pos = grp.loc[grp["deviation"] > 0, "deviation"]
        if len(pos) >= 5:
            yellow = float(np.percentile(pos, YELLOW_PCT))
            red    = float(np.percentile(pos, RED_PCT))
        else:
            yellow = float("inf")
            red    = float("inf")
        thresholds[county] = {"yellow": yellow, "red": red}

    def label(row):
        t = thresholds.get(row["county"], {"yellow": float("inf"), "red": float("inf")})
        if row["deviation"] > t["red"]:
            return "Red"
        elif row["deviation"] > t["yellow"]:
            return "Yellow"
        else:
            return "Green"

    df["label"] = df.apply(label, axis=1)

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df[["county", "date", "deviation", "label"]].to_csv(OUT_CSV, index=False)

    # Summary
    counts = df["label"].value_counts().to_dict()
    n = len(df)
    summary = {
        "n_total": n,
        "yellow_pct_cutoff": YELLOW_PCT,
        "red_pct_cutoff": RED_PCT,
        "label_counts": counts,
        "label_pcts": {k: round(100 * v / n, 1) for k, v in counts.items()},
        "county_thresholds": thresholds,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Print report
    sep = "═" * 68
    print(f"\n{sep}")
    print(f"  BASELINE ALERT LABELS (county-specific thresholds)")
    print(f"  Yellow ≥ {YELLOW_PCT}th pctile | Red ≥ {RED_PCT}th pctile of positive deviations")
    print(f"{sep}\n")
    print(f"  n = {n:,} county-months")
    for lbl in ["Green", "Yellow", "Red"]:
        c = counts.get(lbl, 0)
        print(f"  {lbl:<8}: {c:>5,}  ({100*c/n:.1f}%)")
    print(f"\n  Labels → {OUT_CSV}")
    print(f"  Summary → {OUT_JSON}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
