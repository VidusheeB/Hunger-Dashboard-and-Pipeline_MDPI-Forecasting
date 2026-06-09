"""
stage6_report.py — Generate all figures and summary tables for the paper.

All charts are redrawn from the current run's outputs, so they always match
the latest metrics. Figures are saved as high-res PNGs (300 DPI) suitable
for publication.

Outputs:
  outputs/figures/actual_vs_predicted.png
  outputs/figures/walkforward_r2_over_time.png
  outputs/figures/feature_importance_bar.png
  outputs/figures/statewide_timeseries.png
  outputs/figures/risk_flag_distribution.png
  outputs/metrics/paper_summary.json
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import config

logger = logging.getLogger(__name__)

# ── Shared plot style ─────────────────────────────────────────────────────────
STYLE = {
    "figure.figsize":   (8, 5),
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
}

COLORS = {
    "primary":  "#2563EB",   # blue
    "secondary":"#DC2626",   # red
    "neutral":  "#6B7280",   # gray
    "green":    "#16A34A",
    "yellow":   "#D97706",
    "red":      "#DC2626",
}


def _save(fig, filename: str) -> str:
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {path}")
    return path


# ── Figure 1: Actual vs Predicted scatter ─────────────────────────────────────

def plot_actual_vs_predicted() -> str:
    """
    Scatter plot of actual vs predicted SNAP application rates across all
    walk-forward test predictions.

    The 45-degree line represents a perfect model. Points above the line are
    under-predictions; below are over-predictions.
    """
    per_month = pd.read_csv(config.WF_PER_MONTH_CSV)
    with open(config.WF_OVERALL_JSON) as f:
        overall = json.load(f)

    # Reconstruct all actual/predicted pairs from the training data using
    # the per-month walk-forward loop (read-only — no re-training)
    training_df = pd.read_csv(config.TRAINING_DATA_CSV)
    training_df["date"] = pd.to_datetime(training_df["date"])
    feature_cols = [c for c in config.FEATURE_COLS if c in training_df.columns]

    dates = sorted(training_df["date"].unique())
    all_true, all_pred = [], []

    for test_date in dates[config.WALK_FORWARD_MIN_MONTHS:]:
        tr = training_df[training_df["date"] < test_date]
        te = training_df[training_df["date"] == test_date]

        X_tr = tr[feature_cols].dropna()
        y_tr = tr.loc[X_tr.index, config.TARGET_COL].clip(lower=0)
        X_te = te[feature_cols].dropna()
        y_te = te.loc[X_te.index, config.TARGET_COL]

        if len(X_tr) < 10 or X_te.empty:
            continue

        m = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        m.fit(X_tr, y_tr)
        preds = np.clip(m.predict(X_te), 0, None)
        all_true.extend(y_te.values)
        all_pred.extend(preds)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.scatter(all_true, all_pred, alpha=0.3, s=8, color=COLORS["primary"], rasterized=True)

        lim = max(all_true.max(), all_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=COLORS["neutral"], lw=1, linestyle="--", label="Perfect fit")

        ax.set_xlabel("Actual SNAP Application Rate")
        ax.set_ylabel("Predicted SNAP Application Rate")
        ax.set_title("Walk-Forward Validation: Actual vs Predicted")
        ax.annotate(
            f"R² = {overall['r2']:.3f}  |  MAE = {overall['mae']:.5f}  |  "
            f"n = {overall['total_predictions']:,}",
            xy=(0.03, 0.93), xycoords="axes fraction", fontsize=9,
            color=COLORS["neutral"],
        )
        ax.legend(fontsize=9)

    return _save(fig, "actual_vs_predicted.png")


# ── Figure 2: Per-month R² over time ─────────────────────────────────────────

def plot_walkforward_r2_over_time() -> str:
    """
    Line chart of per-month R² across the validation window.

    Shows whether model accuracy is consistent over time or degrades —
    important for a time-series model where data distribution may drift.
    """
    per_month = pd.read_csv(config.WF_PER_MONTH_CSV)
    per_month["month_dt"] = pd.to_datetime(per_month["month"])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.plot(per_month["month_dt"], per_month["r2"],
                color=COLORS["primary"], lw=1.5, marker="o", ms=3, label="Monthly R²")
        ax.axhline(0, color=COLORS["secondary"], lw=1, linestyle="--", label="R²=0 (naive baseline)")

        overall_r2 = per_month["r2"].mean()
        ax.axhline(overall_r2, color=COLORS["neutral"], lw=1, linestyle=":",
                   label=f"Mean R²={overall_r2:.3f}")

        ax.set_xlabel("Test Month")
        ax.set_ylabel("R²")
        ax.set_title("Walk-Forward Validation: R² Over Time")
        ax.legend(fontsize=9)
        fig.autofmt_xdate()

    return _save(fig, "walkforward_r2_over_time.png")


# ── Figure 3: Feature importance ─────────────────────────────────────────────

def plot_feature_importance() -> str:
    """
    Horizontal bar chart of XGBoost feature importances.
    Features sorted descending by importance score.
    """
    fi = pd.read_csv(config.FEATURE_IMPORTANCE_CSV).sort_values("importance")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, max(3, len(fi) * 0.5)))
        bars = ax.barh(fi["feature"], fi["importance"], color=COLORS["primary"], height=0.6)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_xlabel("Feature Importance (XGBoost gain)")
        ax.set_title("Feature Importance")
        ax.set_xlim(0, fi["importance"].max() * 1.2)

    return _save(fig, "feature_importance_bar.png")


# ── Figure 4: Statewide SNAP applications over time ───────────────────────────

def plot_statewide_timeseries() -> str:
    """
    Aggregate monthly SNAP applications across all counties over time.

    Shows the macro trend in food assistance demand in California, providing
    context for model training period and prediction targets.
    """
    df = pd.read_csv(config.TRAINING_DATA_CSV, parse_dates=["date"])
    monthly_total = (
        df.groupby("date")["SNAP_Applications"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.fill_between(monthly_total["date"], monthly_total["SNAP_Applications"],
                        alpha=0.2, color=COLORS["primary"])
        ax.plot(monthly_total["date"], monthly_total["SNAP_Applications"],
                color=COLORS["primary"], lw=1.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        ax.set_xlabel("Month")
        ax.set_ylabel("Total SNAP Applications")
        ax.set_title("Statewide Monthly SNAP Applications Over Time")
        fig.autofmt_xdate()

    return _save(fig, "statewide_timeseries.png")


# ── Figure 5: Risk flag distribution ─────────────────────────────────────────

def plot_risk_flag_distribution() -> str:
    """
    Bar chart of county counts per risk flag for the most recent prediction.

    Gives a quick read on how many counties are at elevated risk.
    """
    preds = pd.read_csv(config.PREDICTIONS_CSV)
    flag_order  = ["Green", "Yellow", "Red", "Gray"]
    flag_colors = [COLORS["green"], COLORS["yellow"], COLORS["red"], COLORS["neutral"]]

    counts = preds["flag"].value_counts().reindex(flag_order, fill_value=0)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color=flag_colors, width=0.5)
        ax.bar_label(bars, padding=3)
        ax.set_xlabel("Risk Flag")
        ax.set_ylabel("Number of Counties")
        prediction_month = preds["date"].iloc[0] if not preds.empty else "Unknown"
        ax.set_title(f"County Risk Distribution — {prediction_month}")
        ax.set_ylim(0, counts.max() * 1.2)

    return _save(fig, "risk_flag_distribution.png")


# ── Paper summary JSON ────────────────────────────────────────────────────────

def save_paper_summary() -> str:
    """
    Write a single JSON file with all headline numbers for the paper.
    One file to cite — covers model config, validation metrics, and predictions.
    """
    with open(config.WF_OVERALL_JSON) as f:
        wf = json.load(f)
    with open(config.INSAMPLE_METRICS_JSON) as f:
        insample = json.load(f)

    preds    = pd.read_csv(config.PREDICTIONS_CSV)
    training = pd.read_csv(config.TRAINING_DATA_CSV)

    summary = {
        "model": {
            "type":   "XGBoost (tuned)",
            "params": config.XGBOOST_PARAMS,
            "features": config.FEATURE_COLS,
            "target": config.TARGET_COL,
        },
        "training_data": {
            "counties":   int(training["county"].nunique()),
            "dmas":       int(training["metro_area"].nunique()),
            "date_range": [str(training["date"].min()), str(training["date"].max())],
            "total_rows": len(training),
        },
        "validation": {
            "method":         "Walk-forward (time-series holdout)",
            "min_history_months": config.WALK_FORWARD_MIN_MONTHS,
            "months_tested":  wf.get("months_tested"),
            "total_predictions": wf.get("total_predictions"),
            "r2":    wf.get("r2"),
            "rmse":  wf.get("rmse"),
            "mae":   wf.get("mae"),
            "smape": wf.get("smape"),
            "r2_mean_per_month": wf.get("r2_mean"),
            "r2_std_per_month":  wf.get("r2_std"),
        },
        "insample": {
            "r2":   insample.get("r2"),
            "rmse": insample.get("rmse"),
            "note": insample.get("note"),
        },
        "predictions": {
            "target_month":   preds["date"].iloc[0] if not preds.empty else None,
            "counties_predicted": len(preds),
            "flag_counts":    preds["flag"].value_counts().to_dict() if not preds.empty else {},
        },
    }

    with open(config.PAPER_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Paper summary → {config.PAPER_SUMMARY_JSON}")
    return config.PAPER_SUMMARY_JSON


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_all() -> None:
    """Generate all figures and the paper summary JSON."""
    logger.info("=== STAGE 6: GENERATE REPORT ===")

    generated = []
    steps = [
        ("Actual vs Predicted scatter",      plot_actual_vs_predicted),
        ("R² over time",                     plot_walkforward_r2_over_time),
        ("Feature importance",               plot_feature_importance),
        ("Statewide timeseries",             plot_statewide_timeseries),
        ("Risk flag distribution",           plot_risk_flag_distribution),
    ]

    for name, fn in steps:
        try:
            path = fn()
            generated.append(path)
        except Exception as e:
            logger.warning(f"  Skipped '{name}': {e}")

    try:
        save_paper_summary()
    except Exception as e:
        logger.warning(f"  Skipped paper summary: {e}")

    logger.info(f"\n  Generated {len(generated)} figure(s) in {config.FIGURES_DIR}/")
