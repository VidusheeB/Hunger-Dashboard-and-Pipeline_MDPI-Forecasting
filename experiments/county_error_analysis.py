"""
county_error_analysis.py — Per-county error analysis for the production model.

Runs walk-forward validation and collects a full (county × month) prediction
matrix.  For each county computes:

  MAE          mean absolute error on the SNAP application rate
  MAPE         mean absolute percentage error (|error| / actual × 100)
  MPE          mean percentage error  (signed: + = over-predict, - = under-predict)
  Bias         mean(predicted − actual) in rate units
  Variance     var(predicted − actual) — how consistent are the errors over time?
  Std          std(predicted − actual)
  Max abs err  worst single-month error
  n            number of months tested

Counties are ranked by MAE.  Best = lowest MAE, Worst = highest.

Outputs
-------
  outputs/metrics/county_error_analysis.csv     — full ranked table (all counties)
  outputs/metrics/county_error_best10.csv       — top 10 best-predicted
  outputs/metrics/county_error_worst10.csv      — top 10 worst-predicted
  outputs/figures/county_mae_ranked.png         — horizontal bar chart, all counties
  outputs/figures/county_bias_variance.png      — scatter: bias vs variance
  outputs/figures/county_error_heatmap.png      — heatmap: error by county × month

Run
---
  python experiments/county_error_analysis.py
  python experiments/county_error_analysis.py --model rf   # use a different model
  python experiments/county_error_analysis.py --no-plots
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(key: str):
    if key == "lr":           return LinearRegression()
    if key == "rf":           return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if key == "gb":           return GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                                learning_rate=0.05, random_state=42)
    if key == "xgb_default":  return xgb.XGBRegressor(random_state=42, n_jobs=-1)
    if key == "xgb_tuned":    return xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    raise ValueError(f"Unknown model key: {key}. "
                     "Choose from: lr, rf, gb, xgb_default, xgb_tuned")


# ── Walk-forward: collect per-county predictions ──────────────────────────────

def collect_predictions(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """
    Run walk-forward and return a row for every (county, month) prediction.

    Columns: county, month, actual, predicted, error, abs_error, pct_error
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    feature_cols = [f for f in config.FEATURE_COLS if f in df.columns]
    missing = set(config.FEATURE_COLS) - set(feature_cols)
    if missing:
        logger.warning(f"  Missing features (excluded): {missing}")

    model = make_model(model_key)
    rows  = []

    n_test = len(dates) - config.WALK_FORWARD_MIN_MONTHS
    logger.info(f"  Walk-forward: {n_test} test months, model={model_key}")

    for i, test_date in enumerate(dates[config.WALK_FORWARD_MIN_MONTHS:], 1):
        if i % 5 == 0 or i == 1:
            logger.info(f"  [{i}/{n_test}] {pd.Timestamp(test_date).strftime('%Y-%m')} ...")

        train_mask = df["date"] < test_date
        test_mask  = df["date"] == test_date

        train_df = df[train_mask]
        test_df  = df[test_mask]

        tr_ok = (train_df[feature_cols].notna().all(axis=1)
                 & train_df[config.TARGET_COL].notna())
        te_ok = (test_df[feature_cols].notna().all(axis=1)
                 & test_df[config.TARGET_COL].notna())

        if tr_ok.sum() < 10 or te_ok.sum() == 0:
            continue

        model.fit(
            train_df.loc[tr_ok, feature_cols],
            train_df.loc[tr_ok, config.TARGET_COL].clip(lower=0),
        )
        preds = np.clip(
            model.predict(test_df.loc[te_ok, feature_cols]), 0, None
        )

        for county, pred, actual in zip(
            test_df.loc[te_ok, "county"],
            preds,
            test_df.loc[te_ok, config.TARGET_COL].values,
        ):
            error = float(pred) - float(actual)
            # Percentage error: guard against near-zero actuals
            if abs(actual) > 1e-9:
                pct_error = error / actual * 100.0
            else:
                pct_error = np.nan

            rows.append({
                "county":    county,
                "month":     pd.Timestamp(test_date).strftime("%Y-%m"),
                "actual":    float(actual),
                "predicted": float(pred),
                "error":     error,           # signed: + = over-predict
                "abs_error": abs(error),
                "pct_error": pct_error,       # signed %: + = over-predict
            })

    return pd.DataFrame(rows)


# ── Aggregate per-county metrics ──────────────────────────────────────────────

def aggregate_county_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse (county, month) rows into one row per county with all metrics.
    Returns a DataFrame sorted by MAE ascending (best-predicted first).
    """
    rows = []
    for county, grp in pred_df.groupby("county"):
        errors    = grp["error"].values
        abs_errs  = grp["abs_error"].values
        pct_errs  = grp["pct_error"].dropna().values
        actuals   = grp["actual"].values

        mae   = float(np.mean(abs_errs))
        bias  = float(np.mean(errors))           # mean signed error
        var_e = float(np.var(errors, ddof=1)) if len(errors) > 1 else 0.0
        std_e = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

        mape  = float(np.mean(np.abs(pct_errs))) if len(pct_errs) > 0 else np.nan
        mpe   = float(np.mean(pct_errs))          if len(pct_errs) > 0 else np.nan

        rows.append({
            "county":      county,
            "n_months":    len(grp),
            "mae":         round(mae,   8),
            "mape":        round(mape,  4) if not np.isnan(mape) else np.nan,
            "mpe":         round(mpe,   4) if not np.isnan(mpe)  else np.nan,
            "bias":        round(bias,  8),
            "variance":    round(var_e, 12),
            "std":         round(std_e, 8),
            "max_abs_err": round(float(np.max(abs_errs)), 8),
            "mean_actual": round(float(np.mean(actuals)), 8),
            # Direction label for easy reading
            "bias_direction": "over" if bias > 0 else ("under" if bias < 0 else "neutral"),
        })

    return (pd.DataFrame(rows)
              .sort_values("mae")
              .reset_index(drop=True))


# ── Print summary tables ───────────────────────────────────────────────────────

def print_ranked_table(county_df: pd.DataFrame) -> None:
    print("\n" + "=" * 105)
    print(f"  COUNTY ERROR ANALYSIS — Ranked by MAE (best → worst)  [{len(county_df)} counties]")
    print("=" * 105)
    hdr = (f"  {'County':<20} {'MAE':>10} {'MAPE%':>7} {'MPE%':>7} "
           f"{'Bias':>10} {'Std':>10} {'Variance':>12} {'Direction':<9} {'N':>3}")
    print(hdr)
    print("-" * 105)

    for i, row in county_df.iterrows():
        mape_s = f"{row['mape']:>6.2f}%" if not pd.isna(row["mape"]) else "     —"
        mpe_s  = f"{row['mpe']:>+6.2f}%" if not pd.isna(row["mpe"])  else "     —"
        dir_s  = ("▲ over" if row["bias_direction"] == "over"
                  else ("▼ under" if row["bias_direction"] == "under" else "  —"))
        print(
            f"  {row['county']:<20} {row['mae']:>10.6f} {mape_s:>7} {mpe_s:>7} "
            f"{row['bias']:>+10.6f} {row['std']:>10.6f} {row['variance']:>12.2e} "
            f"{dir_s:<9} {int(row['n_months']):>3}"
        )
    print("=" * 105)


def print_best_worst(county_df: pd.DataFrame, n: int = 10) -> None:
    best  = county_df.head(n)
    worst = county_df.tail(n).sort_values("mae", ascending=False)

    for label, sub in [("BEST", best), ("WORST", worst)]:
        print(f"\n  ── {label}-PREDICTED {n} COUNTIES ──")
        print(f"  {'County':<20} {'MAE':>10} {'MAPE%':>8} {'Bias':>11} {'Direction'}")
        print("  " + "-" * 60)
        for _, row in sub.iterrows():
            mape_s = f"{row['mape']:.2f}%" if not pd.isna(row["mape"]) else "—"
            print(f"  {row['county']:<20} {row['mae']:>10.6f} {mape_s:>8} "
                  f"{row['bias']:>+11.6f}  {row['bias_direction']}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_mae_ranked(county_df: pd.DataFrame, model_key: str, out_path: str) -> None:
    """Horizontal bar chart of per-county MAE, coloured by bias direction."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    df = county_df.sort_values("mae", ascending=True).copy()

    # Colour by bias direction
    colors = ["#F44336" if d == "over" else "#2196F3" if d == "under" else "#9E9E9E"
              for d in df["bias_direction"]]

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.22)))
    bars = ax.barh(df["county"], df["mae"], color=colors, alpha=0.85, edgecolor="white")

    # Median line
    median_mae = df["mae"].median()
    ax.axvline(median_mae, color="black", linewidth=1.2, linestyle="--", alpha=0.6,
               label=f"Median MAE = {median_mae:.5f}")

    ax.set_xlabel("MAE (SNAP application rate)", fontsize=10)
    ax.set_title(f"County MAE — {model_key} walk-forward\n"
                 "Red = over-predicts  ·  Blue = under-predicts", fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_bias_variance(county_df: pd.DataFrame, model_key: str, out_path: str) -> None:
    """
    Scatter plot: bias (x) vs std of errors (y).

    Quadrant interpretation:
      Q1 (+bias, high std): unpredictably over-predicts   → high risk counties
      Q2 (-bias, high std): unpredictably under-predicts  → high miss risk
      Q3 (-bias, low std):  consistently under-predicts   → systematic gap
      Q4 (+bias, low std):  consistently over-predicts    → systematic inflation
    """
    import matplotlib.pyplot as plt

    df = county_df.copy()
    # Scale variance for bubble size
    size = 40 + (df["mae"] / df["mae"].max()) * 300

    fig, ax = plt.subplots(figsize=(11, 7))

    scatter = ax.scatter(
        df["bias"], df["std"],
        c=df["mae"], cmap="YlOrRd",
        s=size, alpha=0.8, edgecolors="white", linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="MAE", shrink=0.8)

    # Label the 10 worst counties (by MAE)
    worst = df.nlargest(10, "mae")
    for _, row in worst.iterrows():
        ax.annotate(
            row["county"],
            (row["bias"], row["std"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=7.5, color="#333333",
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(df["std"].median(), color="gray", linewidth=0.8, linestyle=":",
               alpha=0.6, label=f"Median std = {df['std'].median():.5f}")

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    mid_x = 0; mid_y = df["std"].median()
    ax.text(xlim[1] * 0.6, ylim[1] * 0.92, "high std\nover-predicts",
            ha="center", fontsize=8, color="#B71C1C", alpha=0.7)
    ax.text(xlim[0] * 0.6, ylim[1] * 0.92, "high std\nunder-predicts",
            ha="center", fontsize=8, color="#1A237E", alpha=0.7)
    ax.text(xlim[1] * 0.6, ylim[0] + (ylim[1]-ylim[0])*0.05, "consistent\nover-predict",
            ha="center", fontsize=8, color="#E64A19", alpha=0.7)
    ax.text(xlim[0] * 0.6, ylim[0] + (ylim[1]-ylim[0])*0.05, "consistent\nunder-predict",
            ha="center", fontsize=8, color="#283593", alpha=0.7)

    ax.set_xlabel("Bias = mean(predicted − actual)  [+ = over-predict]", fontsize=10)
    ax.set_ylabel("Std of errors (month-to-month consistency)", fontsize=10)
    ax.set_title(f"Bias vs Error Variance by County — {model_key}\n"
                 "Bubble size = MAE  ·  Colour = MAE  ·  Top-10 worst labelled",
                 fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_error_heatmap(pred_df: pd.DataFrame, county_df: pd.DataFrame,
                       model_key: str, out_path: str) -> None:
    """
    Heatmap: signed error (predicted − actual) for each county × month.
    Counties sorted by MAE (worst at top).  Reveals temporal patterns and
    whether errors cluster in particular months.
    """
    import matplotlib.pyplot as plt

    # Pivot to county × month matrix of signed errors
    pivot = pred_df.pivot_table(
        index="county", columns="month", values="error", aggfunc="mean"
    )
    # Sort counties by MAE (worst at top for visual prominence)
    county_order = county_df.sort_values("mae", ascending=False)["county"].tolist()
    pivot = pivot.reindex([c for c in county_order if c in pivot.index])

    # Symmetric colour scale
    abs_max = np.nanpercentile(np.abs(pivot.values), 95)

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.6),
                                    max(10, len(pivot) * 0.25)))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdBu_r",
        vmin=-abs_max, vmax=abs_max,
    )
    plt.colorbar(im, ax=ax, label="Signed error (predicted − actual)", shrink=0.6)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7.5)
    ax.set_title(
        f"Error Heatmap — {model_key} walk-forward\n"
        "Red = over-predict  ·  Blue = under-predict  ·  Counties sorted worst→best (top→bottom)",
        fontsize=11,
    )
    ax.set_xlabel("Test month", fontsize=10)
    ax.set_ylabel("County (sorted by MAE, worst at top)", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Per-county error analysis")
    p.add_argument("--model",     type=str, default="xgb_tuned",
                   help="Model to analyse: lr, rf, gb, xgb_default, xgb_tuned (default)")
    p.add_argument("--no-plots",  action="store_true")
    p.add_argument("--data",      type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config.ensure_output_dirs()

    data_path = args.data or config.FEATURES_CSV
    if not os.path.exists(data_path):
        data_path = config.TRAINING_DATA_CSV
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data at {data_path}. Run stages 2+25 first.")

    df = pd.read_csv(data_path)
    logger.info(f"  Loaded {data_path}  shape={df.shape}")

    # ── Walk-forward predictions ──────────────────────────────────────────────
    logger.info(f"\n  Running walk-forward for model: {args.model}")
    pred_df = collect_predictions(df, args.model)
    logger.info(f"  Collected {len(pred_df):,} county-month predictions "
                f"across {pred_df['county'].nunique()} counties and "
                f"{pred_df['month'].nunique()} months")

    # ── Aggregate per county ──────────────────────────────────────────────────
    county_df = aggregate_county_metrics(pred_df)
    logger.info(f"  Aggregated metrics for {len(county_df)} counties")

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics_dir = os.path.join(config.OUTPUTS_ROOT, "metrics")
    out_full    = os.path.join(metrics_dir, f"county_error_analysis_{args.model}.csv")
    out_best    = os.path.join(metrics_dir, f"county_error_best10_{args.model}.csv")
    out_worst   = os.path.join(metrics_dir, f"county_error_worst10_{args.model}.csv")

    county_df.to_csv(out_full, index=False)
    county_df.head(10).to_csv(out_best, index=False)
    county_df.tail(10).sort_values("mae", ascending=False).to_csv(out_worst, index=False)

    logger.info(f"  Full table  → {out_full}")
    logger.info(f"  Best-10     → {out_best}")
    logger.info(f"  Worst-10    → {out_worst}")

    # ── Print ─────────────────────────────────────────────────────────────────
    print_ranked_table(county_df)
    print_best_worst(county_df, n=10)

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_over  = (county_df["bias_direction"] == "over").sum()
    n_under = (county_df["bias_direction"] == "under").sum()
    print(f"\n  Bias direction summary across {len(county_df)} counties:")
    print(f"    Over-predicted:  {n_over}  ({n_over/len(county_df)*100:.0f}%)")
    print(f"    Under-predicted: {n_under} ({n_under/len(county_df)*100:.0f}%)")
    print(f"    Median MAE:  {county_df['mae'].median():.6f}")
    print(f"    Median MAPE: {county_df['mape'].median():.2f}%")
    print(f"    MAE range:   [{county_df['mae'].min():.6f}, {county_df['mae'].max():.6f}]")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        fig_dir = config.FIGURES_DIR
        os.makedirs(fig_dir, exist_ok=True)

        plot_mae_ranked(county_df, args.model,
                        os.path.join(fig_dir, f"county_mae_ranked_{args.model}.png"))
        plot_bias_variance(county_df, args.model,
                           os.path.join(fig_dir, f"county_bias_variance_{args.model}.png"))
        plot_error_heatmap(pred_df, county_df, args.model,
                           os.path.join(fig_dir, f"county_error_heatmap_{args.model}.png"))

    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
