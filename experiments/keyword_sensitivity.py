"""
keyword_sensitivity.py — Google Trends keyword selection sensitivity analysis.

Tests 13 keyword configurations under identical walk-forward validation to
determine which Trends terms (if any) improve SNAP application rate prediction.

Keyword sets tested
-------------------
  no_trends       — no Trends features (AR-only baseline)
  calfresh        — "CalFresh" only
  foodbank        — "food bank" only
  cf_fb           — CalFresh + food bank  (closest to production)
  food_stamps     — "food stamps" only
  ebt_card        — "EBT card" only
  snap_benefits   — "SNAP benefits" only
  food_pantry     — "food pantry near me" only
  apply_calfresh  — "apply for CalFresh" only
  snap_program    — "Supplemental Nutrition Assistance Program" only
  awareness       — food stamps + food bank + food pantry  (broad food insecurity)
  intent          — CalFresh + apply for CalFresh + SNAP benefits + SNAP program
  all_8           — all 8 terms from AllTerms download

Source: AllTerms CSVs (src/data/trends/AllTerms/) provide all 8 terms from a
single comparative Google Trends download, ensuring consistent cross-term
normalisation. Individual CalFresh/FoodBank CSVs are NOT used here so that
all comparisons are made on the same scale.

Output
------
  outputs/metrics/keyword_sensitivity.csv
  outputs/figures/keyword_sensitivity_mae.png
  outputs/figures/keyword_sensitivity_r2.png
  outputs/figures/keyword_sensitivity_panel.png
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from pipeline import config

ALLTERMS_DIR   = os.path.join(config.RAW_DATA_ROOT, "trends", "AllTerms")
TRAINING_CSV   = config.TRAINING_DATA_CSV   # has SNAP + county + pop + income + CalFresh/FoodBank
FEATURES_CSV   = config.FEATURES_CSV        # for fixed non-Trends features
METRICS_OUT    = os.path.join(config.OUTPUTS_ROOT, "metrics", "keyword_sensitivity.csv")
FIGURES_DIR    = config.FIGURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "Helvetica",
    "font.size":           10,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
})

# ── Term definitions ───────────────────────────────────────────────────────────
# Maps safe column names → exact header string in AllTerms CSVs
TERM_COLS = {
    "food_stamps":    "food stamps",
    "ebt_card":       "EBT card",
    "snap_benefits":  "SNAP benefits",
    "food_pantry":    "food pantry near me",
    "food_bank":      "food bank",
    "calfresh":       "CalFresh",
    "apply_calfresh": "apply for CalFresh",
    "snap_program":   "Supplemental Nutrition Assistance Program",
}

# ── Keyword set configurations ─────────────────────────────────────────────────
KEYWORD_SETS = {
    "no_trends":      [],
    "calfresh":       ["calfresh"],
    "foodbank":       ["food_bank"],
    "cf_fb":          ["calfresh", "food_bank"],
    "food_stamps":    ["food_stamps"],
    "ebt_card":       ["ebt_card"],
    "snap_benefits":  ["snap_benefits"],
    "food_pantry":    ["food_pantry"],
    "apply_calfresh": ["apply_calfresh"],
    "snap_program":   ["snap_program"],
    "awareness":      ["food_stamps", "food_bank", "food_pantry"],
    "intent":         ["calfresh", "apply_calfresh", "snap_benefits", "snap_program"],
    "all_8":          list(TERM_COLS.keys()),
}

# Human-readable labels for plots
LABELS = {
    "no_trends":      "No Trends\n(AR baseline)",
    "calfresh":       "CalFresh\nonly",
    "foodbank":       "Food Bank\nonly",
    "cf_fb":          "CalFresh +\nFood Bank\n(production)",
    "food_stamps":    "Food Stamps\nonly",
    "ebt_card":       "EBT Card\nonly",
    "snap_benefits":  "SNAP Benefits\nonly",
    "food_pantry":    "Food Pantry\nonly",
    "apply_calfresh": "Apply CalFresh\nonly",
    "snap_program":   "SNAP Program\nonly",
    "awareness":      "Awareness\n(stamps+bank+pantry)",
    "intent":         "Intent\n(CF+applyCF+SNAP*2)",
    "all_8":          "All 8 Terms",
}

# ── Fixed (non-Trends) features ────────────────────────────────────────────────
# These are included in every configuration
FIXED_FEATURES = [
    "rate_lag1", "rate_lag2", "rate_lag3",
    "rate_roll3_mean", "rate_roll3_std",
    "month", "month_sin", "month_cos", "quarter",
    "Population", "Median_Income",
    "log_population", "log_income", "income_quintile",
]

TARGET = config.TARGET_COL   # "SNAP_Application_Rate"
MIN_TRAIN_MONTHS = config.WALK_FORWARD_MIN_MONTHS  # 12


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load AllTerms CSVs
# ══════════════════════════════════════════════════════════════════════════════

def load_allterms() -> pd.DataFrame:
    """
    Load every AllTerms CSV file, parse all 8 term columns, and return a
    long-format monthly DataFrame with columns:
        metro_area, date, <safe_term_name>, ...

    DMA name normalisation: "San Diego.csv" → "SanDiego" (remove spaces) to
    match the metro_area values already in training_data.csv.
    """
    import glob

    files = glob.glob(os.path.join(ALLTERMS_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No AllTerms CSVs found in {ALLTERMS_DIR}")

    # Reverse map: header string → safe column name
    header_to_safe = {v: k for k, v in TERM_COLS.items()}

    dfs = []
    for fpath in files:
        # Normalise DMA name (strip spaces)
        metro = os.path.splitext(os.path.basename(fpath))[0].replace(" ", "")

        raw = pd.read_csv(fpath, parse_dates=["Time"])
        raw = raw.rename(columns={"Time": "date"})

        # Keep only the term columns we know about
        keep = {col: header_to_safe[col] for col in raw.columns if col in header_to_safe}
        if not keep:
            logger.warning(f"  {metro}: no recognised term columns — skipping")
            continue

        raw = raw[["date"] + list(keep.keys())].rename(columns=keep)
        raw["metro_area"] = metro

        # Weekly → monthly average per DMA
        raw["year"]  = raw["date"].dt.year
        raw["month_num"] = raw["date"].dt.month
        monthly = (
            raw.groupby(["metro_area", "year", "month_num"])
            [list(keep.values())]
            .mean()
            .reset_index()
        )
        monthly["date"] = pd.to_datetime(
            monthly[["year", "month_num"]].rename(columns={"month_num": "month"}).assign(day=1)
        )
        monthly = monthly.drop(columns=["year", "month_num"])
        dfs.append(monthly)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"  AllTerms: {len(files)} DMAs, "
        f"{combined['metro_area'].nunique()} unique DMAs, "
        f"{combined['date'].min().date()} – {combined['date'].max().date()}"
    )
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 2. Build base DataFrame with fixed non-Trends features
# ══════════════════════════════════════════════════════════════════════════════

def build_base(allterms: pd.DataFrame) -> pd.DataFrame:
    """
    Start from features.csv (which has all engineered fixed features already),
    join the AllTerms monthly data on (metro_area, date), and return the merged
    frame that all keyword configurations will draw from.

    features.csv date = SNAP month; AllTerms CSVs represent the *previous* month
    (same 1-month lag baked into training_data.csv via trend_date). So we join
    AllTerms on date-1-month.
    """
    logger.info("  Loading features.csv...")
    feat = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    # allterms date represents the Trends month, which maps to SNAP date + 1 month
    # (i.e. trend from month T predicts SNAP in month T+1).
    # In features.csv, training_data was built with trend_date = snap_date - 1 month.
    # So to join: allterms.date = feat.date - 1 month → allterms.date = feat.trend_date
    # We shift allterms forward 1 month to align with the SNAP month.
    allterms = allterms.copy()
    allterms["snap_date"] = allterms["date"] + pd.DateOffset(months=1)
    allterms = allterms.drop(columns=["date"]).rename(columns={"snap_date": "date"})

    merged = feat.merge(allterms, on=["metro_area", "date"], how="left")
    logger.info(
        f"  Base after join: {merged.shape}  "
        f"({merged['county'].nunique()} counties, {merged['date'].nunique()} months)"
    )

    # Check coverage
    term_cols = list(TERM_COLS.keys())
    missing = merged[term_cols].isna().mean()
    for col, pct in missing[missing > 0].items():
        logger.info(f"    {col}: {pct:.1%} NaN (DMA coverage gap)")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build Trends features for a single keyword
# ══════════════════════════════════════════════════════════════════════════════

def add_trends_features(df: pd.DataFrame, keywords: list) -> tuple[pd.DataFrame, list]:
    """
    For each keyword in the list, add: current, lag1, lag2, roll3, momentum.
    Returns (df_with_features, list_of_new_feature_names).

    All lags and rolling windows are computed within each county group, sorted
    by date, to prevent any look-ahead leakage.
    """
    df = df.sort_values(["county", "date"]).copy()
    new_cols = []

    for kw in keywords:
        col = kw  # the safe column name is directly in df after the join

        lag1 = f"{kw}_lag1"
        lag2 = f"{kw}_lag2"
        roll = f"{kw}_roll3"
        mom  = f"{kw}_momentum"

        df[lag1] = df.groupby("county")[col].shift(1)
        df[lag2] = df.groupby("county")[col].shift(2)
        df[roll] = (
            df.groupby("county")[col]
            .transform(lambda s: s.rolling(window=3, min_periods=2).mean())
        )
        df[mom]  = df[col] - df[lag1]

        new_cols.extend([col, lag1, lag2, roll, mom])

    return df, new_cols


# ══════════════════════════════════════════════════════════════════════════════
# 4. Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Walk-forward validation with XGBoost tuned hyperparameters.

    For each month T (starting after MIN_TRAIN_MONTHS of history):
      - Train on all rows where date < T
      - Predict on rows where date == T
    Aggregates MAE, RMSE, R² over all test predictions.

    Returns: dict with mae, rmse, r2, n_test
    """
    df = df.sort_values(["county", "date"]).copy()

    # Drop rows missing any required feature or target
    required = feature_cols + [TARGET]
    df = df.dropna(subset=required).reset_index(drop=True)

    months = sorted(df["date"].unique())
    if len(months) <= MIN_TRAIN_MONTHS:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_test": 0}

    model = XGBRegressor(**config.XGBOOST_PARAMS, verbosity=0)

    all_true, all_pred = [], []

    for i, test_month in enumerate(months[MIN_TRAIN_MONTHS:], start=MIN_TRAIN_MONTHS):
        train = df[df["date"] < test_month]
        test  = df[df["date"] == test_month]

        if len(train) < 10 or len(test) == 0:
            continue

        model.fit(train[feature_cols], train[TARGET])
        pred = np.clip(model.predict(test[feature_cols]), 0, None)

        all_true.extend(test[TARGET].values)
        all_pred.extend(pred)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(y_true)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Run all configurations
# ══════════════════════════════════════════════════════════════════════════════

def run_all(base: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over every keyword set, build features, run walk-forward, record results.
    Returns a DataFrame with one row per configuration.
    """
    results = []
    n_sets = len(KEYWORD_SETS)

    for idx, (name, keywords) in enumerate(KEYWORD_SETS.items(), start=1):
        logger.info(f"  [{idx}/{n_sets}] {name}  keywords={keywords or ['(none)']}")

        # Build Trends features for this configuration
        df_cfg, trend_feat_cols = add_trends_features(base, keywords)

        # Final feature list: fixed + trends-specific
        feature_cols = FIXED_FEATURES + trend_feat_cols

        # Run walk-forward
        metrics = walk_forward(df_cfg, feature_cols)

        results.append({
            "config":     name,
            "label":      LABELS[name],
            "n_keywords": len(keywords),
            "n_features": len(feature_cols),
            "keywords":   ", ".join(keywords) if keywords else "(none)",
            "mae":        metrics["mae"],
            "rmse":       metrics["rmse"],
            "r2":         metrics["r2"],
            "n_test":     metrics["n_test"],
        })

        logger.info(
            f"    → MAE={metrics['mae']:.6f}  R²={metrics['r2']:.4f}  "
            f"features={len(feature_cols)}"
        )

    df_results = pd.DataFrame(results)
    return df_results


# ══════════════════════════════════════════════════════════════════════════════
# 6. Print results table
# ══════════════════════════════════════════════════════════════════════════════

def print_results(df: pd.DataFrame) -> None:
    """Print a ranked summary table to stdout."""
    df = df.sort_values("mae").copy()
    df["rank"] = range(1, len(df) + 1)

    # Baseline (no_trends) for delta calculations
    baseline_mae = df.loc[df["config"] == "no_trends", "mae"].values[0]
    df["delta_mae"] = df["mae"] - baseline_mae
    df["delta_pct"]  = (df["delta_mae"] / baseline_mae * 100).round(2)

    print("\n" + "=" * 100)
    print("  KEYWORD SENSITIVITY ANALYSIS — Walk-Forward Results (ranked by MAE)")
    print("=" * 100)
    header = f"  {'Rank':>4}  {'Config':<16}  {'Keywords':<42}  {'N feats':>7}  "
    header += f"{'MAE':>10}  {'R²':>7}  {'ΔMae vs AR':>12}  {'Δ%':>7}"
    print(header)
    print("  " + "─" * 96)

    for _, row in df.iterrows():
        delta_str = f"{row['delta_mae']:+.6f}" if not np.isnan(row["delta_mae"]) else "  —"
        pct_str   = f"{row['delta_pct']:+.1f}%" if not np.isnan(row["delta_pct"]) else "  —"
        marker = " ◄ production" if row["config"] == "cf_fb" else ""
        marker += " ◄ AR baseline" if row["config"] == "no_trends" else ""
        print(
            f"  {int(row['rank']):>4}  {row['config']:<16}  {row['keywords']:<42}  "
            f"{int(row['n_features']):>7}  "
            f"{row['mae']:.6f}  {row['r2']:.4f}  {delta_str:>12}  {pct_str:>7}"
            f"{marker}"
        )

    print("=" * 100)
    best = df.iloc[0]
    print(f"\n  Best config: {best['config']} (MAE={best['mae']:.6f}, R²={best['r2']:.4f})")

    n_beat_baseline = (df[df["config"] != "no_trends"]["delta_mae"] < 0).sum()
    print(f"  Configs that beat the AR baseline: {n_beat_baseline} / {len(df) - 1}")
    n_beat_prod = (df[df["config"] != "cf_fb"]["mae"] < df.loc[df["config"] == "cf_fb", "mae"].values[0]).sum()
    print(f"  Configs that beat cf_fb (production): {n_beat_prod}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plots
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "no_trends":      "#888888",
    "calfresh":       "#1f77b4",
    "foodbank":       "#ff7f0e",
    "cf_fb":          "#2ca02c",
    "food_stamps":    "#d62728",
    "ebt_card":       "#9467bd",
    "snap_benefits":  "#8c564b",
    "food_pantry":    "#e377c2",
    "apply_calfresh": "#17becf",
    "snap_program":   "#bcbd22",
    "awareness":      "#aec7e8",
    "intent":         "#98df8a",
    "all_8":          "#ffbb78",
}


def _bar_chart(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: str,
               higher_is_better: bool = False) -> None:
    """Generic horizontal bar chart for one metric, sorted best → worst."""
    df_sorted = df.sort_values(metric, ascending=not higher_is_better).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(df_sorted))
    colors = [COLORS.get(c, "#aaaaaa") for c in df_sorted["config"]]

    bars = ax.barh(y_pos, df_sorted[metric], color=colors, height=0.6, edgecolor="white")

    # Value labels on bars
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(
            bar.get_width() + (bar.get_width() * 0.005 if higher_is_better else -bar.get_width() * 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}" if metric == "mae" else f"{val:.4f}",
            va="center", ha="left" if True else "right",
            fontsize=8.5,
        )

    # Reference line for no_trends baseline
    baseline_val = df.loc[df["config"] == "no_trends", metric].values[0]
    ax.axvline(baseline_val, color="#888888", linestyle="--", linewidth=1.2,
               label="No Trends baseline")

    # Reference line for cf_fb (production)
    prod_val = df.loc[df["config"] == "cf_fb", metric].values[0]
    ax.axvline(prod_val, color="#2ca02c", linestyle=":", linewidth=1.5,
               label="Production (cf_fb)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["config"], fontsize=9)
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.margins(x=0.12)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_panel(df: pd.DataFrame, out_path: str) -> None:
    """2-panel figure: MAE (left) + R² (right), both ranked by MAE."""
    df_sorted = df.sort_values("mae").copy()
    n = len(df_sorted)
    y_pos = np.arange(n)
    colors = [COLORS.get(c, "#aaaaaa") for c in df_sorted["config"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: MAE ──────────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(y_pos, df_sorted["mae"], color=colors, height=0.6, edgecolor="white")
    for bar, val in zip(bars, df_sorted["mae"]):
        ax.text(bar.get_width() + df_sorted["mae"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=8)
    ax.axvline(df.loc[df["config"] == "no_trends", "mae"].values[0],
               color="#888888", linestyle="--", linewidth=1.2, label="AR baseline")
    ax.axvline(df.loc[df["config"] == "cf_fb", "mae"].values[0],
               color="#2ca02c", linestyle=":", linewidth=1.5, label="Production (cf_fb)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["config"], fontsize=9)
    ax.set_xlabel("Walk-Forward MAE", fontsize=10)
    ax.set_title("Mean Absolute Error\n(lower = better)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.margins(x=0.14)

    # ── Right: R² ──────────────────────────────────────────────────────────────
    ax = axes[1]
    # Keep same row order as MAE plot (best MAE = lowest bar)
    bars = ax.barh(y_pos, df_sorted["r2"], color=colors, height=0.6, edgecolor="white")
    for bar, val in zip(bars, df_sorted["r2"]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.axvline(df.loc[df["config"] == "no_trends", "r2"].values[0],
               color="#888888", linestyle="--", linewidth=1.2, label="AR baseline")
    ax.axvline(df.loc[df["config"] == "cf_fb", "r2"].values[0],
               color="#2ca02c", linestyle=":", linewidth=1.5, label="Production (cf_fb)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["config"], fontsize=9)
    ax.set_xlabel("Walk-Forward R²", fontsize=10)
    ax.set_title("R-Squared\n(higher = better)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.margins(x=0.08)

    fig.suptitle("Google Trends Keyword Sensitivity Analysis — Walk-Forward Validation",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_delta(df: pd.DataFrame, out_path: str) -> None:
    """Bar chart of ΔMae relative to AR baseline — shows which sets help/hurt."""
    baseline_mae = df.loc[df["config"] == "no_trends", "mae"].values[0]
    df = df[df["config"] != "no_trends"].copy()
    df["delta_mae"] = df["mae"] - baseline_mae
    df = df.sort_values("delta_mae")  # most helpful first (most negative)

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(df))
    colors_list = ["#2ca02c" if v < 0 else "#d62728" for v in df["delta_mae"]]
    bars = ax.barh(y_pos, df["delta_mae"], color=colors_list, height=0.6, edgecolor="white")
    for bar, val in zip(bars, df["delta_mae"]):
        offset = df["delta_mae"].abs().max() * 0.01
        ax.text(bar.get_width() + (offset if val >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.6f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8)
    prod_delta = df.loc[df["config"] == "cf_fb", "delta_mae"].values[0] if "cf_fb" in df["config"].values else None
    if prod_delta is not None:
        ax.axvline(prod_delta, color="#2ca02c", linestyle=":", linewidth=1.5, label="Production (cf_fb)")
        ax.legend(fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["config"], fontsize=9)
    ax.set_xlabel("ΔMae vs AR Baseline (negative = improvement)", fontsize=10)
    ax.set_title("Keyword Set Effect on Walk-Forward MAE\nRelative to No-Trends Baseline",
                 fontsize=12, fontweight="bold", pad=10)
    ax.margins(x=0.14)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Google Trends keyword sensitivity analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    config.ensure_output_dirs()

    logger.info("=== KEYWORD SENSITIVITY ANALYSIS ===\n")

    # Load AllTerms data
    logger.info("Loading AllTerms CSVs...")
    allterms = load_allterms()

    # Build base DataFrame with fixed features + AllTerms columns
    logger.info("\nBuilding base DataFrame...")
    base = build_base(allterms)

    # Check which terms are actually available
    available_terms = [t for t in TERM_COLS.keys() if t in base.columns]
    missing_terms   = [t for t in TERM_COLS.keys() if t not in base.columns]
    logger.info(f"  Available terms: {available_terms}")
    if missing_terms:
        logger.warning(f"  Missing terms (will skip): {missing_terms}")

    # Filter KEYWORD_SETS to only include available terms
    def _filter(kws):
        return [k for k in kws if k in available_terms]

    filtered_sets = {
        name: _filter(kws)
        for name, kws in KEYWORD_SETS.items()
    }

    # Run all configurations
    logger.info("\nRunning walk-forward for each keyword configuration...")
    results = run_all(base)

    # Save results
    results.to_csv(METRICS_OUT, index=False)
    logger.info(f"\n  Results → {METRICS_OUT}")

    # Print table
    print_results(results)

    # Plots
    if not args.no_plots:
        logger.info("  Generating plots...")
        _bar_chart(
            results, "mae", "Walk-Forward MAE", "Keyword Sensitivity — MAE (lower = better)",
            os.path.join(FIGURES_DIR, "keyword_sensitivity_mae.png"),
            higher_is_better=False,
        )
        _bar_chart(
            results, "r2", "Walk-Forward R²", "Keyword Sensitivity — R² (higher = better)",
            os.path.join(FIGURES_DIR, "keyword_sensitivity_r2.png"),
            higher_is_better=True,
        )
        plot_delta(results, os.path.join(FIGURES_DIR, "keyword_sensitivity_delta.png"))
        plot_panel(results, os.path.join(FIGURES_DIR, "keyword_sensitivity_panel.png"))

    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
