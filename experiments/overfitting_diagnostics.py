"""
overfitting_diagnostics.py — Diagnose overfitting by comparing in-sample,
                              cross-validation, and walk-forward performance.

For each model, three performance estimates are computed:

  In-sample    — fit on ALL training data, score on THE SAME data.
                 This is the upper bound on apparent performance and captures
                 how much the model memorises training data.

  5-fold CV    — score on randomly held-out folds.
                 Time-INCORRECT for panel time-series, but included to show
                 its optimism bias vs walk-forward.

  Walk-forward — score on future-only months (time-correct).
                 The only honest generalisation estimate.

Overfitting signal:
  gap_insample_to_wf  = in-sample MAE − walk-forward MAE   (negative = overfit)
  gap_cv_to_wf        = CV MAE − walk-forward MAE           (negative = CV over-optimistic)

Additionally produces learning curves: walk-forward MAE as a function of the
number of training months available.  A curve that is still falling at the
right edge suggests more data would help; a flat curve means diminishing returns.

Outputs
-------
  outputs/metrics/overfitting_diagnostics.csv
  outputs/figures/overfitting_comparison.png   — grouped bar: in-sample/CV/WF MAE
  outputs/figures/overfitting_gap.png          — heatmap + bar: gap magnitude
  outputs/figures/learning_curves.png          — MAE vs training months

Run
---
  python experiments/overfitting_diagnostics.py
  python experiments/overfitting_diagnostics.py --no-plots
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

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

# ── Model registry (same as benchmark_models.py) ─────────────────────────────
# ARIMA/SARIMAX excluded — no meaningful in-sample vs WF comparison for
# per-county univariate models.

MODELS = {
    "naive":       "Naive (last-month)",
    "lr":          "Linear Regression",
    "rf":          "Random Forest",
    "gb":          "Gradient Boosting",
    "xgb_default": "XGBoost (default)",
    "xgb_tuned":   "XGBoost (tuned)",
}

def make_model(key: str):
    if key == "lr":          return LinearRegression()
    if key == "rf":          return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if key == "gb":          return GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                               learning_rate=0.05, random_state=42)
    if key == "xgb_default": return xgb.XGBRegressor(random_state=42, n_jobs=-1)
    if key == "xgb_tuned":   return xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    return None


def smape(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    denom  = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2 * np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)) * 100)


def metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return {
        "r2":    float(r2_score(y_true, y_pred)),
        "rmse":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
    }


# ── 1. In-sample metrics ──────────────────────────────────────────────────────

def compute_insample(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Fit each model on the full dataset and score on the same data.

    This intentionally uses the same rows for training and testing —
    that is the definition of in-sample performance.  The gap between
    this and walk-forward MAE is the overfitting diagnostic.
    """
    mask = df[feature_cols].notna().all(axis=1) & df[config.TARGET_COL].notna()
    X = df.loc[mask, feature_cols].values
    y = df.loc[mask, config.TARGET_COL].clip(lower=0).values

    results = {}

    # Naive: predict mean of each county's history (best in-sample naive)
    # For fairness, we use the previous month (same as WF naive)
    naive_df = df[mask].copy().sort_values(["county", "date"])
    naive_df["naive_pred"] = naive_df.groupby("county")[config.TARGET_COL].shift(1)
    naive_clean = naive_df.dropna(subset=["naive_pred"])
    if not naive_clean.empty:
        results["naive"] = metrics(
            naive_clean[config.TARGET_COL].values,
            naive_clean["naive_pred"].clip(lower=0).values,
        )
        logger.info(f"  In-sample  naive        MAE={results['naive']['mae']:.6f}")

    for key in ["lr", "rf", "gb", "xgb_default", "xgb_tuned"]:
        m = make_model(key)
        m.fit(X, y)
        preds = m.predict(X)
        results[key] = metrics(y, preds)
        logger.info(f"  In-sample  {key:<12} MAE={results[key]['mae']:.6f}  R²={results[key]['r2']:.4f}")

    return results


# ── 2. 5-fold CV metrics ──────────────────────────────────────────────────────

def compute_kfold(df: pd.DataFrame, feature_cols: list, n_folds: int = 5) -> dict:
    """5-fold CV — time-incorrect, included only to show its optimism bias."""
    mask = df[feature_cols].notna().all(axis=1) & df[config.TARGET_COL].notna()
    X = df.loc[mask, feature_cols].values
    y = df.loc[mask, config.TARGET_COL].clip(lower=0).values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for key in ["lr", "rf", "gb", "xgb_default", "xgb_tuned"]:
        all_true, all_pred = [], []
        m = make_model(key)
        for tr_idx, te_idx in kf.split(X):
            m.fit(X[tr_idx], y[tr_idx])
            all_pred.extend(m.predict(X[te_idx]).tolist())
            all_true.extend(y[te_idx].tolist())
        results[key] = metrics(np.array(all_true), np.array(all_pred))

    # Naive has no meaningful CV formulation — skip
    return results


# ── 3. Walk-forward metrics ───────────────────────────────────────────────────

def load_walkforward() -> dict:
    """Load pre-computed walk-forward results from benchmark_models.py output."""
    path = os.path.join(config.OUTPUTS_ROOT, "metrics", "benchmark_comparison.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Walk-forward results not found at {path}.\n"
            "Run:  python experiments/benchmark_models.py --no-arima --no-cv  first."
        )
    df = pd.read_csv(path)
    return {row["model"]: {"r2": row["r2"], "mae": row["mae"],
                            "rmse": row["rmse"], "smape": row["smape"]}
            for _, row in df.iterrows()}


# ── 4. Learning curves ────────────────────────────────────────────────────────

def compute_learning_curves(
    df: pd.DataFrame,
    feature_cols: list,
    keys: list = ("rf", "gb", "xgb_tuned", "naive"),
    min_train_months: int = 6,
) -> pd.DataFrame:
    """
    Walk-forward MAE as a function of the number of training months available.

    For each test month T (with enough history), we record:
      - n_train_months = number of distinct months in the training window
      - MAE on the test month across all counties

    This directly answers: "does performance improve as we accumulate more
    training data?"  A still-decreasing curve at the right edge means more
    historical data would help; a flat curve means the model is saturated.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    rows = []
    for i, test_date in enumerate(dates[min_train_months:], min_train_months):
        train_mask = df["date"] < test_date
        test_mask  = df["date"] == test_date
        train_df = df[train_mask]
        test_df  = df[test_mask]

        ml_tr_ok = train_df[feature_cols].notna().all(axis=1) & train_df[config.TARGET_COL].notna()
        ml_te_ok = test_df[feature_cols].notna().all(axis=1)  & test_df[config.TARGET_COL].notna()

        X_tr = train_df.loc[ml_tr_ok, feature_cols].values
        y_tr = train_df.loc[ml_tr_ok, config.TARGET_COL].clip(lower=0).values
        X_te = test_df.loc[ml_te_ok, feature_cols].values
        y_te = test_df.loc[ml_te_ok, config.TARGET_COL].values

        n_train_months = int(train_df["date"].nunique())

        if len(X_tr) < 10 or len(X_te) == 0:
            continue

        for key in keys:
            try:
                if key == "naive":
                    naive_lookup = (train_df[ml_tr_ok]
                                    .sort_values(["county", "date"])
                                    .groupby("county")[config.TARGET_COL].last())
                    te_counties = test_df.loc[ml_te_ok, "county"]
                    preds   = te_counties.map(naive_lookup).fillna(np.nan).values
                    valid   = ~np.isnan(preds)
                    if valid.sum() == 0:
                        continue
                    mae_val = float(mean_absolute_error(y_te[valid], np.clip(preds[valid], 0, None)))
                else:
                    m = make_model(key)
                    m.fit(X_tr, y_tr)
                    preds   = np.clip(m.predict(X_te), 0, None)
                    mae_val = float(mean_absolute_error(y_te, preds))

                rows.append({
                    "model":          key,
                    "model_name":     MODELS[key],
                    "test_month":     pd.Timestamp(test_date).strftime("%Y-%m"),
                    "n_train_months": n_train_months,
                    "mae":            mae_val,
                })
            except Exception as e:
                logger.warning(f"  learning curve [{key}/{test_date}]: {e}")

    return pd.DataFrame(rows)


# ── 5. Compile diagnostics table ─────────────────────────────────────────────

def build_diagnostics_table(
    insample: dict,
    kfold: dict,
    walkforward: dict,
) -> pd.DataFrame:
    """One row per model: in-sample / CV / WF MAE and R², plus gap columns."""
    rows = []
    for key, name in MODELS.items():
        ins = insample.get(key, {})
        cv  = kfold.get(key, {})
        wf  = walkforward.get(key, {})

        ins_mae = ins.get("mae", np.nan)
        cv_mae  = cv.get("mae",  np.nan)
        wf_mae  = wf.get("mae",  np.nan)

        ins_r2  = ins.get("r2",  np.nan)
        cv_r2   = cv.get("r2",   np.nan)
        wf_r2   = wf.get("r2",   np.nan)

        rows.append({
            "model":            key,
            "model_name":       name,
            # MAE (lower = better)
            "mae_insample":     ins_mae,
            "mae_cv":           cv_mae,
            "mae_walkforward":  wf_mae,
            # MAE gaps (negative means WF is worse — overfitting signal)
            "gap_ins_to_wf":    ins_mae - wf_mae,   # negative = overfit
            "gap_cv_to_wf":     cv_mae  - wf_mae,   # negative = CV over-optimistic
            # R²
            "r2_insample":      ins_r2,
            "r2_cv":            cv_r2,
            "r2_walkforward":   wf_r2,
            "gap_r2_ins_to_wf": ins_r2  - wf_r2,    # positive = overfit
            "gap_r2_cv_to_wf":  cv_r2   - wf_r2,    # positive = CV over-optimistic
        })
    return pd.DataFrame(rows)


def print_diagnostics(diag_df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("  OVERFITTING DIAGNOSTICS — MAE comparison (lower = better)")
    print("=" * 100)
    print(f"  {'Model':<22} {'In-sample':>11} {'5-fold CV':>11} {'Walk-fwd':>11}  "
          f"{'Gap IS→WF':>11} {'Gap CV→WF':>11}  {'Verdict'}")
    print("-" * 100)
    for _, row in diag_df.iterrows():
        gap_is = row["gap_ins_to_wf"]
        gap_cv = row["gap_cv_to_wf"]

        # Verdict based on MAE gaps
        if np.isnan(gap_is):
            verdict = "—"
        elif gap_is < -0.0002:
            verdict = "⚠ OVERFIT (large)"
        elif gap_is < -0.00005:
            verdict = "~ mild overfit"
        else:
            verdict = "✓ good fit"

        print(
            f"  {row['model_name']:<22} "
            f"{row['mae_insample']:>11.6f} "
            f"{row['mae_cv']:>11.6f} "
            f"{row['mae_walkforward']:>11.6f}  "
            f"{gap_is:>+11.6f} "
            f"{gap_cv:>+11.6f}  "
            f"{verdict}"
        )
    print("-" * 100)
    print("  Gap IS→WF: in-sample MAE minus walk-forward MAE")
    print("             negative = model performs worse on unseen data (overfitting)")
    print("  Gap CV→WF: 5-fold CV MAE minus walk-forward MAE")
    print("             negative = CV is over-optimistic vs true time-series performance")
    print("=" * 100 + "\n")


# ── 6. Plots ──────────────────────────────────────────────────────────────────

def plot_comparison_bars(diag_df: pd.DataFrame, out_path: str) -> None:
    """
    Grouped bar chart: in-sample / CV / walk-forward MAE per model.

    The visual gap between the leftmost bar (in-sample) and rightmost bar
    (walk-forward) is the overfitting signal for each model.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Drop naive from CV bars (no CV formulation)
    df = diag_df.copy()
    models    = df["model_name"].tolist()
    x         = np.arange(len(models))
    bar_w     = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    bars_is = ax.bar(x - bar_w, df["mae_insample"],   bar_w, label="In-sample",    color="#2196F3", alpha=0.85)
    bars_cv = ax.bar(x,          df["mae_cv"],          bar_w, label="5-fold CV",    color="#FF9800", alpha=0.85)
    bars_wf = ax.bar(x + bar_w, df["mae_walkforward"], bar_w, label="Walk-forward", color="#4CAF50", alpha=0.85)

    # Annotate walk-forward bars with the value
    for bar in bars_wf:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.000005,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MAE (SNAP application rate)", fontsize=10)
    ax.set_title("Overfitting Diagnostics: In-sample vs CV vs Walk-forward MAE\n"
                 "Gap between blue and green bars = overfitting signal", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_overfitting_gap(diag_df: pd.DataFrame, out_path: str) -> None:
    """
    Two-panel figure:
      Left:  heatmap of R² and MAE for each split type × model
      Right: bar chart of in-sample→WF R² gap (positive = overfit)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    df = diag_df.dropna(subset=["r2_insample", "r2_walkforward"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: heatmap of R² by model × split ────────────────────────────────
    ax = axes[0]
    hmap_data = df[["r2_insample", "r2_cv", "r2_walkforward"]].copy()
    hmap_data.columns = ["In-sample", "5-fold CV", "Walk-forward"]
    hmap_data.index   = df["model_name"]

    im = ax.imshow(hmap_data.values, aspect="auto", cmap="RdYlGn",
                   vmin=max(0, hmap_data.values[~np.isnan(hmap_data.values)].min() - 0.05),
                   vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(hmap_data.columns, fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(hmap_data.index, fontsize=9)
    ax.set_title("R² by model and evaluation type\n(green = high, red = low)", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate cells
    for i in range(len(df)):
        for j in range(3):
            val = hmap_data.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.9 else "white")

    # ── Right: R² gap bars ───────────────────────────────────────────────────
    ax2 = axes[1]
    gap_col = "gap_r2_ins_to_wf"
    gaps    = df[gap_col].values
    colors  = ["#F44336" if g > 0.05 else "#FF9800" if g > 0.01 else "#4CAF50"
               for g in gaps]

    bars = ax2.barh(df["model_name"], gaps, color=colors, alpha=0.85, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("R² gap (in-sample − walk-forward)\nPositive = overfit; larger bar = more overfitting", fontsize=9)
    ax2.set_title("Overfitting gap per model\n(red > 0.05, orange > 0.01, green ≤ 0.01)", fontsize=10)

    # Colour legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#F44336", label=">0.05 (severe)"),
              Patch(color="#FF9800", label="0.01–0.05 (moderate)"),
              Patch(color="#4CAF50", label="≤0.01 (minimal)")]
    ax2.legend(handles=legend, fontsize=8, loc="lower right")

    for bar, val in zip(bars, gaps):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_learning_curves(lc_df: pd.DataFrame, out_path: str) -> None:
    """
    Walk-forward MAE vs number of training months for each model.

    Each point is one test month.  A smoothed trend line is overlaid.
    A still-declining curve at the right edge means more data would help.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = {
        "naive":       "#9E9E9E",
        "rf":          "#2196F3",
        "gb":          "#FF9800",
        "xgb_tuned":   "#4CAF50",
        "xgb_default": "#9C27B0",
        "lr":          "#F44336",
    }

    for key, grp in lc_df.groupby("model"):
        grp = grp.sort_values("n_train_months")
        color = palette.get(key, "#333333")
        name  = grp["model_name"].iloc[0]

        # Scatter
        ax.scatter(grp["n_train_months"], grp["mae"], color=color, s=20, alpha=0.4)

        # Smooth trend (only if enough points)
        if len(grp) >= 5:
            smoothed = uniform_filter1d(grp["mae"].values, size=3)
            ax.plot(grp["n_train_months"], smoothed, color=color,
                    linewidth=2, label=name, alpha=0.9)

    ax.set_xlabel("Training months available", fontsize=10)
    ax.set_ylabel("Walk-forward MAE (SNAP application rate)", fontsize=10)
    ax.set_title("Learning curves: does performance improve with more training data?\n"
                 "Still-declining curve at right = more data would help", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-plots",   action="store_true", help="Skip figure generation")
    p.add_argument("--no-lc",      action="store_true", help="Skip learning curves (slower)")
    p.add_argument("--data",       type=str, default=None)
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
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = [f for f in config.FEATURE_COLS if f in df.columns]
    logger.info(f"  Loaded {data_path}  shape={df.shape}  features={len(feature_cols)}")

    # ── Step 1: in-sample ─────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("  Step 1/4: Computing in-sample metrics...")
    insample = compute_insample(df, feature_cols)

    # ── Step 2: 5-fold CV ─────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("  Step 2/4: Computing 5-fold CV metrics...")
    kfold = compute_kfold(df, feature_cols)
    for key, m in kfold.items():
        logger.info(f"  5-fold CV  {key:<12} MAE={m['mae']:.6f}  R²={m['r2']:.4f}")

    # ── Step 3: Load walk-forward ─────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("  Step 3/4: Loading walk-forward metrics...")
    walkforward = load_walkforward()
    for key, m in walkforward.items():
        logger.info(f"  Walk-fwd   {key:<12} MAE={m['mae']:.6f}  R²={m['r2']:.4f}")

    # ── Step 4: Learning curves ───────────────────────────────────────────────
    lc_df = pd.DataFrame()
    if not args.no_lc:
        logger.info("\n" + "─" * 50)
        logger.info("  Step 4/4: Computing learning curves...")
        lc_df = compute_learning_curves(df, feature_cols)
        lc_path = os.path.join(config.OUTPUTS_ROOT, "metrics", "learning_curves.csv")
        lc_df.to_csv(lc_path, index=False)
        logger.info(f"  Saved → {lc_path}")

    # ── Build and save diagnostics table ──────────────────────────────────────
    diag_df = build_diagnostics_table(insample, kfold, walkforward)
    out_csv = os.path.join(config.OUTPUTS_ROOT, "metrics", "overfitting_diagnostics.csv")
    diag_df.to_csv(out_csv, index=False)
    logger.info(f"\n  Diagnostics table → {out_csv}")

    print_diagnostics(diag_df)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("  Generating figures...")
        fig_dir = config.FIGURES_DIR
        os.makedirs(fig_dir, exist_ok=True)

        plot_comparison_bars(
            diag_df,
            os.path.join(fig_dir, "overfitting_comparison.png"),
        )
        plot_overfitting_gap(
            diag_df,
            os.path.join(fig_dir, "overfitting_gap.png"),
        )
        if not lc_df.empty:
            try:
                from scipy.ndimage import uniform_filter1d  # noqa: used in plot fn
                plot_learning_curves(
                    lc_df,
                    os.path.join(fig_dir, "learning_curves.png"),
                )
            except ImportError:
                logger.warning("  scipy not available — learning curves plot skipped")

    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
