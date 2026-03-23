"""
feature_importance.py — Feature importance analysis for the tuned XGBoost model.

Produces four complementary views of feature importance:

  1. Built-in XGBoost importance (gain, weight, cover)
     Gain = average improvement in the loss function per split using that
     feature.  This is the most interpretable built-in metric.

  2. Permutation importance (out-of-sample)
     For each feature, the walk-forward test rows are shuffled and the
     increase in MAE is measured.  This is the gold-standard importance
     estimate because it uses held-out data — it measures real predictive
     contribution, not just how often the model used the feature.

  3. Ranked importance table
     Built-in gain and permutation importance side-by-side, with a
     consensus rank and feature-group labels (Trends / SNAP lag /
     Demographic / Seasonality).

  4. Google Trends ablation
     Two walk-forward runs (all 24 features vs 14 non-Trends features)
     directly measure how much the Trends features improve MAE and R².

Outputs
-------
  outputs/metrics/feature_importance_table.csv
  outputs/figures/feature_importance_builtin.png
  outputs/figures/feature_importance_permutation.png
  outputs/figures/feature_importance_comparison.png   — gain vs permutation
  outputs/figures/feature_importance_panel.png        — combined publication panel

Run
---
  python experiments/feature_importance.py
  python experiments/feature_importance.py --n-repeats 20   # more stable permutation
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "font.family":        "Helvetica",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Feature group labels ──────────────────────────────────────────────────────
# Colour and group assigned to each feature for visual grouping in plots.

FEATURE_GROUPS = {
    # Google Trends
    "monthly_average_CalFresh":  ("Trends",      "#1565C0"),
    "monthly_average_FoodBank":  ("Trends",      "#1565C0"),
    "calfresh_lag1":             ("Trends",      "#1976D2"),
    "calfresh_lag2":             ("Trends",      "#1976D2"),
    "foodbank_lag1":             ("Trends",      "#1976D2"),
    "foodbank_lag2":             ("Trends",      "#1976D2"),
    "calfresh_roll3":            ("Trends",      "#42A5F5"),
    "foodbank_roll3":            ("Trends",      "#42A5F5"),
    "calfresh_momentum":         ("Trends",      "#42A5F5"),
    "foodbank_momentum":         ("Trends",      "#42A5F5"),
    # SNAP lags
    "rate_lag1":                 ("SNAP lag",    "#E53935"),
    "rate_lag2":                 ("SNAP lag",    "#EF5350"),
    "rate_lag3":                 ("SNAP lag",    "#EF9A9A"),
    "rate_roll3_mean":           ("SNAP lag",    "#EF5350"),
    "rate_roll3_std":            ("SNAP lag",    "#EF9A9A"),
    # Demographics
    "Population":                ("Demographic", "#2E7D32"),
    "Median_Income":             ("Demographic", "#388E3C"),
    "log_population":            ("Demographic", "#66BB6A"),
    "log_income":                ("Demographic", "#66BB6A"),
    "income_quintile":           ("Demographic", "#A5D6A7"),
    # Seasonality / calendar
    "month":                     ("Seasonality", "#F57F17"),
    "month_sin":                 ("Seasonality", "#F9A825"),
    "month_cos":                 ("Seasonality", "#F9A825"),
    "quarter":                   ("Seasonality", "#FDD835"),
}

GROUP_LEGEND_COLOR = {
    "Trends":      "#1565C0",
    "SNAP lag":    "#E53935",
    "Demographic": "#2E7D32",
    "Seasonality": "#F57F17",
}

TRENDS_FEATURES = [f for f, (g, _) in FEATURE_GROUPS.items() if g == "Trends"]


# ── Walk-forward: collect test-set predictions + feature matrix ───────────────

def collect_wf_test_data(
    df: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Run walk-forward and return:
      model_full   — final model trained on ALL data (for built-in importance)
      X_test_all   — concatenated test-fold feature matrices
      y_test_all   — concatenated test-fold targets
      wf_preds     — concatenated walk-forward predictions

    The concatenated test data is used for permutation importance so that
    the score is always on held-out data.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    X_all, y_all, pred_all = [], [], []
    n_test = len(dates) - config.WALK_FORWARD_MIN_MONTHS

    for i, test_date in enumerate(dates[config.WALK_FORWARD_MIN_MONTHS:], 1):
        if i % 5 == 0 or i == 1:
            logger.info(f"  [{i}/{n_test}] {pd.Timestamp(test_date).strftime('%Y-%m')} ...")

        train_mask = df["date"] < test_date
        test_mask  = df["date"] == test_date

        tr_ok = (df.loc[train_mask, feature_cols].notna().all(axis=1)
                 & df.loc[train_mask, config.TARGET_COL].notna())
        te_ok = (df.loc[test_mask, feature_cols].notna().all(axis=1)
                 & df.loc[test_mask, config.TARGET_COL].notna())

        X_tr = df.loc[train_mask][tr_ok][feature_cols].values
        y_tr = df.loc[train_mask][tr_ok][config.TARGET_COL].clip(lower=0).values
        X_te = df.loc[test_mask][te_ok][feature_cols].values
        y_te = df.loc[test_mask][te_ok][config.TARGET_COL].values

        if len(X_tr) < 10 or len(X_te) == 0:
            continue

        m = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        m.fit(X_tr, y_tr)
        preds = np.clip(m.predict(X_te), 0, None)

        X_all.append(X_te)
        y_all.append(y_te)
        pred_all.append(preds)

    X_test = np.vstack(X_all)
    y_test = np.concatenate(y_all)
    preds  = np.concatenate(pred_all)

    # Full-data model for built-in importance
    mask = df[feature_cols].notna().all(axis=1) & df[config.TARGET_COL].notna()
    X_full = df.loc[mask, feature_cols].values
    y_full = df.loc[mask, config.TARGET_COL].clip(lower=0).values
    model_full = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    model_full.fit(X_full, y_full)

    return model_full, X_test, y_test, preds, feature_cols


# ── 1. Built-in feature importance ───────────────────────────────────────────

def get_builtin_importance(
    model: xgb.XGBRegressor,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Extract gain, weight (frequency), and cover importance from XGBoost.
    All three are normalised to sum to 1 for comparability.
    """
    booster = model.get_booster()
    rows = []
    for imp_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=imp_type)
        total  = sum(scores.values()) or 1.0
        for feat, val in scores.items():
            rows.append({"feature": feat, "importance_type": imp_type,
                         "importance": val / total})

    df_imp = (pd.DataFrame(rows)
                .pivot(index="feature", columns="importance_type", values="importance")
                .reset_index()
                .fillna(0))

    # Map f0, f1, … back to real names if XGBoost used integer indices
    def _resolve(name):
        if name.startswith("f") and name[1:].isdigit():
            idx = int(name[1:])
            return feature_cols[idx] if idx < len(feature_cols) else name
        return name

    df_imp["feature"] = df_imp["feature"].apply(_resolve)
    df_imp = df_imp.sort_values("gain", ascending=False).reset_index(drop=True)
    df_imp["rank_gain"] = range(1, len(df_imp) + 1)
    return df_imp


# ── 2. Permutation importance ─────────────────────────────────────────────────

def get_permutation_importance(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance on walk-forward test data.

    For each feature, shuffles its values across all test rows n_repeats
    times and records the mean increase in MAE.  Features that matter more
    cause a larger MAE increase when shuffled.

    Using MAE (not R²) as the scoring function because MAE is what we
    optimise operationally and is less sensitive to outliers than MSE.
    """
    logger.info(f"  Computing permutation importance ({n_repeats} repeats)...")

    rng = np.random.default_rng(seed)
    X_arr = X_test.values if hasattr(X_test, "values") else np.array(X_test)
    y_arr = np.array(y_test)

    baseline_mae = mean_absolute_error(y_arr, np.clip(model.predict(X_arr), 0, None))

    importances = np.zeros((len(feature_cols), n_repeats))
    for j, feat in enumerate(feature_cols):
        for r in range(n_repeats):
            X_perm = X_arr.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_mae = mean_absolute_error(y_arr, np.clip(model.predict(X_perm), 0, None))
            importances[j, r] = perm_mae - baseline_mae  # positive = important

    perm_df = pd.DataFrame({
        "feature":         feature_cols,
        "perm_importance": importances.mean(axis=1),
        "perm_std":        importances.std(axis=1),
    })
    perm_df = perm_df.sort_values("perm_importance", ascending=False).reset_index(drop=True)
    perm_df["rank_perm"] = range(1, len(perm_df) + 1)
    return perm_df


# ── 3. Trends ablation ────────────────────────────────────────────────────────

def trends_ablation(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Walk-forward comparison: all features vs non-Trends features only.
    Returns a dict with metrics for both configurations.
    """
    no_trends_cols = [f for f in feature_cols if f not in TRENDS_FEATURES]
    logger.info(f"  Ablation: {len(feature_cols)} features (all) vs "
                f"{len(no_trends_cols)} (no Trends)")

    results = {}
    for label, cols in [("all_features", feature_cols),
                         ("no_trends",    no_trends_cols)]:
        df_c  = df.copy()
        df_c["date"] = pd.to_datetime(df_c["date"])
        dates = sorted(df_c["date"].unique())
        trues, preds = [], []

        for test_date in dates[config.WALK_FORWARD_MIN_MONTHS:]:
            tr_mask = df_c["date"] < test_date
            te_mask = df_c["date"] == test_date
            tr_ok = df_c.loc[tr_mask, cols].notna().all(axis=1) & df_c.loc[tr_mask, config.TARGET_COL].notna()
            te_ok = df_c.loc[te_mask, cols].notna().all(axis=1) & df_c.loc[te_mask, config.TARGET_COL].notna()
            if tr_ok.sum() < 10 or te_ok.sum() == 0:
                continue
            X_tr = df_c.loc[tr_mask][tr_ok][cols].values
            y_tr = df_c.loc[tr_mask][tr_ok][config.TARGET_COL].clip(lower=0).values
            X_te = df_c.loc[te_mask][te_ok][cols].values
            y_te = df_c.loc[te_mask][te_ok][config.TARGET_COL].values
            m = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
            m.fit(X_tr, y_tr)
            p = np.clip(m.predict(X_te), 0, None)
            trues.extend(y_te); preds.extend(p)

        y_true = np.array(trues); y_pred = np.array(preds)
        results[label] = {
            "n_features": len(cols),
            "mae":        float(mean_absolute_error(y_true, y_pred)),
            "r2":         float(r2_score(y_true, y_pred)),
            "n":          len(y_true),
        }
        logger.info(f"  {label}: MAE={results[label]['mae']:.6f}  R²={results[label]['r2']:.4f}")

    delta_mae = results["all_features"]["mae"] - results["no_trends"]["mae"]
    delta_r2  = results["all_features"]["r2"]  - results["no_trends"]["r2"]
    results["delta"] = {
        "mae_improvement": -delta_mae,  # positive = Trends help
        "r2_improvement":   delta_r2,
        "mae_pct_improvement": -delta_mae / results["no_trends"]["mae"] * 100,
    }
    return results


# ── Build ranked table ────────────────────────────────────────────────────────

def build_importance_table(
    builtin_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = builtin_df.merge(
        perm_df[["feature", "perm_importance", "perm_std", "rank_perm"]],
        on="feature", how="outer",
    ).fillna(0)

    # Consensus rank = average of gain rank and permutation rank
    merged["consensus_rank"] = ((merged["rank_gain"] + merged["rank_perm"]) / 2).round(1)
    merged = merged.sort_values("consensus_rank")

    # Add group labels
    merged["group"] = merged["feature"].map(
        {f: g for f, (g, _) in FEATURE_GROUPS.items()}
    ).fillna("Other")
    merged["is_trends"] = merged["group"] == "Trends"

    return merged.reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _feature_color(feat: str) -> str:
    return FEATURE_GROUPS.get(feat, ("Other", "#9E9E9E"))[1]


def plot_builtin(imp_table: pd.DataFrame, top_n: int, out_path: str) -> None:
    df = imp_table.nsmallest(top_n, "rank_gain").sort_values("gain")
    colors = [_feature_color(f) for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(7, max(5, top_n * 0.32)))
    bars = ax.barh(df["feature"], df["gain"], color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.4)

    for bar, val in zip(bars, df["gain"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5)

    ax.set_xlabel("Normalised gain (fraction of total model improvement)")
    ax.set_title(f"XGBoost Built-in Feature Importance (Gain)\nTop {top_n} features")
    ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    handles = [mpatches.Patch(color=c, label=g, alpha=0.85)
               for g, c in GROUP_LEGEND_COLOR.items()]
    ax.legend(handles=handles, fontsize=8, title="Feature group",
              title_fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_permutation(imp_table: pd.DataFrame, top_n: int, out_path: str) -> None:
    df = imp_table.nsmallest(top_n, "rank_perm").sort_values("perm_importance")
    colors = [_feature_color(f) for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(7, max(5, top_n * 0.32)))
    ax.barh(df["feature"], df["perm_importance"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.4, label="_nolegend_")
    ax.errorbar(df["perm_importance"], df["feature"],
                xerr=df["perm_std"], fmt="none",
                color="#333333", linewidth=1.0, capsize=3, alpha=0.6)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean increase in MAE when feature is permuted\n"
                  "(larger = more important; ≤ 0 = feature adds no value)")
    ax.set_title(f"Permutation Feature Importance (out-of-sample)\nTop {top_n} features")
    ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    handles = [mpatches.Patch(color=c, label=g, alpha=0.85)
               for g, c in GROUP_LEGEND_COLOR.items()]
    ax.legend(handles=handles, fontsize=8, title="Feature group",
              title_fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_comparison(imp_table: pd.DataFrame, top_n: int, out_path: str) -> None:
    """Side-by-side normalised gain vs permutation importance, top N by consensus rank."""
    df = imp_table.head(top_n).copy()

    # Normalise permutation to 0-1 range for visual comparability
    pmax = df["perm_importance"].clip(lower=0).max() or 1.0
    df["perm_norm"] = df["perm_importance"].clip(lower=0) / pmax
    df["gain_norm"] = df["gain"] / (df["gain"].max() or 1.0)

    df = df.sort_values("gain_norm", ascending=True)
    y  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))

    ax.barh(y - 0.2, df["gain_norm"],   height=0.35, color="#1565C0", alpha=0.80,
            label="Built-in gain (normalised)")
    ax.barh(y + 0.2, df["perm_norm"],   height=0.35, color="#E53935", alpha=0.80,
            label="Permutation importance (normalised)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=8.5)
    ax.set_xlabel("Normalised importance score (within each method)")
    ax.set_title(f"Built-in vs Permutation Importance — Top {top_n} features\n"
                 "Features sorted by built-in gain (left) to permutation rank agreement")
    ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    # Colour left ytick labels by group
    for label, feat in zip(ax.get_yticklabels(), df["feature"]):
        label.set_color(_feature_color(feat))

    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_panel(
    imp_table: pd.DataFrame,
    ablation: dict,
    top_n: int,
    out_path: str,
) -> None:
    """Combined publication panel: top predictors + ablation bar."""
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Left: built-in gain ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    df0 = imp_table.nsmallest(top_n, "rank_gain").sort_values("gain")
    colors0 = [_feature_color(f) for f in df0["feature"]]
    ax0.barh(df0["feature"], df0["gain"], color=colors0, alpha=0.85, edgecolor="white")
    ax0.set_xlabel("Normalised gain")
    ax0.set_title(f"(a) Built-in Importance\n(top {top_n} by gain)")
    ax0.xaxis.grid(True, alpha=0.3); ax0.set_axisbelow(True)
    handles = [mpatches.Patch(color=c, label=g, alpha=0.85)
               for g, c in GROUP_LEGEND_COLOR.items()]
    ax0.legend(handles=handles, fontsize=7.5, title="Group", title_fontsize=7.5)

    # ── Middle: permutation importance ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    df1 = imp_table.nsmallest(top_n, "rank_perm").sort_values("perm_importance")
    colors1 = [_feature_color(f) for f in df1["feature"]]
    ax1.barh(df1["feature"], df1["perm_importance"].clip(lower=0),
             color=colors1, alpha=0.85, edgecolor="white")
    ax1.errorbar(df1["perm_importance"].clip(lower=0), df1["feature"],
                 xerr=df1["perm_std"], fmt="none",
                 color="#333", linewidth=0.9, capsize=2.5, alpha=0.55)
    ax1.set_xlabel("Mean MAE increase when permuted")
    ax1.set_title(f"(b) Permutation Importance\n(top {top_n}, out-of-sample)")
    ax1.xaxis.grid(True, alpha=0.3); ax1.set_axisbelow(True)

    # ── Right: Trends ablation ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    labels   = ["All 24\nfeatures", "No Trends\n(14 features)"]
    mae_vals = [ablation["all_features"]["mae"], ablation["no_trends"]["mae"]]
    r2_vals  = [ablation["all_features"]["r2"],  ablation["no_trends"]["r2"]]
    colors2  = ["#1565C0", "#9E9E9E"]

    x = np.array([0, 1])
    bars = ax2.bar(x, mae_vals, width=0.5, color=colors2, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, mae_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.000005,
                 f"{val:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Delta annotation
    delta_mae = ablation["delta"]["mae_improvement"]
    delta_pct = ablation["delta"]["mae_pct_improvement"]
    ax2.annotate(
        "",
        xy=(0, ablation["no_trends"]["mae"]),
        xytext=(1, ablation["no_trends"]["mae"]),
        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5),
    )
    ax2.text(0.5, ablation["no_trends"]["mae"] * 1.015,
             f"Δ={delta_mae:+.5f}\n({delta_pct:+.1f}%)",
             ha="center", va="bottom", fontsize=9, color="#333")

    # R² as secondary labels
    for xi, r2 in zip(x, r2_vals):
        ax2.text(xi, mae_vals[int(xi)] * 0.5, f"R²={r2:.4f}",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Walk-forward MAE")
    ax2.set_title("(c) Google Trends Ablation\n(walk-forward MAE with vs without)")
    ax2.yaxis.grid(True, alpha=0.3); ax2.set_axisbelow(True)
    ymax = max(mae_vals) * 1.12
    ax2.set_ylim(0, ymax)

    fig.suptitle("Feature Importance Analysis — XGBoost (tuned), Walk-Forward Validation",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_importance_table(imp_table: pd.DataFrame, top_n: int = 24) -> None:
    print("\n" + "=" * 100)
    print(f"  FEATURE IMPORTANCE — Ranked Table (top {top_n})")
    print("=" * 100)
    print(f"  {'#':>3}  {'Feature':<26} {'Group':<13} "
          f"{'Gain':>8} {'Weight':>8} {'Cover':>8}  "
          f"{'Perm ΔMae':>10} {'±Std':>8}  {'Perm rank':>9}  {'Trends?'}")
    print("  " + "─" * 96)
    for i, row in imp_table.head(top_n).iterrows():
        trend_flag = "✓" if row["is_trends"] else ""
        print(
            f"  {i+1:>3}  {row['feature']:<26} {row['group']:<13} "
            f"{row.get('gain',0):>8.4f} {row.get('weight',0):>8.4f} "
            f"{row.get('cover',0):>8.4f}  "
            f"{row['perm_importance']:>+10.6f} {row['perm_std']:>8.6f}  "
            f"{int(row['rank_perm']):>9}  {trend_flag}"
        )
    print("=" * 100)


def print_ablation(ablation: dict) -> None:
    af = ablation["all_features"]
    nt = ablation["no_trends"]
    d  = ablation["delta"]
    print("\n" + "=" * 65)
    print("  GOOGLE TRENDS ABLATION — Walk-Forward MAE Comparison")
    print("=" * 65)
    print(f"  {'Configuration':<30} {'N feats':>7} {'MAE':>12} {'R²':>8}")
    print("  " + "─" * 61)
    print(f"  {'All 24 features':<30} {af['n_features']:>7} {af['mae']:>12.6f} {af['r2']:>8.4f}")
    print(f"  {'No Trends (14 features)':<30} {nt['n_features']:>7} {nt['mae']:>12.6f} {nt['r2']:>8.4f}")
    print("  " + "─" * 61)
    sign = "+" if d["mae_improvement"] > 0 else ""
    print(f"  {'Improvement with Trends':<30} {'':>7} "
          f"{sign}{d['mae_improvement']:>12.6f} {sign}{d['r2_improvement']:>7.4f}")
    print(f"  MAE reduction: {d['mae_pct_improvement']:.2f}%  "
          f"({'Trends help' if d['mae_improvement'] > 0 else 'Trends do NOT help'})")
    print("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top-n",      type=int, default=15,
                   help="Number of top features to show in plots (default 15)")
    p.add_argument("--n-repeats",  type=int, default=10,
                   help="Permutation importance repeats (default 10)")
    p.add_argument("--no-ablation",action="store_true",
                   help="Skip Trends ablation (saves ~1 min)")
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
    feature_cols = [f for f in config.FEATURE_COLS if f in df.columns]
    logger.info(f"  Loaded {data_path}  shape={df.shape}  features={len(feature_cols)}")

    # ── Collect walk-forward test data ────────────────────────────────────────
    logger.info("\n  Running walk-forward to collect test data...")
    model_full, X_test, y_test, wf_preds, feat_cols = collect_wf_test_data(df, feature_cols)
    wf_mae = float(mean_absolute_error(y_test, wf_preds))
    wf_r2  = float(r2_score(y_test, wf_preds))
    logger.info(f"  Walk-forward: MAE={wf_mae:.6f}  R²={wf_r2:.4f}  n={len(y_test):,}")

    # ── Built-in importance ───────────────────────────────────────────────────
    logger.info("\n  Computing built-in feature importance...")
    builtin_df = get_builtin_importance(model_full, feat_cols)
    logger.info(f"  Top 5 by gain: {builtin_df.head(5)['feature'].tolist()}")

    # ── Permutation importance ────────────────────────────────────────────────
    perm_df = get_permutation_importance(
        model_full, X_test, y_test, feat_cols, n_repeats=args.n_repeats
    )
    logger.info(f"  Top 5 by permutation: {perm_df.head(5)['feature'].tolist()}")

    # ── Ranked table ──────────────────────────────────────────────────────────
    imp_table = build_importance_table(builtin_df, perm_df)
    print_importance_table(imp_table)

    # ── Trends ablation ───────────────────────────────────────────────────────
    if not args.no_ablation:
        logger.info("\n  Running Trends ablation walk-forward...")
        ablation = trends_ablation(df, feature_cols)
        print_ablation(ablation)
    else:
        ablation = None

    # ── Save table ────────────────────────────────────────────────────────────
    metrics_dir = os.path.join(config.OUTPUTS_ROOT, "metrics")
    table_path  = os.path.join(metrics_dir, "feature_importance_table.csv")
    imp_table.to_csv(table_path, index=False)
    logger.info(f"\n  Table → {table_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig_dir = config.FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)
    top_n = args.top_n

    logger.info("  Generating plots...")
    plot_builtin(imp_table, top_n,
                 os.path.join(fig_dir, "feature_importance_builtin.png"))
    plot_permutation(imp_table, top_n,
                     os.path.join(fig_dir, "feature_importance_permutation.png"))
    plot_comparison(imp_table, top_n,
                    os.path.join(fig_dir, "feature_importance_comparison.png"))

    if ablation:
        plot_panel(imp_table, ablation, top_n,
                   os.path.join(fig_dir, "feature_importance_panel.png"))

    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
