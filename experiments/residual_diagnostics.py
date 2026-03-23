"""
residual_diagnostics.py — Publication-quality residual diagnostics for the
                           final XGBoost model.

Generates five diagnostic plots from walk-forward predictions:

  1. residual_histogram.png      — distribution of residuals with normal overlay
                                   and summary statistics annotated
  2. actual_vs_predicted.png     — scatter with identity line, coloured by
                                   county density tier, hexbin density inset
  3. residuals_over_time.png     — monthly mean ± 1 std band; individual county
                                   residuals shown as a faded strip
  4. residuals_by_county.png     — horizontal box plots, counties sorted by
                                   median absolute residual
  5. residuals_vs_fitted.png     — classic diagnostic: residuals vs fitted values
                                   with LOESS smooth and ±2 MAE bands

Individual files are saved at 300 DPI.  A combined 2×3 panel figure is also
saved as residual_diagnostics_panel.png for use directly in a paper.

Run
---
  python experiments/residual_diagnostics.py
  python experiments/residual_diagnostics.py --model rf
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config
# Reuse walk-forward collection from county_error_analysis
from experiments.county_error_analysis import collect_predictions

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# ── Publication style ─────────────────────────────────────────────────────────
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
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#cccccc",
})

ACCENT   = "#2C6FAC"   # main blue
ACCENT2  = "#E05C2A"   # orange for contrast
GREY     = "#888888"
LIGHTGREY= "#DDDDDD"

# Density tier colours (consistent with subgroup_analysis.py)
DENSITY_COLORS = {"Urban": "#E53935", "Suburban": "#FF8F00", "Rural": "#5D4037"}

PUB_DPI   = 300
SAVE_ARGS = dict(dpi=PUB_DPI, bbox_inches="tight", pad_inches=0.05)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Density labels (from popData.csv) ────────────────────────────────────────

def load_density_labels() -> dict:
    """Return {county: density_group} using the same thresholds as subgroup_analysis."""
    pop = pd.read_csv(config.POP_FILE)
    pop["county"] = pop["County"].str.strip()
    def _tier(d):
        if d >= 500:  return "Urban"
        if d >= 100:  return "Suburban"
        return "Rural"
    pop["density_group"] = pop["Population Density"].apply(_tier)
    return dict(zip(pop["county"], pop["density_group"]))


# ── LOESS / lowess smoother ───────────────────────────────────────────────────

def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.4) -> np.ndarray:
    """
    Simple local linear smoother (manual implementation — avoids statsmodels dep).
    Returns smoothed y values at the same x positions, sorted by x.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n   = len(xs)
    h   = int(np.ceil(frac * n))
    out = np.empty(n)
    for i in range(n):
        lo  = max(0, i - h // 2)
        hi  = min(n, lo + h)
        lo  = max(0, hi - h)
        xi, yi = xs[lo:hi], ys[lo:hi]
        w   = 1 - (np.abs(xi - xs[i]) / (np.abs(xi - xs[i]).max() + 1e-12)) ** 3
        w   = np.clip(w, 0, None) ** 3
        if w.sum() == 0:
            out[i] = np.mean(yi)
        else:
            out[i] = np.average(yi, weights=w)
    return out[np.argsort(order)]


# ── Plot 1: Residual histogram ────────────────────────────────────────────────

def plot_residual_histogram(pred_df: pd.DataFrame, out_path: str) -> plt.Figure:
    resid = pred_df["error"].values

    fig, ax = plt.subplots(figsize=(6, 4))

    # Histogram
    n_bins = min(50, max(20, len(resid) // 20))
    counts, edges, patches = ax.hist(
        resid, bins=n_bins, color=ACCENT, alpha=0.75,
        edgecolor="white", linewidth=0.4, density=True, label="Residuals"
    )

    # Normal overlay fitted to residuals
    mu, sigma = float(np.mean(resid)), float(np.std(resid))
    x_range   = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
            color=ACCENT2, linewidth=1.8, label=f"Normal fit\n(μ={mu:.2e}, σ={sigma:.2e})")

    # Zero line
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    # Shapiro-Wilk test (first 5000 samples)
    sw_stat, sw_p = stats.shapiro(resid[:5000])
    skew  = float(stats.skew(resid))
    kurt  = float(stats.kurtosis(resid))

    textstr = (f"n = {len(resid):,}\n"
               f"Mean  = {mu:.2e}\n"
               f"Std   = {sigma:.2e}\n"
               f"Skew  = {skew:.3f}\n"
               f"Kurt  = {kurt:.3f}\n"
               f"Shapiro p = {sw_p:.3f}")
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=LIGHTGREY, alpha=0.9))

    ax.set_xlabel("Residual (predicted − actual SNAP rate)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))

    plt.tight_layout()
    fig.savefig(out_path, **SAVE_ARGS)
    logger.info(f"  Saved → {out_path}")
    return fig


# ── Plot 2: Actual vs predicted ───────────────────────────────────────────────

def plot_actual_vs_predicted(
    pred_df: pd.DataFrame,
    density_labels: dict,
    out_path: str,
) -> plt.Figure:
    df = pred_df.copy()
    df["density"] = df["county"].map(density_labels).fillna("Rural")

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for tier in ["Urban", "Suburban", "Rural"]:
        sub = df[df["density"] == tier]
        ax.scatter(
            sub["actual"], sub["predicted"],
            color=DENSITY_COLORS[tier], s=6, alpha=0.35,
            edgecolors="none", label=tier, rasterized=True,
        )

    # Identity line (perfect prediction)
    lim_lo = min(df["actual"].min(), df["predicted"].min()) * 0.98
    lim_hi = max(df["actual"].max(), df["predicted"].max()) * 1.02
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="black", linewidth=1.2, linestyle="--", alpha=0.8,
            label="Perfect prediction (y = x)", zorder=5)

    # ±MAE bands
    mae = float(pred_df["abs_error"].mean())
    ax.fill_between([lim_lo, lim_hi],
                    [lim_lo - mae, lim_hi - mae],
                    [lim_lo + mae, lim_hi + mae],
                    color=ACCENT, alpha=0.08, label=f"±MAE band ({mae:.4f})")

    # Overall R²
    from sklearn.metrics import r2_score
    r2 = r2_score(df["actual"], df["predicted"])
    ax.text(0.04, 0.97,
            f"R² = {r2:.4f}\nMAE = {mae:.5f}\nn = {len(df):,}",
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=LIGHTGREY, alpha=0.9))

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Actual SNAP application rate")
    ax.set_ylabel("Predicted SNAP application rate")
    ax.set_title("Actual vs Predicted")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="sci", scilimits=(-3, -3))

    plt.tight_layout()
    fig.savefig(out_path, **SAVE_ARGS)
    logger.info(f"  Saved → {out_path}")
    return fig


# ── Plot 3: Residuals over time ───────────────────────────────────────────────

def plot_residuals_over_time(pred_df: pd.DataFrame, out_path: str) -> plt.Figure:
    df = pred_df.copy()
    df["month"] = pd.Categorical(df["month"], categories=sorted(df["month"].unique()), ordered=True)

    # Per-county residuals as faded strips
    fig, ax = plt.subplots(figsize=(9, 4))

    month_order = sorted(df["month"].unique())
    x_pos       = {m: i for i, m in enumerate(month_order)}
    df["x"]     = df["month"].map(x_pos)

    # Individual county traces (faded)
    for county, grp in df.groupby("county"):
        grp = grp.sort_values("x")
        ax.plot(grp["x"], grp["error"], color=GREY, linewidth=0.4,
                alpha=0.2, zorder=1)

    # Monthly statistics
    monthly = df.groupby("month")["error"].agg(["mean", "std"]).reset_index()
    monthly["x"] = monthly["month"].map(x_pos)
    monthly = monthly.sort_values("x")

    ax.fill_between(
        monthly["x"],
        monthly["mean"] - monthly["std"],
        monthly["mean"] + monthly["std"],
        color=ACCENT, alpha=0.20, label="±1 SD across counties", zorder=2,
    )
    ax.plot(monthly["x"], monthly["mean"],
            color=ACCENT, linewidth=2.0, marker="o", markersize=4,
            label="Monthly mean residual", zorder=3)

    # Zero line
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_xticks(range(len(month_order)))
    ax.set_xticklabels(month_order, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Test month")
    ax.set_ylabel("Residual (predicted − actual)")
    ax.set_title("Residuals over Time")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, **SAVE_ARGS)
    logger.info(f"  Saved → {out_path}")
    return fig


# ── Plot 4: Residuals by county ───────────────────────────────────────────────

def plot_residuals_by_county(
    pred_df: pd.DataFrame,
    density_labels: dict,
    out_path: str,
) -> plt.Figure:
    # Sort counties by median absolute residual (worst at top)
    county_order = (pred_df.groupby("county")["abs_error"]
                    .median()
                    .sort_values(ascending=True)
                    .index.tolist())

    n = len(county_order)
    fig, ax = plt.subplots(figsize=(7, max(8, n * 0.22)))

    # Build data in display order
    data    = [pred_df.loc[pred_df["county"] == c, "error"].values for c in county_order]
    colors  = [DENSITY_COLORS.get(density_labels.get(c, "Rural"), GREY) for c in county_order]

    bp = ax.boxplot(
        data, vert=False, patch_artist=True,
        medianprops=dict(color="white", linewidth=1.5),
        flierprops=dict(marker=".", markersize=2.5, alpha=0.4,
                        markeredgewidth=0),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels(county_order, fontsize=7.5)
    ax.set_xlabel("Residual (predicted − actual SNAP rate)")
    ax.set_title("Residuals by County\n(sorted by median absolute error, best at bottom)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))

    # Density legend
    handles = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.8, label=g)
               for g, c in DENSITY_COLORS.items()]
    ax.legend(handles=handles, title="Density tier", fontsize=8,
              title_fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, **SAVE_ARGS)
    logger.info(f"  Saved → {out_path}")
    return fig


# ── Plot 5: Residuals vs fitted ───────────────────────────────────────────────

def plot_residuals_vs_fitted(
    pred_df: pd.DataFrame,
    density_labels: dict,
    out_path: str,
) -> plt.Figure:
    df    = pred_df.copy()
    df["density"] = df["county"].map(density_labels).fillna("Rural")

    fitted = df["predicted"].values
    resid  = df["error"].values
    mae    = float(df["abs_error"].mean())

    fig, ax = plt.subplots(figsize=(6.5, 5))

    for tier in ["Urban", "Suburban", "Rural"]:
        sub = df[df["density"] == tier]
        ax.scatter(
            sub["predicted"], sub["error"],
            color=DENSITY_COLORS[tier], s=6, alpha=0.30,
            edgecolors="none", label=tier, rasterized=True,
        )

    # Zero line
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    # ±MAE reference lines
    ax.axhline( mae, color=ACCENT2, linewidth=1.0, linestyle=":",
                alpha=0.8, label=f"+MAE ({mae:.4f})")
    ax.axhline(-mae, color=ACCENT2, linewidth=1.0, linestyle=":",
                alpha=0.8, label=f"−MAE")

    # LOESS smooth
    sort_idx  = np.argsort(fitted)
    smooth_y  = lowess_smooth(fitted, resid, frac=0.35)
    ax.plot(fitted[sort_idx], smooth_y[sort_idx],
            color="black", linewidth=1.8, alpha=0.85, label="LOESS smooth", zorder=5)

    # Highlight largest outlier residuals
    top_outliers = df.nlargest(5, "abs_error")
    for _, row in top_outliers.iterrows():
        ax.annotate(
            row["county"],
            (row["predicted"], row["error"]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        )

    ax.set_xlabel("Fitted value (predicted SNAP application rate)")
    ax.set_ylabel("Residual (predicted − actual)")
    ax.set_title("Residuals vs Fitted Values")
    ax.ticklabel_format(style="sci", scilimits=(-3, -3))

    handles, labels = ax.get_legend_handles_labels()
    # Put density tiers first, then reference lines, then LOESS
    tier_h = [h for h, l in zip(handles, labels) if l in DENSITY_COLORS]
    tier_l = [l for l in labels if l in DENSITY_COLORS]
    rest_h = [h for h, l in zip(handles, labels) if l not in DENSITY_COLORS]
    rest_l = [l for l in labels if l not in DENSITY_COLORS]
    ax.legend(tier_h + rest_h, tier_l + rest_l, fontsize=7.5,
              markerscale=2, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, **SAVE_ARGS)
    logger.info(f"  Saved → {out_path}")
    return fig


# ── Combined panel ────────────────────────────────────────────────────────────

def plot_panel(pred_df: pd.DataFrame, density_labels: dict, out_path: str) -> None:
    """
    2×3 panel combining all five diagnostics (plus one summary stats cell).
    Designed to fit on a single paper page.
    """
    from sklearn.metrics import r2_score

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # --- shared data prep ---
    resid  = pred_df["error"].values
    fitted = pred_df["predicted"].values
    actual = pred_df["actual"].values
    mae    = float(pred_df["abs_error"].mean())
    mu, sigma = float(np.mean(resid)), float(np.std(resid))

    month_order = sorted(pred_df["month"].unique())
    x_pos = {m: i for i, m in enumerate(month_order)}

    # ── Cell (0,0): Histogram ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    n_bins = min(45, max(20, len(resid) // 20))
    ax0.hist(resid, bins=n_bins, color=ACCENT, alpha=0.72,
             edgecolor="white", linewidth=0.3, density=True)
    x_r = np.linspace(resid.min(), resid.max(), 300)
    ax0.plot(x_r, stats.norm.pdf(x_r, mu, sigma), color=ACCENT2, linewidth=1.6)
    ax0.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    sw_stat, sw_p = stats.shapiro(resid[:5000])
    ax0.set_title("(a) Residual Distribution")
    ax0.set_xlabel("Residual"); ax0.set_ylabel("Density")
    ax0.text(0.97, 0.97,
             f"μ={mu:.2e}\nσ={sigma:.2e}\nSkew={stats.skew(resid):.2f}\nSW p={sw_p:.3f}",
             transform=ax0.transAxes, fontsize=7.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LIGHTGREY, alpha=0.9))
    ax0.ticklabel_format(axis="x", style="sci", scilimits=(-3,-3))

    # ── Cell (0,1): Actual vs predicted ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    df_tmp = pred_df.copy()
    df_tmp["density"] = df_tmp["county"].map(density_labels).fillna("Rural")
    for tier in ["Urban", "Suburban", "Rural"]:
        sub = df_tmp[df_tmp["density"] == tier]
        ax1.scatter(sub["actual"], sub["predicted"],
                    color=DENSITY_COLORS[tier], s=5, alpha=0.3,
                    edgecolors="none", label=tier, rasterized=True)
    lim_lo = min(actual.min(), fitted.min()) * 0.97
    lim_hi = max(actual.max(), fitted.max()) * 1.03
    ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
             color="black", linewidth=1.1, linestyle="--", alpha=0.75)
    ax1.fill_between([lim_lo, lim_hi],
                     [lim_lo-mae, lim_hi-mae], [lim_lo+mae, lim_hi+mae],
                     color=ACCENT, alpha=0.08)
    r2 = r2_score(actual, fitted)
    ax1.text(0.04, 0.97, f"R²={r2:.4f}\nMAE={mae:.5f}",
             transform=ax1.transAxes, fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LIGHTGREY, alpha=0.9))
    ax1.set_xlim(lim_lo, lim_hi); ax1.set_ylim(lim_lo, lim_hi)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted")
    ax1.set_title("(b) Actual vs Predicted")
    ax1.legend(fontsize=7, markerscale=2, handletextpad=0.3)
    ax1.ticklabel_format(style="sci", scilimits=(-3,-3))

    # ── Cell (0,2): Residuals vs fitted ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for tier in ["Urban", "Suburban", "Rural"]:
        sub = df_tmp[df_tmp["density"] == tier]
        ax2.scatter(sub["predicted"], sub["error"],
                    color=DENSITY_COLORS[tier], s=5, alpha=0.28,
                    edgecolors="none", label=tier, rasterized=True)
    ax2.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax2.axhline( mae, color=ACCENT2, linewidth=0.9, linestyle=":", alpha=0.8)
    ax2.axhline(-mae, color=ACCENT2, linewidth=0.9, linestyle=":", alpha=0.8)
    sort_idx = np.argsort(fitted)
    smooth_y = lowess_smooth(fitted, resid, frac=0.35)
    ax2.plot(fitted[sort_idx], smooth_y[sort_idx],
             color="black", linewidth=1.6, alpha=0.85, label="LOESS")
    ax2.set_xlabel("Fitted value"); ax2.set_ylabel("Residual")
    ax2.set_title("(c) Residuals vs Fitted")
    ax2.legend(fontsize=7, markerscale=2, ncol=2, handletextpad=0.3)
    ax2.ticklabel_format(style="sci", scilimits=(-3,-3))

    # ── Cell (1,0–1): Residuals over time (wide) ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    for county, grp in pred_df.groupby("county"):
        grp = grp.copy(); grp["x"] = grp["month"].map(x_pos)
        grp = grp.sort_values("x")
        ax3.plot(grp["x"], grp["error"], color=GREY, linewidth=0.35, alpha=0.18)
    monthly = (pred_df.groupby("month")["error"]
               .agg(["mean","std"]).reset_index())
    monthly["x"] = monthly["month"].map(x_pos)
    monthly = monthly.sort_values("x")
    ax3.fill_between(monthly["x"],
                     monthly["mean"] - monthly["std"],
                     monthly["mean"] + monthly["std"],
                     color=ACCENT, alpha=0.18, label="±1 SD")
    ax3.plot(monthly["x"], monthly["mean"],
             color=ACCENT, linewidth=1.8, marker="o", markersize=3.5,
             label="Monthly mean")
    ax3.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax3.set_xticks(range(len(month_order)))
    ax3.set_xticklabels(month_order, rotation=40, ha="right", fontsize=7.5)
    ax3.set_xlabel("Test month"); ax3.set_ylabel("Residual")
    ax3.set_title("(d) Residuals over Time")
    ax3.legend(fontsize=8)
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(-3,-3))

    # ── Cell (1,2): Residuals by county (top/bottom 15 for readability) ───────
    ax4 = fig.add_subplot(gs[1, 2])
    county_med = (pred_df.groupby("county")["abs_error"].median()
                  .sort_values(ascending=True))
    show_n   = 15  # top + bottom
    counties = list(county_med.tail(show_n).index[::-1]) + \
               list(county_med.head(show_n).index[::-1])
    counties = list(dict.fromkeys(counties))  # dedup preserving order

    data_c   = [pred_df.loc[pred_df["county"] == c, "error"].values for c in counties]
    colors_c = [DENSITY_COLORS.get(density_labels.get(c,"Rural"), GREY) for c in counties]

    bp4 = ax4.boxplot(data_c, vert=False, patch_artist=True,
                      medianprops=dict(color="white", linewidth=1.2),
                      flierprops=dict(marker=".", markersize=2, alpha=0.35,
                                      markeredgewidth=0),
                      whiskerprops=dict(linewidth=0.7),
                      capprops=dict(linewidth=0.7))
    for patch, color in zip(bp4["boxes"], colors_c):
        patch.set_facecolor(color); patch.set_alpha(0.80)

    # Separator line between worst and best halves
    ax4.axhline(show_n + 0.5, color=GREY, linewidth=0.8, linestyle=":", alpha=0.6)
    ax4.text(ax4.get_xlim()[0] if ax4.get_xlim()[0] != 0 else -0.001,
             show_n + 0.5, " ← best / worst →", fontsize=6.5,
             va="center", color=GREY)

    ax4.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax4.set_yticks(range(1, len(counties)+1))
    ax4.set_yticklabels(counties, fontsize=6.8)
    ax4.set_xlabel("Residual"); ax4.set_title("(e) Residuals by County\n(best & worst 15)")
    ax4.ticklabel_format(axis="x", style="sci", scilimits=(-3,-3))

    fig.suptitle(
        "Residual Diagnostics — XGBoost (tuned), Walk-Forward Validation",
        fontsize=13, fontweight="bold", y=1.01,
    )

    fig.savefig(out_path, **SAVE_ARGS)
    plt.close(fig)
    logger.info(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="xgb_tuned")
    p.add_argument("--data",  type=str, default=None)
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

    logger.info(f"  Collecting walk-forward predictions (model={args.model})...")
    pred_df = collect_predictions(df, args.model)
    logger.info(f"  {len(pred_df):,} county-month predictions collected")

    density_labels = load_density_labels()

    fig_dir = config.FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)
    name = args.model

    logger.info("\n  Generating publication-quality plots (300 DPI)...")

    plot_residual_histogram(
        pred_df, os.path.join(fig_dir, f"residual_histogram_{name}.png"))
    plot_actual_vs_predicted(
        pred_df, density_labels, os.path.join(fig_dir, f"actual_vs_predicted_{name}.png"))
    plot_residuals_over_time(
        pred_df, os.path.join(fig_dir, f"residuals_over_time_{name}.png"))
    plot_residuals_by_county(
        pred_df, density_labels, os.path.join(fig_dir, f"residuals_by_county_{name}.png"))
    plot_residuals_vs_fitted(
        pred_df, density_labels, os.path.join(fig_dir, f"residuals_vs_fitted_{name}.png"))

    logger.info("\n  Generating combined panel figure...")
    plot_panel(pred_df, density_labels,
               os.path.join(fig_dir, f"residual_diagnostics_panel_{name}.png"))

    logger.info("\n  Done. All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
