"""
subgroup_analysis.py — Subgroup performance analysis across county characteristics.

Compares model predictive performance across three subgroup dimensions:

  1. Population size     — high vs low (median split)
  2. Median income       — high vs low (median split)
  3. Population density  — urban / suburban / rural (natural breaks in the
                           density distribution; see DENSITY_THRESHOLDS below)

For each dimension the script reports:
  - N counties per group
  - Mean / median MAE and MAPE
  - Fraction of counties with positive bias (over-prediction)
  - Mean bias and std of errors
  - Mann-Whitney U p-value for MAE difference between groups (non-parametric;
    appropriate given n=58 counties with a highly skewed distribution)

Walk-forward predictions are loaded from county_error_analysis_xgb_tuned.csv
(produced by county_error_analysis.py).  Run that first if the file is absent.

Outputs
-------
  outputs/metrics/subgroup_analysis.csv        — per-county table with all labels
  outputs/metrics/subgroup_summary.csv         — one row per (dimension, group)
  outputs/figures/subgroup_boxplots.png        — MAE box plots for all dimensions
  outputs/figures/subgroup_mae_strips.png      — strip + box overlays
  outputs/figures/subgroup_heatmap.png         — mean MAE heatmap across subgroups

Run
---
  python experiments/subgroup_analysis.py
  python experiments/subgroup_analysis.py --model rf   # use a different model
  python experiments/subgroup_analysis.py --no-plots
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

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

# ── Density classification thresholds (people / sq mile) ─────────────────────
# Based on the California county density distribution:
#   rural:     < 100  (lower half; mostly Sierra Nevada / rural NorCal)
#   suburban:  100–499
#   urban:     >= 500 (SF, LA, Orange, Alameda, San Diego, San Mateo, etc.)
DENSITY_THRESHOLDS = {"rural": 100, "suburban": 500}


# ── Load subgroup labels from raw data ────────────────────────────────────────

def load_county_attributes() -> pd.DataFrame:
    """
    Build a county-level attribute table with population, income, and density.
    Returns one row per county with columns:
      county, population, pop_density, median_income
    """
    # Population + density
    pop = pd.read_csv(config.POP_FILE)
    pop = pop.rename(columns={
        "County":             "county",
        "Population":         "population",
        "Population Density": "pop_density",
    })
    pop["county"] = pop["county"].str.strip()

    # Income — strip commas and cast to float
    inc = pd.read_csv(config.INCOME_FILE)
    inc = inc.rename(columns={"County": "county", "Median Income": "median_income"})
    inc["county"]        = inc["county"].str.strip()
    inc["median_income"] = (inc["median_income"]
                             .astype(str)
                             .str.replace(",", "")
                             .astype(float))

    attrs = pop[["county", "population", "pop_density"]].merge(
        inc[["county", "median_income"]], on="county", how="left"
    )
    return attrs


def assign_subgroups(attrs: pd.DataFrame) -> pd.DataFrame:
    """
    Add subgroup label columns to the attributes table.

    population_group:   'High population' / 'Low population'  (median split)
    income_group:       'High income'     / 'Low income'       (median split)
    density_group:      'Urban' / 'Suburban' / 'Rural'         (threshold-based)
    """
    attrs = attrs.copy()

    # Population — median split
    pop_median = attrs["population"].median()
    attrs["population_group"] = np.where(
        attrs["population"] >= pop_median, "High population", "Low population"
    )

    # Income — median split
    inc_median = attrs["median_income"].median()
    attrs["income_group"] = np.where(
        attrs["median_income"] >= inc_median, "High income", "Low income"
    )

    # Density — three tiers
    def _density_label(d):
        if d >= DENSITY_THRESHOLDS["suburban"]:
            return "Urban"
        elif d >= DENSITY_THRESHOLDS["rural"]:
            return "Suburban"
        else:
            return "Rural"

    attrs["density_group"] = attrs["pop_density"].apply(_density_label)

    logger.info(
        f"  Population split at median={pop_median:,.0f}: "
        + attrs["population_group"].value_counts().to_dict().__str__()
    )
    logger.info(
        f"  Income split at median=${inc_median:,.0f}: "
        + attrs["income_group"].value_counts().to_dict().__str__()
    )
    logger.info(
        "  Density groups: " + attrs["density_group"].value_counts().to_dict().__str__()
    )
    return attrs


# ── Group-level statistics ────────────────────────────────────────────────────

def group_stats(county_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute per-group summary statistics for a given subgroup dimension.

    Includes a Mann-Whitney U test comparing MAE distributions between each
    pair of groups.  For binary splits (2 groups), a single U p-value is
    reported.  For three-group density, pairwise p-values are reported.
    """
    rows = []
    groups = county_df[group_col].dropna().unique()

    # All individual group stats
    for grp in sorted(groups):
        sub = county_df[county_df[group_col] == grp]
        maes  = sub["mae"].values
        mapes = sub["mape"].dropna().values
        biases = sub["bias"].values

        rows.append({
            "dimension":     group_col,
            "group":         grp,
            "n_counties":    len(sub),
            "mae_mean":      round(float(np.mean(maes)),   8),
            "mae_median":    round(float(np.median(maes)), 8),
            "mae_std":       round(float(np.std(maes)),    8),
            "mae_min":       round(float(np.min(maes)),    8),
            "mae_max":       round(float(np.max(maes)),    8),
            "mape_mean":     round(float(np.mean(mapes)),  4) if len(mapes) else np.nan,
            "mape_median":   round(float(np.median(mapes)),4) if len(mapes) else np.nan,
            "bias_mean":     round(float(np.mean(biases)), 8),
            "pct_over":      round(float((biases > 0).mean() * 100), 1),
        })

    df_stats = pd.DataFrame(rows)

    # Pairwise Mann-Whitney U tests
    mw_results = {}
    group_list = sorted(groups)
    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            g1, g2 = group_list[i], group_list[j]
            v1 = county_df.loc[county_df[group_col] == g1, "mae"].values
            v2 = county_df.loc[county_df[group_col] == g2, "mae"].values
            if len(v1) >= 3 and len(v2) >= 3:
                stat, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                mw_results[f"{g1} vs {g2}"] = round(p, 4)

    return df_stats, mw_results


# ── Print results ─────────────────────────────────────────────────────────────

def print_subgroup_results(
    stats_df: pd.DataFrame,
    mw_results: dict,
    dimension: str,
) -> None:
    print(f"\n  {'─'*70}")
    print(f"  SUBGROUP: {dimension.upper()}")
    print(f"  {'─'*70}")
    print(f"  {'Group':<22} {'N':>4} {'MAE mean':>10} {'MAE med':>10} "
          f"{'MAPE%':>7} {'Bias mean':>11} {'% Over':>7}")
    print("  " + "─" * 70)
    for _, row in stats_df.iterrows():
        mape_s = f"{row['mape_mean']:.2f}%" if not pd.isna(row["mape_mean"]) else "    —"
        print(
            f"  {row['group']:<22} {int(row['n_counties']):>4} "
            f"{row['mae_mean']:>10.6f} {row['mae_median']:>10.6f} "
            f"{mape_s:>7} {row['bias_mean']:>+11.6f} {row['pct_over']:>6.0f}%"
        )
    if mw_results:
        print("  Mann-Whitney U tests (MAE):")
        for pair, p in mw_results.items():
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"    {pair}: p={p:.4f} {sig}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_subgroup_boxplots(
    county_df: pd.DataFrame,
    dimensions: list,
    out_path: str,
) -> None:
    """
    One row of box plots per subgroup dimension.
    Each box = one group within that dimension.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_dims = len(dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5), sharey=False)
    if n_dims == 1:
        axes = [axes]

    palette = {
        # Population
        "High population": "#1565C0", "Low population": "#90CAF9",
        # Income
        "High income": "#2E7D32",     "Low income": "#A5D6A7",
        # Density
        "Urban": "#E53935", "Suburban": "#FF8F00", "Rural": "#6D4C41",
    }

    for ax, dim_col in zip(axes, dimensions):
        groups  = sorted(county_df[dim_col].dropna().unique())
        data    = [county_df.loc[county_df[dim_col] == g, "mae"].values for g in groups]
        colors  = [palette.get(g, "#888888") for g in groups]

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        # Overlay individual county dots
        for k, (grp_data, color) in enumerate(zip(data, colors), 1):
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=len(grp_data))
            ax.scatter(k + jitter, grp_data, color=color, alpha=0.5, s=22, zorder=3,
                       edgecolors="white", linewidths=0.4)

        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, fontsize=9, rotation=10)
        ax.set_ylabel("MAE" if ax == axes[0] else "", fontsize=9)
        ax.set_title(dim_col.replace("_group", "").replace("_", " ").title(), fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Annotate group medians
        for k, (grp_data, grp) in enumerate(zip(data, groups), 1):
            med = np.median(grp_data)
            ax.text(k, med * 1.02, f"{med:.5f}", ha="center", va="bottom",
                    fontsize=7, color="black")

    fig.suptitle("Subgroup Performance Comparison — MAE by County Group\n"
                 "(dots = individual counties)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_mae_strips(county_df: pd.DataFrame, out_path: str) -> None:
    """
    Strip plots coloured by density group across all three subgroup dimensions,
    with a secondary axis showing the MAPE.
    """
    import matplotlib.pyplot as plt

    density_colors = {"Urban": "#E53935", "Suburban": "#FF8F00", "Rural": "#6D4C41"}
    dims = [
        ("population_group", "Population\n(High vs Low)"),
        ("income_group",     "Income\n(High vs Low)"),
        ("density_group",    "Density\n(Urban/Suburban/Rural)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    for ax, (dim_col, label) in zip(axes, dims):
        groups = sorted(county_df[dim_col].dropna().unique())
        rng = np.random.default_rng(99)

        for k, grp in enumerate(groups):
            sub    = county_df[county_df[dim_col] == grp]
            jitter = rng.uniform(-0.25, 0.25, size=len(sub))
            colors = [density_colors.get(d, "#888") for d in sub["density_group"]]

            ax.scatter(k + jitter, sub["mae"], c=colors, alpha=0.75,
                       s=40, edgecolors="white", linewidths=0.4, zorder=3)

            # Median line
            med = sub["mae"].median()
            ax.hlines(med, k - 0.35, k + 0.35, color="black", linewidth=2, zorder=4)
            ax.text(k, med * 1.015, f"med={med:.5f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=9, rotation=8)
        ax.set_title(label, fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("MAE (SNAP application rate)", fontsize=9)

    # Density colour legend
    handles = [plt.scatter([], [], c=c, s=50, label=g, alpha=0.8)
               for g, c in density_colors.items()]
    fig.legend(handles=handles, title="Density group\n(dot colour)",
               fontsize=9, title_fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    fig.suptitle("MAE by Subgroup — Dot colour = density tier, black line = group median",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


def plot_subgroup_heatmap(
    summary_df: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Heatmap: mean MAE for each (dimension × group) cell.
    Normalised within each dimension so colour is comparable across splits.
    """
    import matplotlib.pyplot as plt

    # Pivot to dimension × group
    pivot = summary_df.pivot(index="dimension", columns="group", values="mae_mean")
    pivot.index = [i.replace("_group", "").replace("_", " ").title() for i in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="Mean MAE", shrink=0.8)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10, rotation=15, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("Mean MAE by Subgroup\n(red = worse, green = better)", fontsize=11)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.5f}", ha="center", va="center",
                        fontsize=9.5, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str, default="xgb_tuned")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config.ensure_output_dirs()

    # ── Load county error analysis ────────────────────────────────────────────
    county_errors_path = os.path.join(
        config.OUTPUTS_ROOT, "metrics", f"county_error_analysis_{args.model}.csv"
    )
    if not os.path.exists(county_errors_path):
        raise FileNotFoundError(
            f"County error file not found: {county_errors_path}\n"
            f"Run first:  python experiments/county_error_analysis.py --model {args.model}"
        )
    county_df = pd.read_csv(county_errors_path)
    logger.info(f"  Loaded county errors: {county_errors_path}  ({len(county_df)} counties)")

    # ── Build subgroup labels ─────────────────────────────────────────────────
    logger.info("\n  Building subgroup labels...")
    attrs    = load_county_attributes()
    attrs    = assign_subgroups(attrs)

    # Merge — normalise county name casing
    attrs["county"] = attrs["county"].str.strip()
    county_df["county"] = county_df["county"].str.strip()
    merged = county_df.merge(attrs, on="county", how="left")

    n_unmatched = merged["population_group"].isna().sum()
    if n_unmatched:
        logger.warning(f"  {n_unmatched} counties had no attribute match — check name spelling")
        logger.warning(merged.loc[merged["population_group"].isna(), "county"].tolist())

    # ── Run group statistics ──────────────────────────────────────────────────
    dimensions = {
        "population_group": "Population size",
        "income_group":     "Median income",
        "density_group":    "Population density (urban/rural)",
    }

    all_stats = []
    for dim_col, dim_label in dimensions.items():
        stats_df, mw = group_stats(merged, dim_col)
        stats_df["dimension_label"] = dim_label
        all_stats.append(stats_df)
        print_subgroup_results(stats_df, mw, dim_label)

    summary_df = pd.concat(all_stats, ignore_index=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics_dir = os.path.join(config.OUTPUTS_ROOT, "metrics")
    merged.to_csv(os.path.join(metrics_dir, "subgroup_analysis.csv"), index=False)
    summary_df.to_csv(os.path.join(metrics_dir, "subgroup_summary.csv"), index=False)
    logger.info(f"\n  County table  → {metrics_dir}/subgroup_analysis.csv")
    logger.info(f"  Summary table → {metrics_dir}/subgroup_summary.csv")

    # ── Overall summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  KEY FINDINGS SUMMARY")
    print("=" * 75)
    for dim_col, dim_label in dimensions.items():
        sub = summary_df[summary_df["dimension"] == dim_col].copy()
        best = sub.loc[sub["mae_mean"].idxmin()]
        worst = sub.loc[sub["mae_mean"].idxmax()]
        ratio = worst["mae_mean"] / best["mae_mean"]
        print(f"\n  {dim_label}:")
        print(f"    Best  group: {best['group']:<22} mean MAE={best['mae_mean']:.6f}  "
              f"MAPE={best['mape_mean']:.1f}%")
        print(f"    Worst group: {worst['group']:<22} mean MAE={worst['mae_mean']:.6f}  "
              f"MAPE={worst['mape_mean']:.1f}%")
        print(f"    Performance gap (worst/best ratio): {ratio:.2f}×")
    print("=" * 75)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        fig_dir = config.FIGURES_DIR
        os.makedirs(fig_dir, exist_ok=True)

        plot_subgroup_boxplots(
            merged, list(dimensions.keys()),
            os.path.join(fig_dir, "subgroup_boxplots.png"),
        )
        plot_mae_strips(
            merged,
            os.path.join(fig_dir, "subgroup_mae_strips.png"),
        )
        plot_subgroup_heatmap(
            summary_df,
            os.path.join(fig_dir, "subgroup_heatmap.png"),
        )

    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
