"""
stage2_build_features.py — Merge raw data into the canonical modeling table.

Steps:
  1. Monthly-aggregate per-DMA weekly trends for each keyword
  2. Load and interpolate SNAP application data
  3. Remove outlier SNAP rows (data-entry spikes)
  4. Merge SNAP + county-metro + trends + population + income
  5. Compute the target variable (SNAP_Application_Rate = apps / population)
  6. Compute per-DMA scaling params needed at prediction time
  7. Save training_data.csv and trend_scaling_params.json

Outputs:
  outputs/data/training_data.csv
  outputs/data/trend_scaling_params.json
"""

import json
import logging

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.stage1_load_raw import (
    load_trend_csvs,
    load_snap_applications,
    load_county_metro,
    load_population,
    load_income,
)

logger = logging.getLogger(__name__)


# ── Step 1: Aggregate weekly trends to monthly ────────────────────────────────

def aggregate_trends_monthly(raw_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Convert per-DMA weekly Trends rows to monthly averages.

    The 0-100 Google Trends scale is preserved — no population normalization.
    Population enters as a separate model feature, so dividing trends by it
    would destroy the cross-DMA comparability of the signal.

    Returns columns: metro_area, date (first of month), monthly_average_{keyword}.
    """
    if raw_df.empty:
        return pd.DataFrame(columns=["metro_area", "date", f"monthly_average_{keyword}"])

    raw_df = raw_df.copy()
    raw_df["year"]  = raw_df["date"].dt.year
    raw_df["month"] = raw_df["date"].dt.month

    monthly = (
        raw_df
        .groupby(["metro_area", "year", "month"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": f"monthly_average_{keyword}"})
    )
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    return monthly[["metro_area", "date", f"monthly_average_{keyword}"]]


# ── Step 2: Interpolate missing SNAP values ───────────────────────────────────

def interpolate_snap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing SNAP_Applications per county using:
      1. Linear interpolation for interior gaps (between valid values)
      2. County-mean fill for edge gaps (leading/trailing NaNs)

    Monthly SNAP data sometimes has suppressed ('*') values for small counties.
    Interpolation avoids dropping those counties entirely from the model.
    """
    result = []
    for county, grp in df.groupby("county"):
        grp = grp.sort_values("date").copy()
        grp["SNAP_Applications"] = grp["SNAP_Applications"].interpolate(method="linear")
        county_mean = grp["SNAP_Applications"].mean()
        grp["SNAP_Applications"] = grp["SNAP_Applications"].fillna(county_mean)
        result.append(grp)
    return pd.concat(result, ignore_index=True)


# ── Step 3: Remove outlier SNAP rows ─────────────────────────────────────────

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where SNAP_Applications > OUTLIER_THRESHOLD × county median.

    This catches data-entry errors without needing manual inspection.
    Example: Madera Jan 2023 reported 11,090 applications vs a typical ~1,000 —
    a 10× spike that would distort the model's rate target.
    """
    before = len(df)

    def flag(grp):
        median = grp["SNAP_Applications"].median()
        return grp[grp["SNAP_Applications"] <= config.OUTLIER_THRESHOLD * median]

    df = df.groupby("county", group_keys=False).apply(flag).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.info(f"  Removed {removed} outlier SNAP row(s) (> {config.OUTLIER_THRESHOLD}× county median)")
    return df


# ── Step 4: Compute per-DMA scaling params ────────────────────────────────────

def compute_scaling_params(trend_frames: dict) -> dict:
    """
    Compute per-DMA training averages for each keyword.

    At prediction time, we rescale the forward-period trends to the training
    reference frame using: scaled = latest_avg × (train_avg / pred_window_avg).
    This preserves spike signals while correcting for different download windows.

    Returns: {keyword: {metro_area: mean_monthly_average}}
    Saves to outputs/data/trend_scaling_params.json.
    """
    scaling_params = {}
    for kw, monthly_df in trend_frames.items():
        col = f"monthly_average_{kw}"
        params = (
            monthly_df.groupby("metro_area")[col]
            .mean()
            .round(4)
            .to_dict()
        )
        scaling_params[kw] = params
        logger.info(f"  Scaling params: {kw} — {len(params)} DMAs")

    with open(config.SCALING_PARAMS_JSON, "w") as f:
        json.dump(scaling_params, f, indent=2)
    logger.info(f"  Saved → {config.SCALING_PARAMS_JSON}")
    return scaling_params


# ── Main entry point ──────────────────────────────────────────────────────────

def build_training_data() -> pd.DataFrame:
    """
    Orchestrate the full data build:
      load → aggregate trends → interpolate SNAP → remove outliers
      → merge all tables → compute target → save

    Returns the final training DataFrame.
    """
    logger.info("=== STAGE 2: BUILD FEATURES ===")

    # Load supporting lookup tables
    county_metro = load_county_metro()
    pop_df       = load_population()
    income_df    = load_income()

    # Load and monthly-aggregate trend data for each keyword
    logger.info("Loading trend data...")
    trend_frames = {}
    for kw in config.KEYWORDS:
        raw = load_trend_csvs(kw)
        if not raw.empty:
            trend_frames[kw] = aggregate_trends_monthly(raw, kw)

    # Save per-DMA scaling params for use at prediction time (stage 5)
    compute_scaling_params(trend_frames)

    # Load SNAP data
    logger.info("Loading SNAP application data...")
    snap_df = load_snap_applications()
    snap_df = interpolate_snap(snap_df)

    # Start with SNAP as the base table; add metro area from county map
    base = snap_df.merge(county_metro, on="county", how="left")

    # Join trend features — trends are matched on trend_date (1 month before SNAP date),
    # which was set in load_snap_applications() to implement the temporal lag
    for kw, monthly_df in trend_frames.items():
        tdf = monthly_df.rename(columns={"date": "trend_date"})
        base = base.merge(tdf, on=["metro_area", "trend_date"], how="left")

    # Join population (needed for rate computation and as a model feature)
    base = base.merge(pop_df[["county", "Population"]], on="county", how="left")

    # Join income (spaces stripped to handle name mismatches like 'San Benito'/'SanBenito')
    base["county_key"] = base["county"].str.replace(" ", "")
    base = base.merge(income_df[["county_key", "Median_Income"]], on="county_key", how="left")
    base = base.drop(columns=["county_key"])

    # Compute target variable: SNAP application rate per capita
    base[config.TARGET_COL] = base["SNAP_Applications"] / base["Population"]

    # Remove outlier SNAP rows after rate computation
    base = remove_outliers(base)

    # Add month feature (captures seasonal patterns in SNAP applications)
    base["month"] = pd.to_datetime(base["date"]).dt.month

    # Drop rows with no metro mapping (counties not in county_to_metro.csv)
    before = len(base)
    base = base.dropna(subset=["metro_area"])
    if len(base) < before:
        logger.info(f"  Dropped {before - len(base)} rows with no metro mapping")

    # Canonical column order for the output CSV
    trend_cols = [f"monthly_average_{kw}" for kw in config.KEYWORDS if kw in trend_frames]
    keep_cols  = [
        "county", "date", "metro_area",
        "SNAP_Applications", "Population", "Median_Income",
        "month", config.TARGET_COL,
    ] + trend_cols
    base = base[[c for c in keep_cols if c in base.columns]].sort_values(["county", "date"])
    base = base.reset_index(drop=True)

    # Save
    base.to_csv(config.TRAINING_DATA_CSV, index=False)

    # Summary
    logger.info(f"\n  Output: {config.TRAINING_DATA_CSV}")
    logger.info(f"  Shape:  {base.shape}")
    logger.info(f"  Counties: {base['county'].nunique()}, DMAs: {base['metro_area'].nunique()}")
    logger.info(f"  Date range: {base['date'].min()} → {base['date'].max()}")
    logger.info(
        f"  {config.TARGET_COL}: "
        f"min={base[config.TARGET_COL].min():.5f}, "
        f"mean={base[config.TARGET_COL].mean():.5f}, "
        f"max={base[config.TARGET_COL].max():.5f}"
    )
    nan_counts = base[trend_cols].isna().sum()
    for col, n in nan_counts.items():
        if n > 0:
            logger.info(f"  Warning: {col} has {n} NaN values")

    return base
