"""
Build Training Data

Combines Google Trends data with SNAP application data to produce:
  - src/data/training_data.csv         (features + target for model training)
  - src/data/trend_scaling_params.json (per-DMA training averages for prediction scaling)

Run this when raw source data changes (new SNAP data, new trend downloads).
"""

import os
import glob
import json
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
TRENDS_DIR       = "src/data/trends"
SNAP_FILE        = "src/data/SNAPApps/SNAPData.csv"
COUNTY_METRO     = "src/data/county_to_metro.csv"
POP_FILE         = "src/data/popData.csv"
INCOME_FILE      = "src/data/MedianIncome.csv"
OUTPUT_CSV       = "src/data/training_data.csv"
OUTPUT_PARAMS    = "src/data/trend_scaling_params.json"
KEYWORDS         = ["CalFresh", "FoodBank"]


# ── Step 1: Load and monthly-aggregate trend CSVs ─────────────────────────────
def load_trends(keyword):
    """
    Read all per-DMA weekly trend CSVs for a keyword.
    Returns a DataFrame with columns: metro_area, date (first of month), monthly_average_{keyword}
    """
    pattern = os.path.join(TRENDS_DIR, keyword, "*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"  WARNING: no trend files found for {keyword}")
        return pd.DataFrame()

    dfs = []
    for fpath in files:
        metro = os.path.splitext(os.path.basename(fpath))[0]
        df = pd.read_csv(fpath, header=None, names=["date", "value"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df["metro_area"] = metro
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["year"]  = combined["date"].dt.year
    combined["month"] = combined["date"].dt.month

    monthly = (
        combined
        .groupby(["metro_area", "year", "month"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": f"monthly_average_{keyword}"})
    )
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    return monthly[["metro_area", "date", f"monthly_average_{keyword}"]]


# ── Step 2: Load SNAP data ────────────────────────────────────────────────────
def load_snap():
    """
    Load SNAP application counts. Applies a 1-month temporal shift so that
    month-N trends are paired with month-(N+1) SNAP applications.
    Returns DataFrame with columns: county, date (SNAP month), SNAP_Applications, trend_date
    """
    df = pd.read_csv(SNAP_FILE, header=None, names=["county", "month_year", "SNAP_Applications"])
    df["date"] = pd.to_datetime(df["month_year"].str.strip(), format="%b %Y", errors="coerce")
    df.loc[df["date"].isna(), "date"] = pd.to_datetime(
        df.loc[df["date"].isna(), "month_year"].str.strip(), format="%B %Y", errors="coerce"
    )
    df["SNAP_Applications"] = pd.to_numeric(
        df["SNAP_Applications"].replace("*", pd.NA), errors="coerce"
    )
    # Temporal shift: trend_date is 1 month before the SNAP month
    df["trend_date"] = df["date"] - pd.DateOffset(months=1)
    return df[["county", "date", "trend_date", "SNAP_Applications"]]


# ── Step 3: Interpolate missing SNAP values per county ────────────────────────
def interpolate_snap(df):
    """
    Fill missing SNAP_Applications per county:
    1. Linear interpolation (interior gaps)
    2. County mean for any remaining NaNs (edge gaps)
    """
    result = []
    for county, grp in df.groupby("county"):
        grp = grp.sort_values("date").copy()
        grp["SNAP_Applications"] = grp["SNAP_Applications"].interpolate(method="linear")
        county_mean = grp["SNAP_Applications"].mean()
        grp["SNAP_Applications"] = grp["SNAP_Applications"].fillna(county_mean)
        result.append(grp)
    return pd.concat(result, ignore_index=True)


# ── Step 4: Remove outlier SNAP rows ─────────────────────────────────────────
def remove_snap_outliers(df):
    """
    Drop rows where SNAP_Applications > 3× the county's own median.
    Catches data-entry errors (e.g. Madera Jan 2023 = 11,090 vs typical ~1,000).
    """
    def flag_outliers(grp):
        median = grp["SNAP_Applications"].median()
        return grp[grp["SNAP_Applications"] <= 3 * median]

    before = len(df)
    df = df.groupby("county", group_keys=False).apply(flag_outliers).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed} outlier SNAP row(s) (> 3× county median)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def build_training_data():
    print("=== BUILD TRAINING DATA ===\n")

    # Load supporting tables
    county_metro = pd.read_csv(COUNTY_METRO)
    county_metro.columns = county_metro.columns.str.strip()

    pop_df = pd.read_csv(POP_FILE)
    pop_df.columns = pop_df.columns.str.strip()
    pop_df = pop_df.rename(columns={"County": "county"})

    income_df = pd.read_csv(INCOME_FILE)
    income_df["Median_Income"] = (
        income_df["Median Income"].str.replace(",", "").astype(float)
    )
    # Normalize county names: strip spaces so "San Benito" matches "SanBenito" etc.
    income_df["county_key"] = income_df["County"].str.replace(" ", "")

    # Load and aggregate trend data
    print("Loading trend data...")
    trend_frames = {}
    scaling_params = {}
    for kw in KEYWORDS:
        print(f"  {kw}...")
        tdf = load_trends(kw)
        if tdf.empty:
            continue
        trend_frames[kw] = tdf
        # Compute per-DMA training average (on 0-100 scale) for prediction scaling
        avg_col = f"monthly_average_{kw}"
        params = (
            tdf.groupby("metro_area")[avg_col]
            .mean()
            .round(4)
            .to_dict()
        )
        scaling_params[kw] = params
        print(f"    {len(tdf['metro_area'].unique())} DMAs loaded")

    # Save scaling params for use at prediction time
    with open(OUTPUT_PARAMS, "w") as f:
        json.dump(scaling_params, f, indent=2)
    print(f"\nSaved trend scaling params → {OUTPUT_PARAMS}")

    # Load SNAP data
    print("\nLoading SNAP data...")
    snap_df = load_snap()
    print(f"  {snap_df['county'].nunique()} counties, {len(snap_df)} rows")

    # Interpolate missing SNAP values
    snap_df = interpolate_snap(snap_df)

    # Build base: all (county, trend_date) combinations from SNAP data
    base = snap_df.copy()
    base = base.merge(county_metro, on="county", how="left")

    # Join trends: match trend_date → date column in trend data
    for kw in KEYWORDS:
        if kw not in trend_frames:
            continue
        tdf = trend_frames[kw].rename(columns={"date": "trend_date"})
        base = base.merge(tdf, on=["metro_area", "trend_date"], how="left")

    # Join population and income
    base = base.merge(pop_df[["county", "Population"]], on="county", how="left")
    base["county_key"] = base["county"].str.replace(" ", "")
    base = base.merge(income_df[["county_key", "Median_Income"]], on="county_key", how="left")
    base = base.drop(columns=["county_key"])

    # Compute target variable
    base["SNAP_Application_Rate"] = base["SNAP_Applications"] / base["Population"]

    # Remove outliers
    base = remove_snap_outliers(base)

    # Add month feature
    base["month"] = pd.to_datetime(base["date"]).dt.month

    # Drop rows with no metro mapping (unmapped counties)
    before = len(base)
    base = base.dropna(subset=["metro_area"])
    if len(base) < before:
        print(f"  Dropped {before - len(base)} rows with no metro mapping")

    # Final column order
    trend_cols = [f"monthly_average_{kw}" for kw in KEYWORDS if kw in trend_frames]
    keep_cols = ["county", "date", "metro_area", "SNAP_Applications", "Population",
                 "Median_Income", "month", "SNAP_Application_Rate"] + trend_cols
    base = base[[c for c in keep_cols if c in base.columns]].sort_values(["county", "date"])

    base.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Shape:  {base.shape}")
    print(f"DMAs:   {sorted(base['metro_area'].dropna().unique())}")
    print(f"Counties: {base['county'].nunique()}")
    print(f"Date range: {base['date'].min()} → {base['date'].max()}")
    print(f"SNAP_Application_Rate: {base['SNAP_Application_Rate'].min():.5f} – {base['SNAP_Application_Rate'].max():.5f}")
    for kw in KEYWORDS:
        col = f"monthly_average_{kw}"
        if col in base.columns:
            nans = base[col].isna().sum()
            print(f"{col}: min={base[col].min():.1f}  mean={base[col].mean():.1f}  max={base[col].max():.1f}  NaN={nans}")

    return base


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_training_data()
