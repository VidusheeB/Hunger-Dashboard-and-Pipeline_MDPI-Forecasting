"""
stage1_load_raw.py — Load every raw data source into clean DataFrames.

No merging or feature engineering happens here — each function loads exactly
one source file and returns a validated DataFrame. Missing files produce an
empty DataFrame with a WARNING, so the pipeline can report clearly what's
missing rather than crashing silently.
"""

import os
import glob
import logging

import pandas as pd

from pipeline import config

logger = logging.getLogger(__name__)


# ── Trend data ────────────────────────────────────────────────────────────────

def load_trend_csvs(keyword: str) -> pd.DataFrame:
    """
    Read all per-DMA weekly trend CSVs for one keyword (e.g. 'CalFresh').

    Each file is named {DMA}.csv and contains two headerless columns:
        date (YYYY-MM-DD), value (0-100 Google Trends index)

    Returns DataFrame with columns: metro_area, date (datetime), value (float).
    Rows with unparseable dates or non-numeric values are dropped.
    """
    pattern = os.path.join(config.TRENDS_DIR, keyword, "*.csv")
    files = glob.glob(pattern)

    if not files:
        logger.warning(f"No trend files found for keyword '{keyword}' at {pattern}")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    dfs = []
    for fpath in files:
        metro = os.path.splitext(os.path.basename(fpath))[0]
        df = pd.read_csv(fpath, header=None, names=["date", "value"])
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df["metro_area"] = metro
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"  {keyword}: {len(files)} DMA files, "
        f"{combined['metro_area'].nunique()} DMAs, "
        f"{combined['date'].min().date()} – {combined['date'].max().date()}, "
        f"{len(combined):,} weekly rows"
    )
    return combined[["metro_area", "date", "value"]]


# ── SNAP applications ─────────────────────────────────────────────────────────

def load_snap_applications() -> pd.DataFrame:
    """
    Load monthly SNAP application counts by county from SNAPData.csv.

    The file has no header; columns are: county, month_year, SNAP_Applications.
    Date formats handled: 'Jan 2023' and 'January 2023'.
    '*' values (suppressed by the state) are treated as NaN.

    Temporal alignment: trend data predicts SNAP applications 1 month later,
    so trend_date = snap_date - 1 month. The model is trained with trends joined
    on trend_date, so it learns "last month's search signal → this month's apps."

    Returns columns: county, date (SNAP month), trend_date, SNAP_Applications.
    """
    if not os.path.exists(config.SNAP_FILE):
        logger.warning(f"SNAP file not found: {config.SNAP_FILE}")
        return pd.DataFrame(columns=["county", "date", "trend_date", "SNAP_Applications"])

    df = pd.read_csv(
        config.SNAP_FILE,
        header=None,
        names=["county", "month_year", "SNAP_Applications"],
    )

    # Parse abbreviated and full month names
    df["date"] = pd.to_datetime(df["month_year"].str.strip(), format="%b %Y", errors="coerce")
    mask_failed = df["date"].isna()
    df.loc[mask_failed, "date"] = pd.to_datetime(
        df.loc[mask_failed, "month_year"].str.strip(), format="%B %Y", errors="coerce"
    )

    # Replace suppressed values, coerce to numeric
    df["SNAP_Applications"] = pd.to_numeric(
        df["SNAP_Applications"].replace("*", pd.NA), errors="coerce"
    )

    # Temporal shift: join trends from 1 month prior
    df["trend_date"] = df["date"] - pd.DateOffset(months=1)

    result = df[["county", "date", "trend_date", "SNAP_Applications"]].copy()
    logger.info(
        f"  SNAP: {result['county'].nunique()} counties, "
        f"{len(result):,} rows, "
        f"{result['date'].min().date()} – {result['date'].max().date()}, "
        f"{result['SNAP_Applications'].isna().sum()} NaN values"
    )
    return result


# ── Supporting lookup tables ──────────────────────────────────────────────────

def load_county_metro() -> pd.DataFrame:
    """
    Load the county → DMA (metro area) mapping.

    Returns columns: county, metro_area.
    Column names are stripped of whitespace to handle export artifacts.
    """
    if not os.path.exists(config.COUNTY_METRO_FILE):
        logger.warning(f"County-metro map not found: {config.COUNTY_METRO_FILE}")
        return pd.DataFrame(columns=["county", "metro_area"])

    df = pd.read_csv(config.COUNTY_METRO_FILE)
    df.columns = df.columns.str.strip()
    logger.info(
        f"  County-metro: {len(df)} mappings, "
        f"{df['metro_area'].nunique()} unique DMAs"
    )
    return df[["county", "metro_area"]]


def load_population() -> pd.DataFrame:
    """
    Load county population estimates.

    Returns columns: county, Population (int).
    """
    if not os.path.exists(config.POP_FILE):
        logger.warning(f"Population file not found: {config.POP_FILE}")
        return pd.DataFrame(columns=["county", "Population"])

    df = pd.read_csv(config.POP_FILE)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"County": "county"})
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    logger.info(f"  Population: {len(df)} counties")
    return df[["county", "Population"]]


def load_income() -> pd.DataFrame:
    """
    Load county median household income.

    'Median Income' is stored as a comma-formatted string (e.g. '75,000').
    A join key 'county_key' strips spaces so 'San Benito' matches 'SanBenito'
    in the SNAP table.

    Returns columns: county_key, Median_Income (float).
    """
    if not os.path.exists(config.INCOME_FILE):
        logger.warning(f"Income file not found: {config.INCOME_FILE}")
        return pd.DataFrame(columns=["county_key", "Median_Income"])

    df = pd.read_csv(config.INCOME_FILE)
    df["Median_Income"] = (
        df["Median Income"].astype(str).str.replace(",", "").pipe(pd.to_numeric, errors="coerce")
    )
    df["county_key"] = df["County"].str.replace(" ", "")
    logger.info(f"  Income: {len(df)} counties")
    return df[["county_key", "Median_Income"]]


# ── Forward prediction trends ─────────────────────────────────────────────────

def load_prediction_trends(keyword: str) -> pd.DataFrame:
    """
    Load current-month Google Trends CSVs from the prediction folder.

    These are in the Google Trends export format (quoted header row).
    Handles both old format ('Category: All categories' row 0) and new
    format ('"Time","keyword"' row 0).

    Returns columns: metro_area, date (datetime), value (float).
    """
    kw_dir = os.path.join(config.PREDICTION_DIR, keyword)
    if not os.path.exists(kw_dir):
        logger.warning(f"Prediction dir not found: {kw_dir}")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    dfs = []
    for fpath in glob.glob(os.path.join(kw_dir, "*.csv")):
        metro = os.path.splitext(os.path.basename(fpath))[0]
        df = _read_prediction_csv(fpath)
        if not df.empty:
            df["metro_area"] = metro
            dfs.append(df)

    if not dfs:
        logger.warning(f"No readable prediction files for {keyword}")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"  Prediction/{keyword}: {combined['metro_area'].nunique()} DMAs, "
        f"{combined['date'].min().date()} – {combined['date'].max().date()}"
    )
    return combined[["metro_area", "date", "value"]]


def _read_prediction_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse a Google Trends export CSV regardless of format variant.
    Returns DataFrame with columns: date (datetime), value (float).
    Metadata rows and non-date rows are filtered out automatically.
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Find the line that starts the actual time-series data
    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('"Time"') or s.startswith("Time,") or s.startswith("Day,"):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.read_csv(csv_path, skiprows=header_idx)
    if len(df.columns) < 2:
        return pd.DataFrame(columns=["date", "value"])

    df.columns = ["date", "value"] + list(df.columns[2:])
    df = df[["date", "value"]]

    # Drop Google Trends metadata rows that share the date column
    non_data = {
        "Category: All categories", "Region:", "Week", "Day", "Month", "Year",
        "United States", "State", "City", "Metro", "Subregion", "Search term",
        "Note:", "Notes:", "Interest over time", "Time", "Geo", "isPartial",
        "date", "value", "Average", "Total", "N/A", "nan", "", None,
    }
    df = df[~df["date"].astype(str).str.strip().isin(non_data)]

    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"])
