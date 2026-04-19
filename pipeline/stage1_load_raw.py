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
import warnings

import numpy as np
import pandas as pd

from pipeline import config

logger = logging.getLogger(__name__)

# Minimum overlapping weeks between adjacent year chunks to compute a reliable
# rescaling ratio.  Fewer than this → ratio defaults to 1.0 (no rescaling).
_MIN_OVERLAP_WEEKS = 4


# ── Trend data ────────────────────────────────────────────────────────────────

def load_trend_csvs(keyword: str) -> pd.DataFrame:
    """
    Load, stitch, and rescale all per-DMA year-chunk Google Trends CSVs for
    one keyword (e.g. 'CalFresh' or 'FoodBank').

    Data layout on disk:
        TRENDS_DIR/{folder_name}/{DMA}/{DMA}{year}.csv
    where folder_name is looked up from config.TRENDS_FOLDER_MAP.

    Each annual CSV uses the Google Trends export format:
        "Time","CalFresh"
        "2017-01-01",50
        ...

    Because each file is independently normalized 0-100 within its own
    download window, values across different annual files are not directly
    comparable.  This function chain-rescales them into a single continuous
    series anchored to the most recent file, then re-normalizes to 0-100.

    Stitching algorithm (from scripts/stitch_trends.py, generalized):
      1. Sort all files for a DMA by their earliest date (oldest first).
      2. Anchor the newest file at scale = 1.0.
      3. Walk backwards through adjacent pairs (older, newer):
           overlap_dates = dates present in both files
           ratio = mean(newer[overlap]) / mean(older[overlap])
           scale[older] = ratio × scale[newer]
      4. Multiply every file's values by its cumulative scale factor.
      5. Merge all files; where dates overlap the newer file's scaled value wins.
      6. Re-normalize the full series to 0-100.

    Returns DataFrame with columns: metro_area, date (datetime), value (float).
    """
    folder_name = config.TRENDS_FOLDER_MAP.get(keyword)
    if not folder_name:
        logger.warning(f"No folder mapping for keyword '{keyword}' in config.TRENDS_FOLDER_MAP")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    keyword_dir = os.path.join(config.TRENDS_DIR, folder_name)
    if not os.path.exists(keyword_dir):
        logger.warning(f"Trends folder not found: {keyword_dir}")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    dma_dirs = sorted([
        d for d in os.listdir(keyword_dir)
        if os.path.isdir(os.path.join(keyword_dir, d)) and not d.startswith(".")
    ])

    if not dma_dirs:
        logger.warning(f"No DMA sub-directories found under {keyword_dir}")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    all_dfs = []
    for dma in dma_dirs:
        dma_dir = os.path.join(keyword_dir, dma)
        stitched = _stitch_dma(dma_dir, dma, keyword)
        if not stitched.empty:
            stitched["metro_area"] = dma   # folder name is the metro_area key
            all_dfs.append(stitched)

    if not all_dfs:
        logger.warning(f"No usable trend data found for keyword '{keyword}'")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"  {keyword}: {len(dma_dirs)} DMAs, "
        f"{combined['date'].min().date()} – {combined['date'].max().date()}, "
        f"{len(combined):,} weekly rows after stitching"
    )
    return combined[["metro_area", "date", "value"]]


def _read_annual_csv(path: str) -> pd.DataFrame:
    """
    Read one annual Google Trends CSV export.

    Handles quoted and unquoted column names.  The first column is always
    the date; the second is the Trends index value.

    Returns DataFrame with columns: date (datetime), value (float).
    Rows with unparseable dates or non-numeric values are dropped.
    """
    df = pd.read_csv(path)
    # Strip quotes from column names (Google Trends sometimes wraps them)
    df.columns = [c.strip().strip('"') for c in df.columns]

    date_col  = next((c for c in df.columns if c.lower() in ("time", "date")), None)
    value_col = next((c for c in df.columns if c.lower() not in ("time", "date")), None)

    if date_col is None or value_col is None:
        return pd.DataFrame(columns=["date", "value"])

    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"]  = pd.to_datetime(
        df["date"].astype(str).str.strip('"'), errors="coerce"
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (
        df[["date", "value"]]
        .dropna(subset=["date", "value"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _stitch_dma(dma_dir: str, dma_name: str, keyword: str) -> pd.DataFrame:
    """
    Chain-rescale all annual CSVs for one DMA into a single continuous series.

    Anchors on the most recent file (scale = 1.0) and works backwards,
    computing a cumulative scale factor for each older file using the ratio
    of means in the overlapping date window.  Where dates appear in multiple
    files the newer (scaled) file's value takes precedence.

    Final series is re-normalized to 0-100 to keep values in the same range
    as the raw Trends indices that the model was designed around.

    Returns DataFrame with columns: date (datetime), value (float 0-100).
    Returns empty DataFrame if no valid CSVs are found.
    """
    csv_files = sorted(glob.glob(os.path.join(dma_dir, "*.csv")))
    if not csv_files:
        logger.warning(f"  {dma_name}/{keyword}: no CSV files found")
        return pd.DataFrame(columns=["date", "value"])

    # Load all files; attach their earliest date for chronological sorting
    segments = []
    for path in csv_files:
        df = _read_annual_csv(path)
        if not df.empty:
            segments.append((df["date"].min(), df))

    if not segments:
        logger.warning(f"  {dma_name}/{keyword}: no readable CSV content")
        return pd.DataFrame(columns=["date", "value"])

    segments.sort(key=lambda x: x[0])   # oldest first

    n = len(segments)
    scale_factors = np.ones(n)

    # Walk backwards: compute how each older file relates to the next newer one
    for i in range(n - 2, -1, -1):
        _, older_df = segments[i]
        _, newer_df = segments[i + 1]

        overlap = set(older_df["date"]) & set(newer_df["date"])

        if len(overlap) < _MIN_OVERLAP_WEEKS:
            warnings.warn(
                f"{dma_name}/{keyword}: only {len(overlap)} overlapping weeks "
                f"between segments {i} and {i+1} — using scale=1.0"
            )
            scale_factors[i] = scale_factors[i + 1]
            continue

        old_mean = older_df.set_index("date").loc[list(overlap), "value"].mean()
        new_mean = newer_df.set_index("date").loc[list(overlap), "value"].mean()

        ratio = (new_mean / old_mean) if old_mean > 0 else 1.0
        scale_factors[i] = ratio * scale_factors[i + 1]

    # Apply scale factors; merge with newer file winning on duplicate dates
    combined: dict = {}
    for i, (_, df) in enumerate(segments):
        for _, row in df.iterrows():
            combined[row["date"]] = row["value"] * scale_factors[i]

    result = pd.DataFrame(
        sorted(combined.items()), columns=["date", "value"]
    )

    # Re-normalize to 0-100 so values stay interpretable as Trends indices
    vmax = result["value"].max()
    if vmax > 0:
        result["value"] = (result["value"] / vmax * 100).round(2)

    logger.debug(
        f"  {dma_name}/{keyword}: {n} chunks → "
        f"{result['date'].min().date()} – {result['date'].max().date()} "
        f"({len(result)} weeks)"
    )
    return result


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
    Load forward-period Google Trends CSVs from the prediction folder.

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
