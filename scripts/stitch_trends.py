"""
stitch_trends.py
================
Stitch multiple overlapping annual Google Trends CSVs per DMA into a single
continuous series for CalFresh, and write one CSV per DMA to
src/data/trends/CalFresh/ (the format stage1_load_raw.py expects).

Google Trends problem:
  Each annual download is independently normalized 0-100 within its own window.
  You cannot concatenate them directly — a "50" in 2018 and a "50" in 2020
  are not comparable without accounting for the relative scale between periods.

Stitching algorithm (chain-rescaling):
  1. Sort all files by their start date.
  2. Anchor on the most recent file (scale = 1.0).
  3. For each consecutive pair (older → newer), find overlapping weeks.
  4. Compute scale = mean(newer_values_in_overlap) / mean(older_values_in_overlap).
  5. Apply cumulative scale to all older files.
  6. Combine all files; where dates overlap take the newer file's (already-scaled) values.
  7. Re-normalize the final series to 0-100 so values stay interpretable.

Input:  src/data/trends/CalFresh2017-2025/{DMA}/{DMA}{Year}.csv
        Columns: "Time" (YYYY-MM-DD), "CalFresh" (int 0-100)

Output: src/data/trends/CalFresh/{DMA}.csv
        Two columns, no header: date (YYYY-MM-DD), value (float)
        (matches format expected by pipeline/stage1_load_raw.py)

Run from project root:
    python scripts/stitch_trends.py
"""

import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

RAW_TRENDS_BASE = os.path.join(config.RAW_DATA_ROOT, "trends", "CalFresh2017-2025")
OUT_DIR         = os.path.join(config.TRENDS_DIR, "CalFresh")

MIN_OVERLAP_WEEKS = 4   # minimum shared weeks required to compute a reliable scale factor


def read_annual_csv(path: str) -> pd.DataFrame:
    """Read one annual Trends CSV; return DataFrame with columns: date (datetime), value (float)."""
    df = pd.read_csv(path)
    # Normalise column names (files use "Time" and "CalFresh")
    df.columns = [c.strip().strip('"') for c in df.columns]
    date_col  = [c for c in df.columns if c.lower() in ("time", "date")][0]
    value_col = [c for c in df.columns if c.lower() not in ("time", "date")][0]

    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"]  = pd.to_datetime(df["date"].astype(str).str.strip('"'), errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "value"]]


def stitch_dma(dma_dir: str, dma_name: str) -> pd.DataFrame:
    """
    Load all annual CSVs for one DMA and chain-rescale into a single series.
    Returns DataFrame with columns: date, value (stitched, re-normalised 0-100).
    """
    csv_files = sorted(glob.glob(os.path.join(dma_dir, "*.csv")))
    if not csv_files:
        print(f"  {dma_name}: no CSV files found — skipping")
        return pd.DataFrame(columns=["date", "value"])

    # Load all files and sort by their start date
    segments = []
    for path in csv_files:
        df = read_annual_csv(path)
        if not df.empty:
            segments.append((df["date"].min(), path, df))

    segments.sort(key=lambda x: x[0])   # oldest first
    if not segments:
        return pd.DataFrame(columns=["date", "value"])

    # Build cumulative scale factors working backwards from the newest segment
    # scale_factors[i] = factor to multiply segment i's values by
    n = len(segments)
    scale_factors = np.ones(n)

    for i in range(n - 2, -1, -1):           # n-2 down to 0
        _, _, older_df  = segments[i]
        _, _, newer_df  = segments[i + 1]

        overlap_dates = set(older_df["date"]) & set(newer_df["date"])
        if len(overlap_dates) < MIN_OVERLAP_WEEKS:
            warnings.warn(
                f"  {dma_name}: only {len(overlap_dates)} overlapping weeks between "
                f"segment {i} and {i+1} — using scale=1.0"
            )
            scale_factors[i] = scale_factors[i + 1]   # propagate without scaling
            continue

        old_vals = older_df.set_index("date").loc[list(overlap_dates), "value"]
        new_vals = newer_df.set_index("date").loc[list(overlap_dates), "value"]

        old_mean = old_vals.mean()
        new_mean = new_vals.mean()

        if old_mean == 0:
            ratio = 1.0
        else:
            ratio = new_mean / old_mean

        # Cumulative: this segment's absolute scale = ratio × newer segment's scale
        scale_factors[i] = ratio * scale_factors[i + 1]

    # Apply scales and merge (newer file wins on duplicate dates)
    # Build a dict date → scaled_value; process oldest first so newer overwrites
    combined: dict[pd.Timestamp, float] = {}
    for i, (_, _, df) in enumerate(segments):
        scaled = df.copy()
        scaled["value"] = scaled["value"] * scale_factors[i]
        for _, row in scaled.iterrows():
            combined[row["date"]] = row["value"]

    result = pd.DataFrame(
        [(d, v) for d, v in sorted(combined.items())],
        columns=["date", "value"]
    )

    # Re-normalise to 0-100 so values remain interpretable as Trends indices
    vmax = result["value"].max()
    if vmax > 0:
        result["value"] = (result["value"] / vmax * 100).round(2)

    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Stitching CalFresh Trends → {OUT_DIR}\n")

    dma_dirs = sorted([
        d for d in os.listdir(RAW_TRENDS_BASE)
        if os.path.isdir(os.path.join(RAW_TRENDS_BASE, d)) and not d.startswith(".")
    ])

    if not dma_dirs:
        print(f"No DMA subdirectories found under {RAW_TRENDS_BASE}")
        sys.exit(1)

    for dma_name in dma_dirs:
        dma_dir = os.path.join(RAW_TRENDS_BASE, dma_name)
        stitched = stitch_dma(dma_dir, dma_name)

        if stitched.empty:
            continue

        out_path = os.path.join(OUT_DIR, f"{dma_name}.csv")
        # Write without header (stage1_load_raw expects headerless: date, value)
        stitched.to_csv(out_path, index=False, header=False)

        n_files = len(glob.glob(os.path.join(dma_dir, "*.csv")))
        print(
            f"  {dma_name:<40} {n_files} files → "
            f"{stitched['date'].min().date()} – {stitched['date'].max().date()} "
            f"({len(stitched)} weeks)"
        )

    print(f"\nDone. {len(dma_dirs)} DMAs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
