"""
build_snap_csv.py
=================
Parse all CF296 xlsx files (FY2016-17 through FY2025-26) and write a clean
SNAPData.csv with columns: county, month_year, SNAP_Applications.

Three xlsx formats encountered:
  A. FY16-17 to FY19-20  — sheet "FinalData",      header row 4, county col 3, apps col 6
  B. FY20-21 to FY24-25  — sheet "Data_External",  header row 4, county col 1, apps col 7
  C. FY25-26             — sheet "Data_External",  header row 5, county col 1, apps col 7

Run from project root:
    python scripts/build_snap_csv.py
"""

import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

SNAP_DIR = os.path.join(config.RAW_DATA_ROOT, "SNAPApps")
OUT_PATH = config.SNAP_FILE  # src/data/SNAPApps/SNAPData.csv

# ── File-format specs ─────────────────────────────────────────────────────────
# (glob_pattern, sheet_name, header_row, county_col, date_col, apps_col)
FORMAT_SPECS = [
    # Format A: FY16-19
    ("CF296FY16-17.xlsx",   "FinalData",    4, 3, 0, 6),
    ("CF296FY17-18.xlsx",   "FinalData",    4, 3, 0, 6),
    ("CF296FY18-19.xlsx",   "FinalData",    4, 3, 0, 6),
    ("CF296FY19-20.xlsx",   "FinalData",    4, 3, 0, 6),
    # Format B: FY20-25
    ("CF296 FY 2020-21 (EXTERNAL) 2021-09-07.xlsx", "Data_External", 4, 1, 0, 7),
    ("CF296FY21-22 (1).xlsx",  "Data_External", 4, 1, 0, 7),
    ("CF296FY22-23 (1).xlsx",  "Data_External", 4, 1, 0, 7),
    ("CF296FY23-24.xlsx",      "Data_External", 4, 1, 0, 7),
    ("CF296FY24-25.xlsx",      "Data_External", 4, 1, 0, 7),
    # Format C: FY25-26 (header row 5)
    ("CF296 FY 2025-26 (2).xlsx", "Data_External", 5, 1, 0, 7),
]

EXCLUDE_COUNTIES = {"Statewide", "statewide", "STATEWIDE"}


def parse_one_file(filename, sheet, header_row, county_col, date_col, apps_col):
    path = os.path.join(SNAP_DIR, filename)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {filename}")
        return pd.DataFrame()

    df = pd.read_excel(path, sheet_name=sheet, header=None)

    # Slice from the header row downward
    data = df.iloc[header_row + 1:].copy()
    data.columns = range(len(data.columns))

    out = pd.DataFrame()
    out["county"] = data[county_col].astype(str).str.strip()
    out["date"]   = pd.to_datetime(data[date_col], errors="coerce")
    out["SNAP_Applications"] = data[apps_col]

    # Drop statewide totals and rows with unparseable dates / county
    out = out[~out["county"].isin(EXCLUDE_COUNTIES)]
    out = out.dropna(subset=["date", "county"])
    out = out[out["county"].str.len() > 0]

    # Format date as "Mon YYYY" string (pipeline stage1 expects this)
    out["month_year"] = out["date"].dt.strftime("%b %Y")

    # Replace suppressed "*" values with NaN
    out["SNAP_Applications"] = pd.to_numeric(
        out["SNAP_Applications"].astype(str).str.strip().replace("*", ""),
        errors="coerce"
    )

    n = len(out)
    date_min = out["date"].min().strftime("%b %Y") if not out.empty else "?"
    date_max = out["date"].max().strftime("%b %Y") if not out.empty else "?"
    print(f"  {filename:<55} {n:>4} rows  {date_min} – {date_max}")

    return out[["county", "month_year", "SNAP_Applications"]]


def main():
    print("Building SNAPData.csv from CF296 xlsx files...\n")

    frames = []
    for spec in FORMAT_SPECS:
        df = parse_one_file(*spec)
        if not df.empty:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep first occurrence (earlier file) for same county+month
    combined["_sort_date"] = pd.to_datetime(combined["month_year"], format="%b %Y", errors="coerce")
    combined = combined.sort_values(["county", "_sort_date"])
    before = len(combined)
    combined = combined.drop_duplicates(subset=["county", "month_year"], keep="first")
    combined = combined.drop(columns=["_sort_date"])
    after = len(combined)

    print(f"\nTotal rows: {before} → {after} after deduplication")

    date_check = pd.to_datetime(combined["month_year"], format="%b %Y", errors="coerce")
    print(f"Date range: {date_check.min().strftime('%b %Y')} – {date_check.max().strftime('%b %Y')}")
    print(f"Counties:   {combined['county'].nunique()}")

    combined.to_csv(OUT_PATH, index=False, header=False)
    print(f"\nWritten to: {OUT_PATH}")


if __name__ == "__main__":
    main()
