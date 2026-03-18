"""
audit_data.py — Data quality audit for the SNAP prediction pipeline.

Checks:
  1. Missing values in training_data.csv and all raw source files
  2. Duplicate rows
  3. Date coverage by county (expected vs actual months)
  4. Counties with incomplete records
  5. DMA-to-county mapping issues
  6. Outliers in application counts and per-capita rates

Outputs:
  outputs/audit/audit_report.csv  — machine-readable row-per-issue report
  stdout                          — human-readable summary

Usage:
    python scripts/audit_data.py
    python scripts/audit_data.py --raw     # also audit raw source files
    python scripts/audit_data.py --output  path/to/report.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_CSV    = os.path.join(ROOT, "src", "data", "training_data.csv")
SNAP_CSV        = os.path.join(ROOT, "src", "data", "SNAPApps", "SNAPData.csv")
COUNTY_METRO    = os.path.join(ROOT, "src", "data", "county_to_metro.csv")
POP_CSV         = os.path.join(ROOT, "src", "data", "popData.csv")
INCOME_CSV      = os.path.join(ROOT, "src", "data", "MedianIncome.csv")
DEFAULT_OUTPUT  = os.path.join(ROOT, "outputs", "audit", "audit_report.csv")

# ── Issue accumulator ─────────────────────────────────────────────────────────

issues = []   # list of dicts — each dict is one row in the CSV report

def flag(category, check, entity, detail, severity="WARNING", value=None):
    """Record one audit finding."""
    issues.append({
        "severity": severity,
        "category": category,
        "check":    check,
        "entity":   str(entity),
        "detail":   detail,
        "value":    value,
    })

# ── Section printers ──────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

def ok(msg):
    print(f"  ✓  {msg}")

def warn(msg):
    print(f"  ⚠  {msg}")

def err(msg):
    print(f"  ✗  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Missing values
# ══════════════════════════════════════════════════════════════════════════════

def check_missing_values(df: pd.DataFrame):
    section("CHECK 1 — MISSING VALUES  (training_data.csv)")

    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}  |  Total cells: {total_cells:,}")
    print(f"  Total missing cells: {total_missing:,}  ({100*total_missing/total_cells:.2f}%)\n")

    any_issue = False
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        pct = 100 * n_missing / len(df)
        sev = "ERROR" if pct > 20 else "WARNING"
        msg = f"{col}: {n_missing} missing ({pct:.1f}%)"
        if pct > 20:
            err(msg)
        else:
            warn(msg)
        flag("missing_values", "null_check", col, msg, severity=sev, value=n_missing)
        any_issue = True

    if not any_issue:
        ok("No missing values in any column")

    # Per-column missing breakdown by county for trend columns
    trend_cols = [c for c in df.columns if c.startswith("monthly_average_")]
    if trend_cols:
        print(f"\n  Trend column NaN detail (counties with missing trend data):")
        for col in trend_cols:
            county_missing = df.groupby("county")[col].apply(lambda x: x.isna().sum())
            county_missing = county_missing[county_missing > 0]
            if county_missing.empty:
                ok(f"{col}: complete for all counties")
            else:
                warn(f"{col}: {len(county_missing)} counties have NaN rows")
                for county, n in county_missing.items():
                    flag("missing_values", "trend_nan_by_county", county,
                         f"{col}: {n} NaN row(s)", value=n)
                # Show the worst offenders
                top = county_missing.nlargest(5)
                for c, n in top.items():
                    print(f"     {c}: {n} NaN")
                if len(county_missing) > 5:
                    print(f"     ... and {len(county_missing)-5} more")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: Duplicate rows
# ══════════════════════════════════════════════════════════════════════════════

def check_duplicates(df: pd.DataFrame):
    section("CHECK 2 — DUPLICATE ROWS")

    # Exact duplicates
    n_exact = df.duplicated().sum()
    if n_exact:
        err(f"{n_exact} exact duplicate rows")
        flag("duplicates", "exact_duplicate", "training_data", f"{n_exact} duplicate rows",
             severity="ERROR", value=n_exact)
    else:
        ok("No exact duplicate rows")

    # Logical duplicates: same county + date should be unique
    key_cols = ["county", "date"]
    n_key = df.duplicated(subset=key_cols).sum()
    if n_key:
        duped = df[df.duplicated(subset=key_cols, keep=False)][key_cols + ["SNAP_Applications"]]
        err(f"{n_key} rows share the same (county, date) key:")
        print(duped.to_string(index=False))
        flag("duplicates", "key_duplicate", "county+date",
             f"{n_key} rows with duplicate (county, date)", severity="ERROR", value=n_key)
    else:
        ok("All (county, date) combinations are unique")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Date coverage by county
# ══════════════════════════════════════════════════════════════════════════════

def check_date_coverage(df: pd.DataFrame):
    section("CHECK 3 — DATE COVERAGE BY COUNTY")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    global_min = df["date"].min()
    global_max = df["date"].max()
    all_months = pd.date_range(global_min, global_max, freq="MS")
    expected_n = len(all_months)

    print(f"  Global date range: {global_min.date()} → {global_max.date()}")
    print(f"  Expected months per county: {expected_n}")

    coverage = df.groupby("county")["date"].agg(
        first="min", last="max", n_months="count"
    ).reset_index()
    coverage["expected"] = expected_n
    coverage["missing_months"] = coverage["expected"] - coverage["n_months"]
    coverage["coverage_pct"] = 100 * coverage["n_months"] / coverage["expected"]

    # Check for gaps (non-consecutive months)
    gap_counties = []
    for county, grp in df.groupby("county"):
        dates = grp["date"].sort_values().reset_index(drop=True)
        diffs = dates.diff().dropna()
        non_monthly = diffs[diffs != pd.Timedelta("31 days")].dropna()
        # Allow 28–31 day differences (month lengths vary)
        gaps = diffs[~diffs.between(pd.Timedelta("28 days"), pd.Timedelta("32 days"))]
        if not gaps.empty:
            gap_counties.append((county, len(gaps), str(gaps.values)))

    short = coverage[coverage["missing_months"] > 0]
    if short.empty:
        ok(f"All {len(coverage)} counties have {expected_n} months")
    else:
        warn(f"{len(short)} counties have fewer than {expected_n} months:")
        for _, row in short.iterrows():
            print(f"     {row['county']}: {int(row['n_months'])} months "
                  f"(missing {int(row['missing_months'])}, "
                  f"{row['first'].date()} → {row['last'].date()})")
            flag("date_coverage", "incomplete_months", row["county"],
                 f"{int(row['n_months'])}/{expected_n} months present",
                 value=int(row["missing_months"]))

    if gap_counties:
        warn(f"{len(gap_counties)} counties have non-consecutive month gaps:")
        for county, n, vals in gap_counties:
            print(f"     {county}: {n} gap(s)")
            flag("date_coverage", "month_gap", county, f"{n} non-consecutive month gap(s)")
    else:
        ok("No month gaps in any county (all months consecutive)")

    return coverage


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Incomplete county records
# ══════════════════════════════════════════════════════════════════════════════

def check_incomplete_counties(df: pd.DataFrame):
    section("CHECK 4 — INCOMPLETE COUNTY RECORDS")

    key_fields = ["Population", "Median_Income", "SNAP_Application_Rate"]
    trend_fields = [c for c in df.columns if c.startswith("monthly_average_")]
    all_fields = key_fields + trend_fields

    any_issue = False
    for col in all_fields:
        if col not in df.columns:
            warn(f"Column '{col}' not found in training data")
            continue
        county_nulls = df.groupby("county")[col].apply(lambda x: x.isna().sum())
        bad = county_nulls[county_nulls > 0]
        if bad.empty:
            ok(f"{col}: complete for all counties")
        else:
            any_issue = True
            pct_bad = 100 * len(bad) / df["county"].nunique()
            warn(f"{col}: {len(bad)} counties have ≥1 NaN  ({pct_bad:.0f}% of counties)")
            for county, n in bad.items():
                rows_pct = 100 * n / len(df[df["county"] == county])
                flag("incomplete_records", f"null_{col}", county,
                     f"{n} NaN in {col} ({rows_pct:.0f}% of county rows)", value=n)
            # Show per-county breakdown for trend columns
            if col in trend_fields and len(bad) <= 10:
                for c, n in bad.items():
                    print(f"     {c}: {n} NaN")

    if not any_issue:
        ok("All counties have complete Population, Income, Rate, and Trend records")

    # Counties missing entirely from a required lookup table
    print()
    pop   = pd.read_csv(POP_CSV);   pop.columns = pop.columns.str.strip()
    cm    = pd.read_csv(COUNTY_METRO); cm.columns = cm.columns.str.strip()
    inc   = pd.read_csv(INCOME_CSV)
    inc["county_key"] = inc["County"].str.replace(" ", "")
    snap_counties = set(df["county"].unique())
    pop_counties  = set(pop["County"].unique())
    cm_counties   = set(cm["county"].unique())
    inc_counties  = set(inc["county_key"].unique())
    training_keys = {c.replace(" ", "") for c in snap_counties}

    in_snap_not_pop = snap_counties - pop_counties
    in_snap_not_cm  = snap_counties - cm_counties
    in_snap_not_inc = training_keys - inc_counties

    if in_snap_not_pop:
        err(f"Counties in training data but MISSING from popData.csv: {in_snap_not_pop}")
        for c in in_snap_not_pop:
            flag("incomplete_records", "missing_from_pop", c, "Not in popData.csv", "ERROR")
    else:
        ok("All training counties present in popData.csv")

    if in_snap_not_cm:
        err(f"Counties in training data but MISSING from county_to_metro.csv: {in_snap_not_cm}")
        for c in in_snap_not_cm:
            flag("incomplete_records", "missing_from_county_metro", c,
                 "Not in county_to_metro.csv", "ERROR")
    else:
        ok("All training counties present in county_to_metro.csv")

    if in_snap_not_inc:
        warn(f"Counties (key) in training data but MISSING from MedianIncome.csv: {in_snap_not_inc}")
        for c in in_snap_not_inc:
            flag("incomplete_records", "missing_from_income", c,
                 "Not in MedianIncome.csv", "WARNING")
    else:
        ok("All training counties present in MedianIncome.csv")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5: DMA-to-county mapping issues
# ══════════════════════════════════════════════════════════════════════════════

def check_dma_mapping(df: pd.DataFrame):
    section("CHECK 5 — DMA-TO-COUNTY MAPPING")

    cm = pd.read_csv(COUNTY_METRO)
    cm.columns = cm.columns.str.strip()

    all_counties_58 = set(
        pd.read_csv(SNAP_CSV, header=None, names=["county","month_year","val"])["county"].unique()
    )

    # Counties in SNAP data not in county_metro map
    unmapped = all_counties_58 - set(cm["county"].unique())
    if unmapped:
        err(f"{len(unmapped)} SNAP counties have no DMA mapping:")
        for c in sorted(unmapped):
            print(f"     {c}")
            flag("dma_mapping", "unmapped_county", c,
                 "County in SNAP data but not in county_to_metro.csv", "ERROR")
    else:
        ok("All 58 SNAP counties mapped to a DMA")

    # Counties in county_metro not in SNAP data
    extra_in_map = set(cm["county"].unique()) - all_counties_58
    if extra_in_map:
        warn(f"{len(extra_in_map)} counties in county_metro map have no SNAP data:")
        for c in sorted(extra_in_map):
            print(f"     {c}")
            flag("dma_mapping", "mapped_but_no_snap", c,
                 "In county_to_metro.csv but no SNAP data", "WARNING")
    else:
        ok("No extra counties in county_to_metro.csv")

    # DMA consistency: training_data metro vs county_metro map
    merged = df[["county","metro_area"]].drop_duplicates().merge(
        cm, on="county", how="left", suffixes=("_training","_map")
    )
    conflicts = merged[merged["metro_area_training"] != merged["metro_area_map"]]
    if not conflicts.empty:
        err(f"{len(conflicts)} counties have conflicting DMA assignments:")
        for _, row in conflicts.iterrows():
            detail = f"training='{row['metro_area_training']}' vs map='{row['metro_area_map']}'"
            print(f"     {row['county']}: {detail}")
            flag("dma_mapping", "dma_conflict", row["county"], detail, "ERROR")
    else:
        ok("DMA assignments consistent between training data and county_to_metro.csv")

    # DMA coverage: which DMAs have trend data
    trends_dir = os.path.join(ROOT, "src", "data", "trends", "CalFresh")
    if os.path.exists(trends_dir):
        trend_dmas = {f.replace(".csv","") for f in os.listdir(trends_dir) if f.endswith(".csv")}
        map_dmas   = set(cm["metro_area"].unique())
        missing_trend_dmas = map_dmas - trend_dmas
        if missing_trend_dmas:
            warn(f"{len(missing_trend_dmas)} DMAs in county map have no CalFresh trend file:")
            for d in sorted(missing_trend_dmas):
                print(f"     {d}")
                flag("dma_mapping", "no_trend_file", d,
                     "DMA in county map but no CalFresh trend CSV", "WARNING")
        else:
            ok(f"All {len(map_dmas)} DMAs have a CalFresh trend file")

    # Summary table
    dma_summary = cm.groupby("metro_area")["county"].count().reset_index()
    dma_summary.columns = ["DMA", "n_counties"]
    print(f"\n  DMA breakdown ({len(dma_summary)} DMAs):")
    for _, row in dma_summary.sort_values("DMA").iterrows():
        print(f"     {row['DMA']}: {int(row['n_counties'])} counties")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6: Outliers
# ══════════════════════════════════════════════════════════════════════════════

def check_outliers(df: pd.DataFrame):
    section("CHECK 6 — OUTLIERS")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── 6a: Application count outliers (per county, IQR method) ──────────────
    print("  [6a] SNAP Application Count Outliers  (per county, IQR × 3)")
    count_outliers = []
    for county, grp in df.groupby("county"):
        vals = grp["SNAP_Applications"].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        out = grp[(grp["SNAP_Applications"] < lo) | (grp["SNAP_Applications"] > hi)]
        for _, row in out.iterrows():
            count_outliers.append({
                "county": county,
                "date":   row["date"].date(),
                "value":  row["SNAP_Applications"],
                "median": round(vals.median(), 1),
                "ratio":  round(row["SNAP_Applications"] / vals.median(), 2),
            })
            flag("outliers", "applications_iqr", county,
                 f"{row['date'].date()}: {row['SNAP_Applications']:.0f} "
                 f"(ratio to median: {row['SNAP_Applications']/vals.median():.1f}×)",
                 severity="WARNING", value=round(row["SNAP_Applications"], 1))

    if count_outliers:
        out_df = pd.DataFrame(count_outliers).sort_values("ratio", ascending=False)
        warn(f"{len(count_outliers)} application-count outlier(s) detected:")
        print(out_df.to_string(index=False))
    else:
        ok("No application-count outliers (IQR×3 per county)")

    # ── 6b: Per-capita rate outliers (global, z-score) ────────────────────────
    print("\n  [6b] Per-Capita SNAP Rate Outliers  (global z-score, |z| > 4)")
    rates = df["SNAP_Application_Rate"].dropna()
    z_threshold = 4.0
    mean_r, std_r = rates.mean(), rates.std()
    df["_z"] = (df["SNAP_Application_Rate"] - mean_r) / std_r
    rate_outliers = df[df["_z"].abs() > z_threshold][
        ["county", "date", "SNAP_Application_Rate", "_z", "Population"]
    ].copy()
    rate_outliers = rate_outliers.sort_values("_z", ascending=False)

    if not rate_outliers.empty:
        warn(f"{len(rate_outliers)} rate outlier(s) with |z-score| > {z_threshold}:")
        print(rate_outliers.rename(columns={"_z": "z_score"}).to_string(index=False))
        for _, row in rate_outliers.iterrows():
            flag("outliers", "rate_zscore", row["county"],
                 f"{pd.Timestamp(row['date']).date()}: rate={row['SNAP_Application_Rate']:.5f} "
                 f"(z={row['_z']:.2f})",
                 severity="WARNING", value=round(row["SNAP_Application_Rate"], 6))
    else:
        ok(f"No per-capita rate outliers (|z| > {z_threshold})")

    # ── 6c: Zero / negative application counts ────────────────────────────────
    print("\n  [6c] Zero / Negative Application Counts")
    zero_neg = df[df["SNAP_Applications"] <= 0]
    if not zero_neg.empty:
        err(f"{len(zero_neg)} rows with zero or negative SNAP_Applications:")
        print(zero_neg[["county","date","SNAP_Applications"]].to_string(index=False))
        for _, row in zero_neg.iterrows():
            flag("outliers", "zero_or_negative", row["county"],
                 f"{pd.Timestamp(row['date']).date()}: {row['SNAP_Applications']}",
                 severity="ERROR", value=row["SNAP_Applications"])
    else:
        ok("No zero or negative application counts")

    # ── 6d: Population sanity ─────────────────────────────────────────────────
    print("\n  [6d] Population Sanity (per-county consistency)")
    pop_counts = df.groupby("county")["Population"].nunique()
    multi_pop = pop_counts[pop_counts > 1]
    if not multi_pop.empty:
        warn(f"{len(multi_pop)} counties have multiple Population values (should be 1):")
        for c, n in multi_pop.items():
            vals = df[df["county"]==c]["Population"].unique()
            print(f"     {c}: {n} distinct values {vals}")
            flag("outliers", "population_inconsistency", c,
                 f"{n} distinct Population values: {vals}", "WARNING")
    else:
        ok("All counties have a single consistent Population value")

    df.drop(columns=["_z"], inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: Raw source file checks
# ══════════════════════════════════════════════════════════════════════════════

def check_raw_sources():
    section("RAW SOURCE FILES — SUPPLEMENTARY CHECKS")

    # SNAP raw: suppressed values
    snap = pd.read_csv(SNAP_CSV, header=None, names=["county","month_year","SNAP_Applications"])
    n_suppressed = (snap["SNAP_Applications"] == "*").sum()
    pct = 100 * n_suppressed / len(snap)
    if n_suppressed:
        warn(f"SNAPData.csv: {n_suppressed} suppressed '*' values ({pct:.1f}%) — interpolated in pipeline")
        # Which counties are most affected?
        sup_by_county = snap[snap["SNAP_Applications"]=="*"].groupby("county").size()
        for c, n in sup_by_county.sort_values(ascending=False).items():
            flag("raw_sources", "suppressed_snap", c,
                 f"{n} suppressed '*' value(s) in SNAPData.csv", value=n)
        print("  Most suppressed counties:")
        print(sup_by_county.sort_values(ascending=False).to_string())
    else:
        ok("SNAPData.csv: no suppressed '*' values")

    # Population: check for zeros or missing
    pop = pd.read_csv(POP_CSV)
    pop.columns = pop.columns.str.strip()
    zero_pop = pop[pop["Population"] == 0]
    if not zero_pop.empty:
        err(f"popData.csv: {len(zero_pop)} county(ies) with Population = 0")
        flag("raw_sources", "zero_population", str(zero_pop["County"].tolist()),
             "Population = 0", "ERROR")
    else:
        ok(f"popData.csv: all {len(pop)} counties have non-zero population")

    # Income: check for parse errors
    inc = pd.read_csv(INCOME_CSV)
    inc["Median_Income"] = pd.to_numeric(
        inc["Median Income"].str.replace(",",""), errors="coerce"
    )
    bad_inc = inc[inc["Median_Income"].isna()]
    if not bad_inc.empty:
        warn(f"MedianIncome.csv: {len(bad_inc)} unparseable income values")
        flag("raw_sources", "unparseable_income",
             str(bad_inc["County"].tolist()), "Could not parse Median Income", "WARNING")
    else:
        ok(f"MedianIncome.csv: all {len(inc)} counties have parseable income values")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY + EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_final_summary():
    section("AUDIT SUMMARY")

    df_issues = pd.DataFrame(issues)
    if df_issues.empty:
        ok("No issues found — data looks clean!")
        return df_issues

    counts = df_issues.groupby(["severity","category"]).size().reset_index(name="count")
    errors   = df_issues[df_issues["severity"]=="ERROR"]
    warnings = df_issues[df_issues["severity"]=="WARNING"]

    print(f"  Total issues: {len(df_issues)}  "
          f"({len(errors)} errors, {len(warnings)} warnings)\n")

    if not errors.empty:
        print("  ERRORS (must fix before modelling):")
        for _, row in errors.iterrows():
            print(f"    ✗  [{row['category']}] {row['entity']}: {row['detail']}")

    if not warnings.empty:
        print("\n  WARNINGS (review recommended):")
        for _, row in warnings.iterrows():
            print(f"    ⚠  [{row['category']}] {row['entity']}: {row['detail']}")

    return df_issues


def export_report(df_issues: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_issues.to_csv(output_path, index=False)
    print(f"\n  Report saved → {output_path}")
    print(f"  ({len(df_issues)} issue(s) logged)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SNAP data audit script")
    parser.add_argument("--raw",    action="store_true",
                        help="Also audit raw source files (SNAPData, popData, MedianIncome)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Path for CSV report (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    os.chdir(ROOT)

    print("\n" + "═" * 60)
    print("  SNAP DATA AUDIT")
    print("═" * 60)

    if not os.path.exists(TRAINING_CSV):
        print(f"\n  ERROR: training_data.csv not found at {TRAINING_CSV}")
        print("  Run stage 2 first: python run_pipeline.py --stages 2")
        sys.exit(1)

    df = pd.read_csv(TRAINING_CSV, parse_dates=["date"])
    print(f"\n  Source: {TRAINING_CSV}")
    print(f"  Shape:  {df.shape}  ({df['county'].nunique()} counties, "
          f"{df['date'].nunique()} months)")

    check_missing_values(df)
    check_duplicates(df)
    check_date_coverage(df)
    check_incomplete_counties(df)
    check_dma_mapping(df)
    check_outliers(df)

    if args.raw:
        check_raw_sources()

    df_issues = print_final_summary()
    export_report(df_issues, args.output)


if __name__ == "__main__":
    main()
