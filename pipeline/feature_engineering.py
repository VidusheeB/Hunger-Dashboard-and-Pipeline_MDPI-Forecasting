"""
feature_engineering.py — Build the full modelling feature set from training_data.csv.

All features are computed within each county group, sorted by date, so no future
information leaks into lagged or rolling values. The output DataFrame is a drop-in
replacement for training_data.csv in stages 3 and 4.

Feature groups created:
  A. Lag features    — rate_lag{1,2,3}, calfresh_lag{1,2}, foodbank_lag{1,2}
  B. Rolling windows — rate_roll3_{mean,std}, calfresh_roll3, foodbank_roll3
  C. Momentum        — calfresh_momentum, foodbank_momentum (current − lag1)
  D. Seasonality     — month_sin, month_cos, quarter (cyclical month encoding)
  E. Population      — log_population (log10 spans 4 orders of magnitude)
  F. Income          — log_income, income_quintile (1–5 label per California quintile)
  G. Base features   — Population, Median_Income, monthly_average_*, month (unchanged)

Note: year-over-year (lag-12) was tested and dropped — marginal R² gain (+0.003)
at the cost of halving the dataset (1,160 → 638 rows). Not worth the trade-off.

Output:
  outputs/data/features.csv           — full engineered dataframe
  outputs/data/feature_registry.csv   — name, group, description, NaN count for every feature
"""

import logging
import os

import numpy as np
import pandas as pd

from pipeline import config

logger = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
FEATURES_CSV         = os.path.join(config.OUTPUTS_ROOT, "data", "features.csv")
FEATURE_REGISTRY_CSV = os.path.join(config.OUTPUTS_ROOT, "data", "feature_registry.csv")

# ── Canonical engineered feature list for modelling ──────────────────────────
# Update config.py FEATURE_COLS to this list when using the engineered dataset.
ENGINEERED_FEATURE_COLS = [
    # Base
    "Population",
    "Median_Income",
    "monthly_average_CalFresh",
    "monthly_average_FoodBank",
    "month",
    # Lags — SNAP rate
    "rate_lag1",
    "rate_lag2",
    "rate_lag3",
    # Lags — Google Trends
    "calfresh_lag1",
    "calfresh_lag2",
    "foodbank_lag1",
    "foodbank_lag2",
    # Rolling windows
    "rate_roll3_mean",
    "rate_roll3_std",
    "calfresh_roll3",
    "foodbank_roll3",
    # Momentum (trend direction)
    "calfresh_momentum",
    "foodbank_momentum",
    # Seasonality
    "month_sin",
    "month_cos",
    "quarter",
    # Transformed static features
    "log_population",
    "log_income",
    "income_quintile",
]


# ══════════════════════════════════════════════════════════════════════════════
# A. Lag features
# ══════════════════════════════════════════════════════════════════════════════

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged versions of SNAP application rate and Google Trends signals.

    Why lags matter:
    - SNAP application rate is strongly autocorrelated month-to-month. Knowing
      last month's rate is the single most predictive feature for this month.
    - Trends 1–2 months ago capture 'search interest that didn't yet result in
      an application' — a leading indicator effect.

    All lags are computed within each county group to prevent values from one
    county appearing as a lag for another. Rows where the lag window extends
    before the county's first observation are left as NaN (handled downstream
    by dropping NaN rows before modelling).

    Features added:
      rate_lag1    — SNAP_Application_Rate shifted 1 month back
      rate_lag2    — SNAP_Application_Rate shifted 2 months back
      rate_lag3    — SNAP_Application_Rate shifted 3 months back
      calfresh_lag1 — monthly_average_CalFresh shifted 1 month back
      calfresh_lag2 — monthly_average_CalFresh shifted 2 months back
      foodbank_lag1 — monthly_average_FoodBank shifted 1 month back
      foodbank_lag2 — monthly_average_FoodBank shifted 2 months back
    """
    df = df.sort_values(["county", "date"]).copy()

    for lag in [1, 2, 3]:
        df[f"rate_lag{lag}"] = (
            df.groupby("county")[config.TARGET_COL]
            .shift(lag)
        )

    for lag in [1, 2]:
        df[f"calfresh_lag{lag}"] = (
            df.groupby("county")["monthly_average_CalFresh"]
            .shift(lag)
        )
        df[f"foodbank_lag{lag}"] = (
            df.groupby("county")["monthly_average_FoodBank"]
            .shift(lag)
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# B. Rolling window features
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Add rolling mean and std over the past `window` months (default: 3).

    Rolling windows smooth out month-to-month noise and capture the local
    trend level more robustly than a single lag value. A 3-month window is
    a reasonable choice given only 35 months of data per county — larger
    windows would lose too many rows.

    min_periods=2 means the rolling stat is computed even when only 2 prior
    months are available, trading some precision for fewer dropped rows.

    Features added:
      rate_roll3_mean   — 3-month rolling mean of SNAP_Application_Rate
      rate_roll3_std    — 3-month rolling std  (captures volatility / instability)
      calfresh_roll3    — 3-month rolling mean of CalFresh search index
      foodbank_roll3    — 3-month rolling mean of FoodBank search index
    """
    df = df.sort_values(["county", "date"]).copy()

    def rolling_mean(series, w):
        return series.rolling(window=w, min_periods=2).mean()

    def rolling_std(series, w):
        return series.rolling(window=w, min_periods=2).std()

    df[f"rate_roll{window}_mean"] = (
        df.groupby("county")[config.TARGET_COL]
        .transform(lambda s: rolling_mean(s, window))
    )
    df[f"rate_roll{window}_std"] = (
        df.groupby("county")[config.TARGET_COL]
        .transform(lambda s: rolling_std(s, window))
    )
    df[f"calfresh_roll{window}"] = (
        df.groupby("county")["monthly_average_CalFresh"]
        .transform(lambda s: rolling_mean(s, window))
    )
    df[f"foodbank_roll{window}"] = (
        df.groupby("county")["monthly_average_FoodBank"]
        .transform(lambda s: rolling_mean(s, window))
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# C. Momentum features
# ══════════════════════════════════════════════════════════════════════════════

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add month-over-month momentum for Google Trends signals.

    Momentum = current value − lag1 value. A positive value means search
    interest is rising; negative means it is falling. This is the first
    derivative of the trend signal — the direction matters as much as
    the level for predicting next month's SNAP demand.

    Features added:
      calfresh_momentum — monthly_average_CalFresh minus calfresh_lag1
      foodbank_momentum — monthly_average_FoodBank minus foodbank_lag1

    Requires add_lag_features() to have been called first.
    """
    if "calfresh_lag1" not in df.columns:
        raise ValueError("add_lag_features() must be called before add_momentum_features()")

    df["calfresh_momentum"] = df["monthly_average_CalFresh"] - df["calfresh_lag1"]
    df["foodbank_momentum"]  = df["monthly_average_FoodBank"]  - df["foodbank_lag1"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# D. Seasonality features
# ══════════════════════════════════════════════════════════════════════════════

def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical month encoding and quarter.

    Raw month (1–12) is an ordinal variable that misleads tree-based models:
    December (12) appears 'far' from January (1) even though they are adjacent.
    Sine/cosine encoding wraps the calendar into a circle so the model sees
    the correct adjacency between months.

    month_sin = sin(2π × month / 12)
    month_cos = cos(2π × month / 12)

    Together, (month_sin, month_cos) uniquely identify each month and preserve
    the cyclical structure. The original 'month' column is kept as a third
    representation since tree models can still use it as a split point.

    Features added:
      month_sin — sine of month position on the annual cycle
      month_cos — cosine of month position on the annual cycle
      quarter   — calendar quarter (1–4), a coarser seasonal signal
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter"]   = ((df["month"] - 1) // 3) + 1
    return df


# ══════════════════════════════════════════════════════════════════════════════
# E. Population transformations
# ══════════════════════════════════════════════════════════════════════════════

def add_population_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform population.

    California county populations span four orders of magnitude: Alpine (1,043)
    to Los Angeles (9,550,505). On the raw scale, large counties completely
    dominate any population-based splits. Log10 compression makes the range
    comparable to other features (3.0 – 7.0) and is more interpretable:
    each unit increase represents a 10× larger county.

    Features added:
      log_population — log10(Population)
    """
    df = df.copy()
    df["log_population"] = np.log10(df["Population"].clip(lower=1))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# F. Income transformations
# ══════════════════════════════════════════════════════════════════════════════

def add_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform median income and add income quintile label.

    Income (53K–160K across California counties) is right-skewed and has a
    non-linear relationship with SNAP demand: the difference between a
    $55K and $65K county matters more than between $145K and $155K.
    Log10 captures this diminishing sensitivity.

    Income quintile (1–5) is a categorical encoding that lets the model learn
    step-change effects at wealth thresholds without assuming a smooth
    log-linear relationship throughout the full range.

    Quintile boundaries are computed from the counties in this dataset
    (not national benchmarks), so they represent California-relative wealth.

    Features added:
      log_income      — log10(Median_Income)
      income_quintile — integer 1 (lowest) to 5 (highest) within CA counties
    """
    df = df.copy()
    df["log_income"] = np.log10(df["Median_Income"].clip(lower=1))

    # Quintile boundaries from county-level values (one value per county)
    county_incomes = df.drop_duplicates("county")[["county", "Median_Income"]].set_index("county")
    quintile_labels = pd.qcut(
        county_incomes["Median_Income"],
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    )
    quintile_map = quintile_labels.astype(int).to_dict()
    df["income_quintile"] = df["county"].map(quintile_map)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Feature registry
# ══════════════════════════════════════════════════════════════════════════════

# Every engineered feature documented in one place.
# 'nan_policy' describes what NaN in this column means for modelling.
FEATURE_REGISTRY = [
    # ── Base features ─────────────────────────────────────────────────────────
    dict(name="Population",                group="base",       nan_policy="drop",
         description="County population (US Census estimate)"),
    dict(name="Median_Income",             group="base",       nan_policy="drop",
         description="County median household income (ACS 5-year estimate, $)"),
    dict(name="monthly_average_CalFresh",  group="base",       nan_policy="drop",
         description="Monthly average Google Trends index for 'CalFresh' (0–100)"),
    dict(name="monthly_average_FoodBank",  group="base",       nan_policy="drop",
         description="Monthly average Google Trends index for 'FoodBank' (0–100)"),
    dict(name="month",                     group="base",       nan_policy="never",
         description="Calendar month (1–12)"),
    # ── Lag features ──────────────────────────────────────────────────────────
    dict(name="rate_lag1",    group="lag_rate",  nan_policy="drop",
         description="SNAP_Application_Rate lagged 1 month (same county)"),
    dict(name="rate_lag2",    group="lag_rate",  nan_policy="drop",
         description="SNAP_Application_Rate lagged 2 months"),
    dict(name="rate_lag3",    group="lag_rate",  nan_policy="drop",
         description="SNAP_Application_Rate lagged 3 months"),
    dict(name="calfresh_lag1", group="lag_trends", nan_policy="drop",
         description="monthly_average_CalFresh lagged 1 month"),
    dict(name="calfresh_lag2", group="lag_trends", nan_policy="drop",
         description="monthly_average_CalFresh lagged 2 months"),
    dict(name="foodbank_lag1", group="lag_trends", nan_policy="drop",
         description="monthly_average_FoodBank lagged 1 month"),
    dict(name="foodbank_lag2", group="lag_trends", nan_policy="drop",
         description="monthly_average_FoodBank lagged 2 months"),
    # ── Rolling window features ───────────────────────────────────────────────
    dict(name="rate_roll3_mean", group="rolling", nan_policy="drop",
         description="3-month rolling mean of SNAP_Application_Rate (smoothed level)"),
    dict(name="rate_roll3_std",  group="rolling", nan_policy="drop",
         description="3-month rolling std of SNAP_Application_Rate (volatility signal)"),
    dict(name="calfresh_roll3",  group="rolling", nan_policy="drop",
         description="3-month rolling mean of CalFresh trends index"),
    dict(name="foodbank_roll3",  group="rolling", nan_policy="drop",
         description="3-month rolling mean of FoodBank trends index"),
    # ── Momentum features ─────────────────────────────────────────────────────
    dict(name="calfresh_momentum", group="momentum", nan_policy="drop",
         description="CalFresh month-over-month change (current − lag1); positive = rising interest"),
    dict(name="foodbank_momentum", group="momentum", nan_policy="drop",
         description="FoodBank month-over-month change (current − lag1)"),
    # ── Seasonality ───────────────────────────────────────────────────────────
    dict(name="month_sin", group="seasonality", nan_policy="never",
         description="sin(2π×month/12) — cyclical month encoding, adjacent to Dec/Jan"),
    dict(name="month_cos", group="seasonality", nan_policy="never",
         description="cos(2π×month/12) — cyclical month encoding, paired with month_sin"),
    dict(name="quarter",   group="seasonality", nan_policy="never",
         description="Calendar quarter (1–4); coarser seasonal signal than month"),
    # ── Population transforms ─────────────────────────────────────────────────
    dict(name="log_population", group="population", nan_policy="never",
         description="log10(Population); compresses 4-order-of-magnitude range to 3.0–7.0"),
    # ── Income transforms ─────────────────────────────────────────────────────
    dict(name="log_income",      group="income", nan_policy="never",
         description="log10(Median_Income); captures diminishing sensitivity at higher incomes"),
    dict(name="income_quintile", group="income", nan_policy="never",
         description="Income quintile within CA counties: 1=lowest 20%, 5=highest 20%"),
]


def build_feature_registry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the feature registry as a DataFrame, annotated with actual NaN counts
    from the current engineered dataset.
    """
    registry = pd.DataFrame(FEATURE_REGISTRY)
    registry["nan_count"] = registry["name"].apply(
        lambda col: int(df[col].isna().sum()) if col in df.columns else -1
    )
    registry["nan_pct"] = (registry["nan_count"] / len(df) * 100).round(2)
    registry["present"] = registry["name"].apply(lambda col: col in df.columns)
    return registry[["name", "group", "description", "nan_policy", "nan_count", "nan_pct", "present"]]


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(
    input_csv: str = config.TRAINING_DATA_CSV,
    output_csv: str = FEATURES_CSV,
    drop_nan_rows: bool = True,
) -> pd.DataFrame:
    """
    Load training_data.csv, apply all feature engineering steps, and save
    the result to outputs/data/features.csv.

    Args:
        input_csv:     Path to the base training data (output of stage 2).
        output_csv:    Where to write the enriched feature dataset.
        drop_nan_rows: If True, drop rows missing any modelling feature.
                       Set False to inspect the raw engineered frame with NaNs.

    Returns:
        Enriched DataFrame ready for stage 3 (train) and stage 4 (evaluate).
    """
    logger.info("=== FEATURE ENGINEERING ===")

    df = pd.read_csv(input_csv, parse_dates=["date"])
    logger.info(f"  Loaded: {df.shape}  ({df['county'].nunique()} counties, "
                f"{df['date'].nunique()} months)")

    # Apply all feature groups in dependency order
    df = add_lag_features(df)           # A — lags (must precede momentum)
    df = add_rolling_features(df)       # B — rolling windows
    df = add_momentum_features(df)      # C — momentum (needs lag1)
    df = add_seasonality_features(df)   # D — sin/cos/quarter
    df = add_population_features(df)    # E — log_population
    df = add_income_features(df)        # F — log_income, quintile

    logger.info(f"  Features engineered: {df.shape[1]} columns total")

    # Report NaN counts before dropping
    missing_by_feature = {
        col: int(df[col].isna().sum())
        for col in ENGINEERED_FEATURE_COLS
        if col in df.columns and df[col].isna().sum() > 0
    }
    if missing_by_feature:
        logger.info("  NaN counts by feature (before dropping):")
        for col, n in sorted(missing_by_feature.items(), key=lambda x: -x[1]):
            logger.info(f"    {col}: {n} ({100*n/len(df):.1f}%)")

    # Drop rows that are missing any modelling feature
    if drop_nan_rows:
        available = [c for c in ENGINEERED_FEATURE_COLS if c in df.columns]
        before = len(df)
        df = df.dropna(subset=available + [config.TARGET_COL]).reset_index(drop=True)
        logger.info(f"  Dropped {before - len(df)} rows with NaN in modelling features")
        logger.info(f"  Final shape: {df.shape}  ({df['county'].nunique()} counties)")

    # Save feature dataset
    df.to_csv(output_csv, index=False)
    logger.info(f"  Saved → {output_csv}")

    # Build and save feature registry
    registry = build_feature_registry(df)
    registry.to_csv(FEATURE_REGISTRY_CSV, index=False)
    logger.info(f"  Registry → {FEATURE_REGISTRY_CSV}")

    # Print summary table
    _print_summary(df, registry)

    return df


def _print_summary(df: pd.DataFrame, registry: pd.DataFrame):
    """Print a human-readable feature summary to stdout."""
    print(f"\n{'─'*65}")
    print(f"  ENGINEERED FEATURES  ({len(df):,} rows × {df.shape[1]} columns)")
    print(f"  {df['county'].nunique()} counties | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"{'─'*65}")

    for group in registry["group"].unique():
        grp_rows = registry[registry["group"] == group]
        print(f"\n  [{group.upper()}]")
        for _, row in grp_rows.iterrows():
            status = f"{int(row['nan_count'])} NaN" if row["nan_count"] > 0 else "complete"
            print(f"    {row['name']:<30}  {status:<12}  {row['description']}")

    print(f"\n{'─'*65}")
    complete = (registry["nan_count"] == 0).sum()
    print(f"  {complete}/{len(registry)} features complete (no NaN in final dataset)")
    print(f"{'─'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Feature dictionary
# ══════════════════════════════════════════════════════════════════════════════

# Full documentation for every modelling feature.
# Fields: name, group, definition, formula, source, dtype
FEATURE_DICTIONARY = [
    # ── Base ──────────────────────────────────────────────────────────────────
    dict(
        name="Population",
        group="Base",
        definition="County resident population used as denominator for per-capita rate and as a standalone size feature.",
        formula="Raw value from source; no transformation applied.",
        source="popData.csv (US Census / CA DOF estimate)",
        dtype="integer",
    ),
    dict(
        name="Median_Income",
        group="Base",
        definition="Median household income for the county. Higher income counties typically have lower SNAP demand.",
        formula="Raw value from source; no transformation applied.",
        source="MedianIncome.csv (ACS 5-year estimate, $)",
        dtype="float",
    ),
    dict(
        name="monthly_average_CalFresh",
        group="Base",
        definition="Mean Google Trends interest score for the search term 'CalFresh' within the county's DMA for the prediction month. "
                   "Higher values indicate more people actively searching for SNAP enrollment.",
        formula="mean(weekly Trends index) over all weeks in the calendar month, per DMA.",
        source="src/data/trends/CalFresh/{DMA}.csv (Google Trends, 0–100 scale)",
        dtype="float",
    ),
    dict(
        name="monthly_average_FoodBank",
        group="Base",
        definition="Mean Google Trends interest score for 'FoodBank' within the county's DMA. "
                   "Captures broader food-insecurity signal beyond formal SNAP enrollment.",
        formula="mean(weekly Trends index) over all weeks in the calendar month, per DMA.",
        source="src/data/trends/FoodBank/{DMA}.csv (Google Trends, 0–100 scale)",
        dtype="float",
    ),
    dict(
        name="month",
        group="Base",
        definition="Calendar month of the observation. Captures seasonal patterns in SNAP applications "
                   "(e.g. summer peaks in agricultural counties).",
        formula="date.month  [integer 1–12]",
        source="Derived from SNAP application date",
        dtype="integer",
    ),
    # ── Lag — SNAP rate ───────────────────────────────────────────────────────
    dict(
        name="rate_lag1",
        group="Lag (SNAP rate)",
        definition="SNAP application rate from 1 month prior for the same county. "
                   "The single most predictive feature: SNAP demand is strongly autocorrelated month-to-month.",
        formula="SNAP_Application_Rate(t-1)  [grouped by county, sorted by date]",
        source="Derived from SNAPData.csv + popData.csv",
        dtype="float",
    ),
    dict(
        name="rate_lag2",
        group="Lag (SNAP rate)",
        definition="SNAP application rate from 2 months prior. Extends autocorrelation signal and helps "
                   "detect medium-term trends.",
        formula="SNAP_Application_Rate(t-2)  [grouped by county, sorted by date]",
        source="Derived from SNAPData.csv + popData.csv",
        dtype="float",
    ),
    dict(
        name="rate_lag3",
        group="Lag (SNAP rate)",
        definition="SNAP application rate from 3 months prior. Captures quarterly cycle effects.",
        formula="SNAP_Application_Rate(t-3)  [grouped by county, sorted by date]",
        source="Derived from SNAPData.csv + popData.csv",
        dtype="float",
    ),
    # ── Lag — Google Trends ───────────────────────────────────────────────────
    dict(
        name="calfresh_lag1",
        group="Lag (Trends)",
        definition="CalFresh search interest from 1 month prior. Captures the leading-indicator effect: "
                   "people search before they apply, so last month's searches predict this month's applications.",
        formula="monthly_average_CalFresh(t-1)  [grouped by county, sorted by date]",
        source="Derived from src/data/trends/CalFresh/",
        dtype="float",
    ),
    dict(
        name="calfresh_lag2",
        group="Lag (Trends)",
        definition="CalFresh search interest from 2 months prior. Extends the leading-indicator window.",
        formula="monthly_average_CalFresh(t-2)  [grouped by county, sorted by date]",
        source="Derived from src/data/trends/CalFresh/",
        dtype="float",
    ),
    dict(
        name="foodbank_lag1",
        group="Lag (Trends)",
        definition="FoodBank search interest from 1 month prior.",
        formula="monthly_average_FoodBank(t-1)  [grouped by county, sorted by date]",
        source="Derived from src/data/trends/FoodBank/",
        dtype="float",
    ),
    dict(
        name="foodbank_lag2",
        group="Lag (Trends)",
        definition="FoodBank search interest from 2 months prior.",
        formula="monthly_average_FoodBank(t-2)  [grouped by county, sorted by date]",
        source="Derived from src/data/trends/FoodBank/",
        dtype="float",
    ),
    # ── Rolling windows ───────────────────────────────────────────────────────
    dict(
        name="rate_roll3_mean",
        group="Rolling",
        definition="3-month trailing mean of SNAP application rate. Smooths month-to-month noise and "
                   "represents the local level of food-assistance demand.",
        formula="mean(SNAP_Application_Rate(t-2), SNAP_Application_Rate(t-1), SNAP_Application_Rate(t))  "
                "[rolling window=3, min_periods=2, grouped by county]",
        source="Derived from SNAPData.csv + popData.csv",
        dtype="float",
    ),
    dict(
        name="rate_roll3_std",
        group="Rolling",
        definition="3-month trailing standard deviation of SNAP application rate. "
                   "High values indicate an unstable or rapidly changing county — a risk signal.",
        formula="std(SNAP_Application_Rate over trailing 3 months)  [rolling window=3, min_periods=2, grouped by county]",
        source="Derived from SNAPData.csv + popData.csv",
        dtype="float",
    ),
    dict(
        name="calfresh_roll3",
        group="Rolling",
        definition="3-month trailing mean of CalFresh search index. More stable estimate of sustained "
                   "search interest than the raw monthly value.",
        formula="mean(monthly_average_CalFresh over trailing 3 months)  [rolling window=3, min_periods=2, grouped by county]",
        source="Derived from src/data/trends/CalFresh/",
        dtype="float",
    ),
    dict(
        name="foodbank_roll3",
        group="Rolling",
        definition="3-month trailing mean of FoodBank search index.",
        formula="mean(monthly_average_FoodBank over trailing 3 months)  [rolling window=3, min_periods=2, grouped by county]",
        source="Derived from src/data/trends/FoodBank/",
        dtype="float",
    ),
    # ── Momentum ──────────────────────────────────────────────────────────────
    dict(
        name="calfresh_momentum",
        group="Momentum",
        definition="Month-over-month change in CalFresh search interest. Positive = rising interest "
                   "(leading signal of increased demand). Negative = falling interest.",
        formula="monthly_average_CalFresh(t) - monthly_average_CalFresh(t-1)",
        source="Derived from src/data/trends/CalFresh/",
        dtype="float",
    ),
    dict(
        name="foodbank_momentum",
        group="Momentum",
        definition="Month-over-month change in FoodBank search interest.",
        formula="monthly_average_FoodBank(t) - monthly_average_FoodBank(t-1)",
        source="Derived from src/data/trends/FoodBank/",
        dtype="float",
    ),
    # ── Seasonality ───────────────────────────────────────────────────────────
    dict(
        name="month_sin",
        group="Seasonality",
        definition="Sine component of cyclical month encoding. Paired with month_cos, these two features "
                   "represent month as a point on a unit circle so that December and January are adjacent "
                   "(distance ≈ 0) rather than far apart (distance = 11 on a raw 1–12 scale).",
        formula="sin(2π × month / 12)",
        source="Derived from SNAP application date",
        dtype="float",
    ),
    dict(
        name="month_cos",
        group="Seasonality",
        definition="Cosine component of cyclical month encoding. Together with month_sin uniquely identifies "
                   "each calendar month while preserving the circular annual structure.",
        formula="cos(2π × month / 12)",
        source="Derived from SNAP application date",
        dtype="float",
    ),
    dict(
        name="quarter",
        group="Seasonality",
        definition="Calendar quarter (1=Jan–Mar, 2=Apr–Jun, 3=Jul–Sep, 4=Oct–Dec). "
                   "Coarser seasonal grouping that captures broad agricultural and policy cycles.",
        formula="floor((month - 1) / 3) + 1",
        source="Derived from SNAP application date",
        dtype="integer",
    ),
    # ── Population transform ──────────────────────────────────────────────────
    dict(
        name="log_population",
        group="Population",
        definition="Base-10 logarithm of county population. California counties span four orders of "
                   "magnitude (Alpine: 1,043 → Los Angeles: 9,550,505). Log compression brings this into "
                   "a 3.0–7.0 range comparable to other features, preventing large-county dominance.",
        formula="log₁₀(Population)",
        source="popData.csv (US Census / CA DOF estimate)",
        dtype="float",
    ),
    # ── Income transforms ─────────────────────────────────────────────────────
    dict(
        name="log_income",
        group="Income",
        definition="Base-10 logarithm of median household income. The relationship between income and SNAP "
                   "demand is non-linear: a $10K difference matters more at $55K than at $150K. "
                   "Log scaling captures this diminishing sensitivity.",
        formula="log₁₀(Median_Income)",
        source="MedianIncome.csv (ACS 5-year estimate)",
        dtype="float",
    ),
    dict(
        name="income_quintile",
        group="Income",
        definition="Income quintile of the county relative to all California counties in this dataset "
                   "(1 = lowest 20%, 5 = highest 20%). Allows the model to learn step-change effects at "
                   "wealth thresholds without assuming a smooth relationship across the full income range.",
        formula="pd.qcut(Median_Income across CA counties, q=5, labels=[1,2,3,4,5])",
        source="MedianIncome.csv (ACS 5-year estimate)",
        dtype="integer",
    ),
]


def generate_feature_dictionary(
    features_csv: str = None,
    output_csv: str = None,
) -> pd.DataFrame:
    """
    Build the feature dictionary DataFrame, save it as CSV, and print a
    Markdown table suitable for inclusion in a paper or report.

    Args:
        features_csv: Path to the engineered features CSV (used to compute
                      value ranges from actual data). Defaults to config.FEATURES_CSV.
        output_csv:   Where to write the dictionary CSV.
                      Defaults to config.FEATURE_DICTIONARY_CSV.

    Returns:
        Feature dictionary as a DataFrame.
    """
    if features_csv is None:
        features_csv = config.FEATURES_CSV
    if output_csv is None:
        output_csv = config.FEATURE_DICTIONARY_CSV

    # Build base dictionary DataFrame
    df_dict = pd.DataFrame(FEATURE_DICTIONARY)

    # Annotate with observed value ranges from the actual feature dataset
    if os.path.exists(features_csv):
        df_feat = pd.read_csv(features_csv)
        ranges = {}
        for col in df_dict["name"]:
            if col in df_feat.columns:
                s = df_feat[col].dropna()
                ranges[col] = f"{s.min():.4g} – {s.max():.4g}"
            else:
                ranges[col] = "N/A"
        df_dict["observed_range"] = df_dict["name"].map(ranges)
    else:
        df_dict["observed_range"] = "N/A (features.csv not found)"

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_dict.to_csv(output_csv, index=False)
    logger.info(f"  Feature dictionary → {output_csv}")

    # Print Markdown table
    _print_markdown_table(df_dict)

    return df_dict


def _print_markdown_table(df_dict: pd.DataFrame) -> None:
    """Print a Markdown table of all features for copy-paste into a paper."""

    print("\n" + "=" * 80)
    print("  FEATURE DICTIONARY — MARKDOWN TABLE")
    print("=" * 80)
    print()

    # Group into sections for readability
    groups = df_dict["group"].unique()
    for group in groups:
        rows = df_dict[df_dict["group"] == group]
        print(f"### {group}\n")
        print("| Feature | Definition | Formula | Source | Range |")
        print("|---|---|---|---|---|")
        for _, row in rows.iterrows():
            # Escape pipe characters inside cells
            def esc(s):
                return str(s).replace("|", "\\|").replace("\n", " ")
            print(
                f"| `{row['name']}` "
                f"| {esc(row['definition'])} "
                f"| `{esc(row['formula'])}` "
                f"| {esc(row['source'])} "
                f"| {esc(row['observed_range'])} |"
            )
        print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_output_dirs()
    engineer_features()
    generate_feature_dictionary()
