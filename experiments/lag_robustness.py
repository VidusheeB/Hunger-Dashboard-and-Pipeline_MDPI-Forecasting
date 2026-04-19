"""
lag_robustness.py — Does Google Trends help more when SNAP reporting is more delayed?

Design
------
Official SNAP administrative data has a reporting lag of unknown but substantial
length. The exact lag varies across counties and reporting cycles. This experiment
tests robustness by varying the assumed lag from 1 to 12 months and asking:

  At lag L, does adding Google Trends (real-time) improve out-of-sample accuracy
  beyond what can be achieved with demographics + unemployment + SNAP data from L
  months ago?

Feature sets (for each lag L = 1 … 12)
---------------------------------------
  Base (no Trends):  demographics + log transforms + seasonality +
                     unemployment_rate (LAUS, ~1-month lag — always available) +
                     unemployment_rate_lag1 +
                     snap_rate_lag{L}  (most recent SNAP data available at lag L)

  With Trends:       Base + all Google Trends features (CalFresh, FoodBank,
                     FoodStamps, SNAPTopic) with lags, rolling stats, momentum

Note: unemployment is released monthly by BLS (~1-month lag) and is always
available regardless of SNAP lag. Even at SNAP lag = 12 months, you still have
last month's unemployment rate.

Walk-forward setup
------------------
Same as tune_deployable_model.py:
  - For each month t: train on all data before t, predict t
  - Minimum 12 training months before any prediction
  - Non-COVID rows only for regression metrics (COVID = 2020-01-01 to 2021-12-31)

Hyperparameters
---------------
Reuse the already-tuned XGBoost params from deployable_tuning_results.json.
This is intentionally conservative for the No-Trends model (params are tuned on
the With-Trends feature set), which means any Trends advantage in the comparison
is understated, not overstated.

Statistical testing
-------------------
Diebold-Mariano test (HLN corrected) at the main lag (6 months) only, to avoid
inflation from 24 simultaneous tests. Effect sizes (ΔR², ΔMAE, ΔsMAPE) are
reported at all lags. A monotonicity test across lags is reported as a single
nonparametric test (Spearman correlation of Trends gain vs lag).

Outputs
-------
  outputs/metrics/lag_robustness_results.json
  outputs/metrics/lag_robustness_table.csv
  Console: formatted table + statistical summary
"""

import itertools
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config
from experiments.tune_deployable_model import merge_laus, DEPLOYABLE_FEATURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET      = "SNAP_Application_Rate"
COVID_START = "2020-01-01"
COVID_END   = "2021-12-31"
MAIN_LAG    = 6   # lag for statistical test

OUT_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "lag_robustness_results.json")
OUT_CSV  = os.path.join(config.OUTPUTS_ROOT, "metrics", "lag_robustness_table.csv")
TUNE_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "deployable_tuning_results.json")

# Unemployment + demographics + seasonality (always available regardless of SNAP lag)
BASE_FEATURES = [
    "Population", "Median_Income",
    "unemployment_rate", "unemployment_rate_lag1",
    "month_sin", "month_cos", "quarter", "month",
    "log_population", "log_income",
]

# All Google Trends features (real-time, ~2-day lag)
TRENDS_FEATURES = [f for f in DEPLOYABLE_FEATURES if f not in BASE_FEATURES]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = (np.abs(actual) + np.abs(predicted)) > 0
    return float(100 * np.mean(
        np.abs(actual[mask] - predicted[mask]) /
        ((np.abs(actual[mask]) + np.abs(predicted[mask])) / 2)
    ))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae    = float(np.abs(y_true - y_pred).mean())
    smape  = _smape(y_true, y_pred)
    return dict(r2=round(r2, 4), mae=round(mae, 6), smape=round(smape, 2))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> dict:
    """DM test, Harvey-Leybourne-Newbold small-sample corrected. d = e1²-e2²."""
    d      = e1 ** 2 - e2 ** 2
    n      = len(d)
    d_bar  = d.mean()
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 2 else 0.0
    var_d  = (gamma0 + 2 * gamma1) / n
    if var_d <= 0:
        return dict(dm_stat=0.0, p_value=1.0, significant=False)
    dm     = d_bar / np.sqrt(var_d)
    corr   = np.sqrt((n + 1 - 2 + 1 * (1 - 1) / n) / n)
    dm_adj = dm * corr
    p      = float(2 * (1 - stats.t.cdf(abs(dm_adj), df=n - 1)))
    return dict(dm_stat=round(float(dm_adj), 4), p_value=round(p, 4),
                significant=bool(p < 0.05))


# ── Walk-forward ───────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, features: list, params: dict,
                 min_months: int) -> pd.DataFrame:
    """
    Standard walk-forward: train on date < t, predict date == t.
    Returns DataFrame with columns [county, date, actual, predicted].
    Drops rows where any feature or target is NaN.
    """
    dates  = sorted(df["date"].unique())
    rows   = []
    n_skip = 0

    for t in dates:
        train = df[df["date"] < t]
        test  = df[df["date"] == t]
        if train["date"].nunique() < min_months:
            n_skip += 1
            continue
        tr = train[features + [TARGET]].dropna()
        te = test[features + [TARGET]].dropna()
        if len(tr) < 10 or len(te) == 0:
            n_skip += 1
            continue
        model = XGBRegressor(**params, objective="reg:squarederror", verbosity=0)
        model.fit(tr[features].values, tr[TARGET].values)
        preds = np.clip(model.predict(te[features].values), 0, None)
        out   = test.loc[te.index, ["county", "date", TARGET]].copy()
        out["predicted"] = preds
        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=== LAG ROBUSTNESS: Trends benefit across 1–12 month SNAP delays ===")

    # Load features
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    df = merge_laus(df)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["county", "date"]).reset_index(drop=True)
    log.info(f"Loaded {len(df):,} rows | {df['county'].nunique()} counties")

    # Load tuned hyperparameters
    with open(TUNE_JSON) as f:
        params = json.load(f)["best_params"]
    params.pop("random_state", None)
    params.pop("n_jobs", None)
    params["random_state"] = 42
    params["n_jobs"]       = -1
    log.info(f"Using tuned params: {params}")

    # Validate base features
    missing = [f for f in BASE_FEATURES + TRENDS_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in features.csv: {missing}")

    min_months = config.WALK_FORWARD_MIN_MONTHS
    results    = []

    # ── Loop over lags ─────────────────────────────────────────────────────────
    for lag in range(1, 13):
        snap_col = f"snap_rate_lag{lag}"
        df[snap_col] = df.groupby("county")[TARGET].shift(lag)

        features_no    = BASE_FEATURES + [snap_col]
        features_with  = BASE_FEATURES + TRENDS_FEATURES + [snap_col]

        log.info(f"Lag {lag:2d}: walk-forward NO Trends ({len(features_no)} features) …")
        wf_no   = walk_forward(df, features_no,   params, min_months)

        log.info(f"Lag {lag:2d}: walk-forward WITH Trends ({len(features_with)} features) …")
        wf_with = walk_forward(df, features_with, params, min_months)

        # Align on common (county, date) pairs — non-COVID only for metrics
        merged = wf_no.merge(
            wf_with[["county", "date", "predicted"]].rename(
                columns={"predicted": "pred_with"}),
            on=["county", "date"],
        ).rename(columns={"predicted": "pred_no"}).dropna()

        noncovid = merged[~merged["date"].between(COVID_START, COVID_END)]
        y    = noncovid[TARGET].values
        p_no = noncovid["pred_no"].values
        p_wt = noncovid["pred_with"].values

        m_no = _metrics(y, p_no)
        m_wt = _metrics(y, p_wt)

        row = dict(
            lag=lag,
            n=len(noncovid),
            no_r2=m_no["r2"],    with_r2=m_wt["r2"],    delta_r2=round(m_wt["r2"]-m_no["r2"],4),
            no_mae=m_no["mae"],  with_mae=m_wt["mae"],  delta_mae=round(m_wt["mae"]-m_no["mae"],6),
            no_smape=m_no["smape"], with_smape=m_wt["smape"],
            delta_smape=round(m_wt["smape"]-m_no["smape"],2),
        )
        # Diebold-Mariano at main lag only
        if lag == MAIN_LAG:
            e_no = y - p_no
            e_wt = y - p_wt
            dm   = diebold_mariano(e_no, e_wt)
            wilcox = stats.wilcoxon(np.abs(e_no), np.abs(e_wt))
            row["dm_stat"]    = dm["dm_stat"]
            row["dm_p"]       = dm["p_value"]
            row["dm_sig"]     = dm["significant"]
            row["wilcox_p"]   = round(float(wilcox.pvalue), 4)
            row["wilcox_sig"] = bool(wilcox.pvalue < 0.05)
        else:
            row.update(dm_stat=None, dm_p=None, dm_sig=None,
                       wilcox_p=None, wilcox_sig=None)

        results.append(row)
        log.info(
            f"  Lag {lag:2d}: no-Trends R²={m_no['r2']:.4f}  "
            f"with-Trends R²={m_wt['r2']:.4f}  ΔR²={m_wt['r2']-m_no['r2']:+.4f}"
        )

        # Remove temp column before next iteration
        df.drop(columns=[snap_col], inplace=True)

    # ── Monotonicity test (Spearman: lag vs ΔR²) ──────────────────────────────
    lags       = [r["lag"]      for r in results]
    delta_r2s  = [r["delta_r2"] for r in results]
    spear      = stats.spearmanr(lags, delta_r2s)

    # ── Print table ────────────────────────────────────────────────────────────
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  LAG ROBUSTNESS — Trends gain by SNAP reporting delay assumption")
    print(f"  No-Trends: base + unemployment + snap_lag{{L}}")
    print(f"  With-Trends: same + {len(TRENDS_FEATURES)} Google Trends features")
    print(f"  Walk-forward XGBoost | non-COVID rows only")
    print(f"{sep}\n")

    hdr = (f"  {'Lag':>3}  {'No R²':>7}  {'Wt R²':>7}  {'ΔR²':>7}  "
           f"{'No MAE':>8}  {'Wt MAE':>8}  {'ΔMAE':>8}  "
           f"{'No sMAPE':>9}  {'Wt sMAPE':>9}  {'ΔsMAPE':>8}")
    print(hdr)
    print(f"  {'-'*76}")
    for r in results:
        marker = " ◄ MAIN" if r["lag"] == MAIN_LAG else ""
        print(
            f"  {r['lag']:>3}  {r['no_r2']:>7.4f}  {r['with_r2']:>7.4f}  "
            f"{r['delta_r2']:>+7.4f}  "
            f"{r['no_mae']:>8.6f}  {r['with_mae']:>8.6f}  {r['delta_mae']:>+8.6f}  "
            f"{r['no_smape']:>9.2f}  {r['with_smape']:>9.2f}  "
            f"{r['delta_smape']:>+8.2f}{marker}"
        )

    print(f"\n{sep}")
    print(f"  STATISTICAL SUMMARY")
    print(f"{sep}\n")

    main_r = next(r for r in results if r["lag"] == MAIN_LAG)
    print(f"  [1] Diebold-Mariano test at main lag ({MAIN_LAG} months)")
    print(f"  DM stat = {main_r['dm_stat']:+.4f}  p = {main_r['dm_p']:.4f}  "
          f"{'✓ significant' if main_r['dm_sig'] else '✗ not significant'}")

    print(f"\n  [2] Wilcoxon signed-rank at main lag ({MAIN_LAG} months)")
    print(f"  p = {main_r['wilcox_p']:.4f}  "
          f"{'✓ significant' if main_r['wilcox_sig'] else '✗ not significant'}")

    sig_s = "✓" if spear.pvalue < 0.05 else "✗"
    print(f"\n  [3] Monotonicity: does Trends gain increase with lag?")
    print(f"  Spearman ρ(lag, ΔR²) = {spear.statistic:+.4f}  p = {spear.pvalue:.4f}  {sig_s}")
    if spear.statistic > 0 and spear.pvalue < 0.05:
        print(f"  → Trends becomes MORE valuable as SNAP reporting lag grows.")
    elif spear.statistic > 0:
        print(f"  → Trend toward increasing value with lag (not significant at α=0.05).")
    else:
        print(f"  → No monotonic increase; Trends benefit is stable across lag settings.")

    avg_gain = np.mean(delta_r2s)
    min_gain = min(delta_r2s)
    print(f"\n  Avg ΔR² across lags 1–12: {avg_gain:+.4f}")
    print(f"  Min ΔR² (weakest lag):     {min_gain:+.4f}")
    if min_gain > 0:
        print(f"  → Trends consistently improves accuracy at ALL lag assumptions.")
    else:
        print(f"  → Trends improves accuracy at most but not all lag assumptions.")

    print(f"\n{sep}\n")

    # ── Save outputs ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    out_json = dict(
        description="Lag robustness: Trends benefit across 1–12 month SNAP reporting delays",
        main_lag=MAIN_LAG,
        n_trends_features=len(TRENDS_FEATURES),
        trends_features=TRENDS_FEATURES,
        base_features=BASE_FEATURES,
        results=results,
        monotonicity_spearman=dict(
            rho=round(float(spear.statistic), 4),
            p_value=round(float(spear.pvalue), 4),
            significant=bool(spear.pvalue < 0.05),
        ),
        avg_delta_r2=round(avg_gain, 4),
        min_delta_r2=round(min_gain, 4),
    )
    with open(OUT_JSON, "w") as f:
        json.dump(out_json, f, indent=2)
    log.info(f"Results → {OUT_JSON}")
    log.info(f"Table   → {OUT_CSV}")


if __name__ == "__main__":
    main()
