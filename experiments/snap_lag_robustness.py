"""
snap_lag_robustness.py — Does Google Trends help more when SNAP data is more delayed?

Question
--------
At SNAP reporting lag L, the most recent SNAP data available is from month t-L.
Holding this constant, does adding Google Trends (real-time) improve accuracy?

Feature sets (for each lag L = 1 … 12)
---------------------------------------
  No-Trends:  demographics + log transforms + seasonality +
              unemployment_rate + unemployment_rate_lag1 +
              snap_rate_lag{L}  (most recent available SNAP data)

  With-Trends: same + all 20 Google Trends features (CalFresh, FoodBank,
               FoodStamps, SNAPTopic) with lags, rolling stats, momentum

This answers: "given that you CAN use L-month-old SNAP data, does Trends
add further value?" Crossover point identifies when Trends starts mattering.

Walk-forward: standard (train on date < t, predict t). Lag L only affects
which snap_rate column is included as a feature, not the training window.

Outputs
-------
  outputs/metrics/snap_lag_robustness_results.json
  outputs/metrics/snap_lag_robustness_table.csv
"""

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

TARGET      = "SNAP_Application_Rate"
COVID_START = "2020-01-01"
COVID_END   = "2021-12-31"
MAIN_LAG    = 9   # where Trends gain is strongest for DM test

OUT_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "snap_lag_robustness_results.json")
OUT_CSV  = os.path.join(config.OUTPUTS_ROOT, "metrics", "snap_lag_robustness_table.csv")
TUNE_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "deployable_tuning_results.json")

BASE_FEATURES   = [
    "Population", "Median_Income",
    "unemployment_rate", "unemployment_rate_lag1",
    "month_sin", "month_cos", "quarter", "month",
    "log_population", "log_income",
]
TRENDS_FEATURES = [f for f in DEPLOYABLE_FEATURES if f not in BASE_FEATURES]


def _smape(actual, predicted):
    mask = (np.abs(actual) + np.abs(predicted)) > 0
    return float(100 * np.mean(
        np.abs(actual[mask] - predicted[mask]) /
        ((np.abs(actual[mask]) + np.abs(predicted[mask])) / 2)
    ))


def _metrics(y, yhat):
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2  = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.abs(y - yhat).mean())
    return dict(r2=round(r2, 4), mae=round(mae, 6), smape=round(_smape(y, yhat), 2))


def diebold_mariano(e1, e2):
    d      = e1 ** 2 - e2 ** 2
    n      = len(d)
    d_bar  = d.mean()
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 2 else 0.0
    var_d  = (gamma0 + 2 * gamma1) / n
    if var_d <= 0:
        return dict(dm_stat=0.0, p_value=1.0, significant=False)
    dm     = d_bar / np.sqrt(var_d)
    corr   = np.sqrt((n + 1 - 2 + 1 * 0 / n) / n)
    dm_adj = dm * corr
    p      = float(2 * (1 - stats.t.cdf(abs(dm_adj), df=n - 1)))
    return dict(dm_stat=round(float(dm_adj), 4), p_value=round(p, 4),
                significant=bool(p < 0.05))


def walk_forward(df, features, params, min_months):
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


def main():
    log.info("=== SNAP LAG ROBUSTNESS: Trends benefit at each SNAP reporting lag ===")
    log.info("    snap_rate_lag{L} included as most recent available SNAP feature.")

    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    df = merge_laus(df)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["county", "date"]).reset_index(drop=True)
    log.info(f"Loaded {len(df):,} rows | {df['county'].nunique()} counties")

    with open(TUNE_JSON) as f:
        params = json.load(f)["best_params"]
    params.pop("random_state", None)
    params.pop("n_jobs", None)
    params["random_state"] = 42
    params["n_jobs"]       = -1

    min_months = config.WALK_FORWARD_MIN_MONTHS
    results    = []

    for lag in range(1, 13):
        snap_col = f"snap_rate_lag{lag}"
        df[snap_col] = df.groupby("county")[TARGET].shift(lag)

        feats_no   = BASE_FEATURES + [snap_col]
        feats_with = BASE_FEATURES + TRENDS_FEATURES + [snap_col]

        log.info(f"Lag {lag:2d}: walk-forward NO Trends ({len(feats_no)} features) …")
        wf_no   = walk_forward(df, feats_no,   params, min_months)

        log.info(f"Lag {lag:2d}: walk-forward WITH Trends ({len(feats_with)} features) …")
        wf_with = walk_forward(df, feats_with, params, min_months)

        merged = wf_no.merge(
            wf_with[["county", "date", "predicted"]].rename(columns={"predicted": "pred_with"}),
            on=["county", "date"],
        ).rename(columns={"predicted": "pred_no"}).dropna()

        noncovid = merged[~merged["date"].between(COVID_START, COVID_END)]
        y, p_no, p_wt = (noncovid[TARGET].values,
                         noncovid["pred_no"].values,
                         noncovid["pred_with"].values)

        m_no = _metrics(y, p_no)
        m_wt = _metrics(y, p_wt)

        row = dict(
            lag=lag, n=len(noncovid),
            no_r2=m_no["r2"],      with_r2=m_wt["r2"],
            delta_r2=round(m_wt["r2"]    - m_no["r2"],    4),
            no_mae=m_no["mae"],    with_mae=m_wt["mae"],
            delta_mae=round(m_wt["mae"]   - m_no["mae"],   6),
            no_smape=m_no["smape"],with_smape=m_wt["smape"],
            delta_smape=round(m_wt["smape"] - m_no["smape"], 2),
        )
        if lag == MAIN_LAG:
            dm     = diebold_mariano(y - p_no, y - p_wt)
            wilcox = stats.wilcoxon(np.abs(y - p_no), np.abs(y - p_wt))
            row.update(dm_stat=dm["dm_stat"], dm_p=dm["p_value"], dm_sig=dm["significant"],
                       wilcox_p=round(float(wilcox.pvalue), 4),
                       wilcox_sig=bool(wilcox.pvalue < 0.05))
        else:
            row.update(dm_stat=None, dm_p=None, dm_sig=None,
                       wilcox_p=None, wilcox_sig=None)

        results.append(row)
        log.info(f"  Lag {lag:2d}: no-Trends R²={m_no['r2']:.4f}  "
                 f"with-Trends R²={m_wt['r2']:.4f}  ΔR²={m_wt['r2']-m_no['r2']:+.4f}")

        df.drop(columns=[snap_col], inplace=True)

    lags, delta_r2s = [r["lag"] for r in results], [r["delta_r2"] for r in results]
    spear = stats.spearmanr(lags, delta_r2s)

    sep = "═" * 82
    print(f"\n{sep}")
    print(f"  SNAP LAG ROBUSTNESS — Trends gain at each SNAP reporting lag")
    print(f"  No-Trends: base + unemployment + snap_rate_lag{{L}}")
    print(f"  With-Trends: same + {len(TRENDS_FEATURES)} Google Trends features")
    print(f"  Walk-forward XGBoost | non-COVID rows only")
    print(f"{sep}\n")
    print(f"  {'Lag':>4}  {'No R²':>7}  {'Wt R²':>7}  {'ΔR²':>7}  "
          f"{'No MAE':>8}  {'Wt MAE':>8}  {'No sMAPE':>9}  {'Wt sMAPE':>9}  {'ΔsMAPE':>8}")
    print(f"  {'-'*77}")
    for r in results:
        marker = " ◄ DM tested" if r["lag"] == MAIN_LAG else ""
        print(f"  {r['lag']:>4}  {r['no_r2']:>7.4f}  {r['with_r2']:>7.4f}  "
              f"{r['delta_r2']:>+7.4f}  {r['no_mae']:>8.6f}  {r['with_mae']:>8.6f}  "
              f"{r['no_smape']:>9.2f}  {r['with_smape']:>9.2f}  "
              f"{r['delta_smape']:>+8.2f}{marker}")

    print(f"\n{sep}")
    print(f"  STATISTICAL SUMMARY")
    print(f"{sep}\n")
    sig_s = "✓" if spear.pvalue < 0.05 else "✗"
    print(f"  Spearman ρ(lag, ΔR²) = {spear.statistic:+.4f}  p = {spear.pvalue:.4f}  {sig_s}")
    main_r = next(r for r in results if r["lag"] == MAIN_LAG)
    print(f"  DM at lag {MAIN_LAG}: stat={main_r['dm_stat']:+.4f}  "
          f"p={main_r['dm_p']:.4f}  {'✓' if main_r['dm_sig'] else '✗'}")
    print(f"  Wilcoxon at lag {MAIN_LAG}: p={main_r['wilcox_p']:.4f}  "
          f"{'✓' if main_r['wilcox_sig'] else '✗'}")
    crossover = next((r["lag"] for r in results if r["delta_r2"] > 0), None)
    if crossover:
        print(f"  Crossover lag (Trends first positive): {crossover} months")
    print(f"\n{sep}\n")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(
            description="SNAP lag robustness: Trends gain when snap_rate_lag{L} included",
            main_lag=MAIN_LAG, base_features=BASE_FEATURES,
            trends_features=TRENDS_FEATURES, results=results,
            monotonicity_spearman=dict(rho=round(float(spear.statistic), 4),
                                       p_value=round(float(spear.pvalue), 4),
                                       significant=bool(spear.pvalue < 0.05)),
            crossover_lag=crossover,
        ), f, indent=2)
    log.info(f"Results → {OUT_JSON}")
    log.info(f"Table   → {OUT_CSV}")


if __name__ == "__main__":
    main()
