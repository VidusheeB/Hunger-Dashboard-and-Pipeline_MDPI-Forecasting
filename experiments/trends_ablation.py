"""
trends_ablation.py — Does adding Google Trends significantly improve baseline accuracy?

Compares two walk-forward XGBoost models:
  A) No-Trends: demographics + unemployment + seasonality only
  B) With-Trends: same + all Google Trends features

Uses the already-tuned hyperparameters from tune_deployable_model.py.
Statistical test: Diebold-Mariano (1995) on squared forecast errors,
plus Wilcoxon signed-rank as a non-parametric check.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "trends_ablation_results.json")
TARGET   = "SNAP_Application_Rate"
COVID_START, COVID_END = "2020-01-01", "2021-12-31"
WALK_FWD_MIN = config.WALK_FORWARD_MIN_MONTHS

# ── Feature sets ──────────────────────────────────────────────────────────────

TRENDS_COLS = [f for f in DEPLOYABLE_FEATURES if any(k in f for k in [
    "calfresh", "foodbank", "foodstamps", "snaptopic",
    "monthly_average_CalFresh", "monthly_average_FoodBank",
    "monthly_average_FoodStamps", "monthly_average_SNAPTopic",
])]

NO_TRENDS_FEATURES = [f for f in DEPLOYABLE_FEATURES if f not in TRENDS_COLS]


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, features: list, params: dict) -> pd.DataFrame:
    dates = sorted(df["date"].unique())
    rows  = []
    n_skip = 0
    for t in dates:
        train = df[df["date"] < t]
        test  = df[df["date"] == t]
        if train["date"].nunique() < WALK_FWD_MIN:
            n_skip += 1
            continue
        tr = train[features + [TARGET]].dropna()
        te = test[features + [TARGET]].dropna()
        if len(tr) < 10 or len(te) == 0:
            n_skip += 1
            continue
        model = XGBRegressor(**params, verbosity=0)
        model.fit(tr[features].values, tr[TARGET].values)
        preds = np.clip(model.predict(te[features].values), 0, None)
        out = test.loc[te.index, ["county", "date", TARGET]].copy()
        out["predicted"] = preds
        rows.append(out)
    log.info(f"  Walk-forward: {len(rows)} months | {n_skip} skipped")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Diebold-Mariano test ──────────────────────────────────────────────────────

def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> dict:
    """
    DM test (Harvey, Leybourne & Newbold 1997 small-sample correction).
    H0: equal forecast accuracy. Negative z = model 1 is better.
    Loss differential: d = e1² - e2²  (squared error loss).
    """
    d   = e1**2 - e2**2
    n   = len(d)
    d_bar = d.mean()
    # Newey-West variance with lag=1
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 2 else 0.0
    var_d  = (gamma0 + 2 * gamma1) / n
    if var_d <= 0:
        return dict(dm_stat=0.0, p_value=1.0, significant_at_05=False)
    dm = d_bar / np.sqrt(var_d)
    # HLN small-sample correction
    k   = 1  # 1-step ahead
    corr = np.sqrt((n + 1 - 2*k + k*(k-1)/n) / n)
    dm_adj = dm * corr
    p   = 2 * (1 - stats.t.cdf(abs(dm_adj), df=n-1))
    return dict(dm_stat=round(float(dm_adj), 4), p_value=round(float(p), 4),
                significant_at_05=bool(p < 0.05))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== TRENDS ABLATION: With vs Without Google Trends ===")

    # Load data
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    df = merge_laus(df)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Load tuned params
    params_path = os.path.join(config.OUTPUTS_ROOT, "metrics", "deployable_tuning_results.json")
    with open(params_path) as f:
        params = json.load(f)["best_params"]
    params.pop("random_state", None)
    params.pop("n_jobs", None)
    params["random_state"] = 42
    params["n_jobs"] = -1

    log.info(f"No-Trends features ({len(NO_TRENDS_FEATURES)}): {NO_TRENDS_FEATURES}")
    log.info(f"With-Trends features ({len(DEPLOYABLE_FEATURES)}): adds {len(TRENDS_COLS)} Trends cols")

    log.info("Running walk-forward WITHOUT Trends …")
    wf_no = walk_forward(df, NO_TRENDS_FEATURES, params)

    log.info("Running walk-forward WITH Trends …")
    wf_with = walk_forward(df, DEPLOYABLE_FEATURES, params)

    # Align on same rows
    merged = wf_no.merge(
        wf_with[["county", "date", "predicted"]].rename(columns={"predicted": "pred_with"}),
        on=["county", "date"],
    ).rename(columns={"predicted": "pred_no"}).dropna()

    # Non-COVID only for regression metrics
    noncovid = merged[~merged["date"].between(COVID_START, COVID_END)]

    def metrics(df_m, pred_col):
        y, yhat = df_m[TARGET].values, df_m[pred_col].values
        ss_res  = ((y - yhat)**2).sum()
        ss_tot  = ((y - y.mean())**2).sum()
        r2      = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        mae     = np.abs(y - yhat).mean()
        smape   = (2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-9)).mean() * 100
        return dict(r2=round(r2, 4), mae=round(mae, 6), smape=round(smape, 2))

    m_no   = metrics(noncovid, "pred_no")
    m_with = metrics(noncovid, "pred_with")

    e_no   = noncovid[TARGET].values - noncovid["pred_no"].values
    e_with = noncovid[TARGET].values - noncovid["pred_with"].values

    dm     = diebold_mariano(e_no, e_with)
    wilcox = stats.wilcoxon(np.abs(e_no), np.abs(e_with))

    # ── Print ─────────────────────────────────────────────────────────────────
    sep = "═" * 68
    print(f"\n{sep}")
    print(f"  TRENDS ABLATION — Does Google Trends improve CalFresh prediction?")
    print(f"  Walk-forward XGBoost | non-COVID rows only for regression metrics")
    print(f"{sep}\n")

    print(f"  {'Metric':<10}  {'No Trends':>12}  {'With Trends':>12}  {'Δ':>10}")
    print(f"  {'-'*50}")
    print(f"  {'R²':<10}  {m_no['r2']:>12.4f}  {m_with['r2']:>12.4f}  {m_with['r2']-m_no['r2']:>+10.4f}")
    print(f"  {'MAE':<10}  {m_no['mae']:>12.6f}  {m_with['mae']:>12.6f}  {m_with['mae']-m_no['mae']:>+10.6f}")
    print(f"  {'sMAPE %':<10}  {m_no['smape']:>12.2f}  {m_with['smape']:>12.2f}  {m_with['smape']-m_no['smape']:>+10.2f}")

    print(f"\n  {sep}")
    print(f"  STATISTICAL TESTS  (H0: Trends adds no forecast improvement)")
    print(f"  {sep}\n")

    sig_dm = "✓ significant" if dm["significant_at_05"] else "✗ not significant"
    print(f"  [1] Diebold-Mariano test (squared error loss, HLN corrected)")
    print(f"  DM stat = {dm['dm_stat']:+.4f}  p = {dm['p_value']:.4f}  {sig_dm}")
    print(f"  (negative DM = No-Trends has larger errors = Trends helps)")

    sig_w = "✓ significant" if wilcox.pvalue < 0.05 else "✗ not significant"
    print(f"\n  [2] Wilcoxon signed-rank test (absolute errors, non-parametric)")
    print(f"  stat = {wilcox.statistic:.1f}  p = {wilcox.pvalue:.4f}  {sig_w}")

    print(f"\n{sep}\n")

    # Save
    results = dict(
        no_trends_features=NO_TRENDS_FEATURES,
        with_trends_features=DEPLOYABLE_FEATURES,
        n_trends_features_added=len(TRENDS_COLS),
        metrics_no_trends=m_no,
        metrics_with_trends=m_with,
        diebold_mariano=dm,
        wilcoxon=dict(stat=round(float(wilcox.statistic), 2),
                      p_value=round(float(wilcox.pvalue), 4),
                      significant_at_05=bool(wilcox.pvalue < 0.05)),
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {OUT_JSON}")


if __name__ == "__main__":
    main()
