"""
trends_ablation.py — Does adding Google Trends significantly improve baseline accuracy?

Compares two walk-forward XGBoost models:
  A) No-Trends: demographics + unemployment + seasonality only
  B) With-Trends: same + all Google Trends features

Uses the already-tuned hyperparameters from tune_deployable_model.py.
Statistical tests: panel-aware Diebold-Mariano (1995) on monthly mean squared
forecast errors, plus Wilcoxon signed-rank on monthly mean absolute errors.
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config
from experiments.stat_tests import paired_panel_tests
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

    paired_tests = paired_panel_tests(
        noncovid,
        actual_col=TARGET,
        model_a_pred_col="pred_no",
        model_b_pred_col="pred_with",
        model_a_label="No Trends",
        model_b_label="With Trends",
    )
    dm = paired_tests["diebold_mariano"]
    wilcox = paired_tests["wilcoxon"]

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
    print(f"  [1] Diebold-Mariano test (monthly mean squared error, HLN/HAC corrected)")
    print(f"  DM stat = {dm['dm_stat']:+.4f}  p = {dm['p_value']:.4f}  {sig_dm}")
    print(f"  (positive DM = With-Trends has lower squared error)")

    sig_w = "✓ significant" if wilcox["significant_at_05_two_sided"] else "✗ not significant"
    print(f"\n  [2] Wilcoxon signed-rank test (monthly mean absolute error)")
    print(f"  stat = {wilcox['stat']:.1f}  two-sided p = {wilcox['p_value_two_sided']:.4f}  {sig_w}")
    print(f"  one-sided p (With-Trends better) = {wilcox['p_value_model_b_better']:.4f}")

    print(f"\n{sep}\n")

    # Save
    results = dict(
        statistical_test_unit="forecast month; county losses averaged within month",
        dm_sign_convention="positive means With-Trends has lower squared error than No-Trends",
        no_trends_features=NO_TRENDS_FEATURES,
        with_trends_features=DEPLOYABLE_FEATURES,
        n_trends_features_added=len(TRENDS_COLS),
        metrics_no_trends=m_no,
        metrics_with_trends=m_with,
        paired_tests=paired_tests,
        diebold_mariano=dm,
        wilcoxon=wilcox,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {OUT_JSON}")


if __name__ == "__main__":
    main()
