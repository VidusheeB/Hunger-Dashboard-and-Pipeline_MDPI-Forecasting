"""
binary_alert_model.py — Binary XGBoost classifier: Green vs Not-Green.

Predicts whether next month's CalFresh applications will deviate significantly
above the county's baseline (Not-Green = Yellow or Red threshold exceeded).

Uses ALL deployable features: Trends + unemployment + demographics + seasonality
+ baseline model's predicted rate. Walk-forward validation only.

Target: is_notgreen = 1 if deviation_t1 > county Yellow threshold (75th pctile
        of that county's positive deviations).
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config
from experiments.tune_deployable_model import merge_laus, DEPLOYABLE_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_CSV   = config.FEATURES_CSV
WF_PREDS_CSV   = os.path.join(config.OUTPUTS_ROOT, "metrics", "deployable_walkforward_predictions.csv")
OUT_JSON       = os.path.join(config.OUTPUTS_ROOT, "metrics", "binary_alert_results.json")

YELLOW_PCT     = 75   # Not-Green threshold: top 25% of positive deviations per county
WALK_FWD_MIN   = config.WALK_FORWARD_MIN_MONTHS

# All features: deployable set + baseline predicted rate
ALL_FEATURES   = DEPLOYABLE_FEATURES + ["predicted_rate"]

TARGET         = "is_notgreen"


# ── Label construction ────────────────────────────────────────────────────────

def build_dataset(features_csv: str, wf_csv: str) -> pd.DataFrame:
    """
    Join features at t with actual/predicted at t+1 to build (features_t, label_t+1).
    Label = 1 if next month's deviation from baseline exceeds county Yellow threshold.
    """
    df  = pd.read_csv(features_csv, parse_dates=["date"])
    wf  = pd.read_csv(wf_csv,       parse_dates=["date"])

    df  = merge_laus(df)

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    wf["date"] = wf["date"].dt.to_period("M").dt.to_timestamp()

    # t+1 join: get baseline and actual for next month
    df["date_t1"] = (df["date"].dt.to_period("M") + 1).dt.to_timestamp()
    wf_t1 = wf.rename(columns={
        "date":           "date_t1",
        "predicted_rate": "baseline_t1",
        "actual_rate":    "actual_t1",
    })[["county", "date_t1", "baseline_t1", "actual_t1"]]

    paired = df.merge(wf_t1, on=["county", "date_t1"], how="inner")
    paired = paired[paired["baseline_t1"] > 0].copy()
    paired["deviation_t1"] = (paired["actual_t1"] - paired["baseline_t1"]) / paired["baseline_t1"]

    # Also join baseline predicted rate at t (as a feature)
    wf_t = wf.rename(columns={"predicted_rate": "predicted_rate"})[["county", "date", "predicted_rate", "actual_rate"]]
    paired = paired.merge(wf_t, on=["county", "date"], how="left")

    # County-specific Yellow threshold (75th pctile of positive deviations)
    county_thresholds = {}
    for county, grp in paired.groupby("county"):
        pos = grp.loc[grp["deviation_t1"] > 0, "deviation_t1"]
        county_thresholds[county] = np.percentile(pos, YELLOW_PCT) if len(pos) >= 4 else float("inf")

    paired["yellow_thr"] = paired["county"].map(county_thresholds)
    paired[TARGET] = (paired["deviation_t1"] > paired["yellow_thr"]).astype(int)

    log.info(f"Paired dataset: {len(paired):,} county-months | "
             f"Not-Green={paired[TARGET].sum():,} ({100*paired[TARGET].mean():.1f}%)")
    return paired


# ── Walk-forward classifier ───────────────────────────────────────────────────

def walk_forward_classify(paired: pd.DataFrame) -> pd.DataFrame:
    """
    For each month T: train XGBoost on all prior months, predict T.
    Returns scored DataFrame with alert_proba column.
    """
    dates = sorted(paired["date"].unique())
    rows  = []
    n_skipped = 0

    for i, t in enumerate(dates):
        train = paired[paired["date"] < t]
        test  = paired[paired["date"] == t]

        if train["date"].nunique() < WALK_FWD_MIN:
            n_skipped += 1
            continue

        feat_cols = [f for f in ALL_FEATURES if f in paired.columns]
        tr = train[feat_cols + [TARGET]].dropna()
        te = test[feat_cols + [TARGET]].dropna()
        if len(tr) < 10 or len(te) == 0:
            n_skipped += 1
            continue

        pos_rate = tr[TARGET].mean()
        scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(tr[feat_cols].values, tr[TARGET].values)
        proba = model.predict_proba(te[feat_cols].values)[:, 1]

        result = test.loc[te.index, ["county", "date", TARGET, "deviation_t1"]].copy()
        result["alert_proba"] = proba
        rows.append(result)

    log.info(f"Walk-forward: {len(rows)} months predicted | {n_skipped} skipped")
    return pd.concat(rows, ignore_index=True)


# ── Statistical tests ─────────────────────────────────────────────────────────

def delong_test(y_true, score1, score2):
    """DeLong (1988) AUC comparison."""
    def auc_components(y, s):
        pos = s[y == 1]; neg = s[y == 0]
        m, n = len(pos), len(neg)
        psi_pos = np.array([(p > neg).mean() + 0.5*(p == neg).mean() for p in pos])
        psi_neg = np.array([(n_ < pos).mean() + 0.5*(n_ == pos).mean() for n_ in neg])
        return psi_pos, psi_neg, m, n
    p1, n1, m1, n_1 = auc_components(y_true, score1)
    p2, n2, m2, n_2 = auc_components(y_true, score2)
    auc1, auc2 = p1.mean(), p2.mean()
    s01 = (np.var(p1)/m1) + (np.var(n1)/n_1)
    s02 = (np.var(p2)/m2) + (np.var(n2)/n_2)
    s12 = (np.cov(p1, p2)[0,1]/m1) + (np.cov(n1, n2)[0,1]/n_1)
    var = s01 + s02 - 2*s12
    z   = (auc1 - auc2) / np.sqrt(var) if var > 0 else 0.0
    p   = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p


def permutation_test(paired: pd.DataFrame, n_perm: int = 500) -> dict:
    """Shuffle labels within county on a 70/30 temporal split."""
    dates   = sorted(paired["date"].unique())
    cutoff  = dates[int(len(dates) * 0.7)]
    train   = paired[paired["date"] < cutoff]
    test    = paired[paired["date"] >= cutoff]
    feat_cols = [f for f in ALL_FEATURES if f in paired.columns]

    tr = train[feat_cols + [TARGET]].dropna()
    te = test[feat_cols + [TARGET]].dropna()
    if len(te) < 20:
        return {}

    pos_rate = tr[TARGET].mean()
    scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    def fit_predict(labels):
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          scale_pos_weight=scale_pos, use_label_encoder=False,
                          eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
        m.fit(tr[feat_cols].values, labels)
        return m.predict_proba(te[feat_cols].values)[:, 1]

    obs_proba = fit_predict(tr[TARGET].values)
    obs_auc   = roc_auc_score(te[TARGET].values, obs_proba)

    null_aucs = []
    for _ in range(n_perm):
        shuffled = train.copy()
        shuffled[TARGET] = (shuffled.groupby("county")[TARGET]
                            .transform(lambda s: s.sample(frac=1).values))
        tr_s = shuffled[feat_cols + [TARGET]].dropna()
        try:
            p = fit_predict(tr_s[TARGET].values)
            null_aucs.append(roc_auc_score(te[TARGET].values, p))
        except Exception:
            pass

    null_arr = np.array(null_aucs)
    p_val    = (null_arr >= obs_auc).mean()
    return dict(observed_auc=round(obs_auc, 4), null_mean=round(null_arr.mean(), 4),
                null_std=round(null_arr.std(), 4), null_95th=round(np.percentile(null_arr, 95), 4),
                p_value=round(p_val, 4), significant_at_05=bool(p_val < 0.05))


def bootstrap_ci(scored: pd.DataFrame, n_boot: int = 500) -> dict:
    """Cluster bootstrap by county."""
    counties = scored["county"].unique()
    aucs, pr_aucs = [], []
    for _ in range(n_boot):
        s_counties = resample(counties, replace=True, random_state=None)
        boot = pd.concat([scored[scored["county"] == c] for c in s_counties])
        boot = boot.dropna(subset=["alert_proba", TARGET])
        if boot[TARGET].nunique() < 2:
            continue
        aucs.append(roc_auc_score(boot[TARGET], boot["alert_proba"]))
        pr_aucs.append(average_precision_score(boot[TARGET], boot["alert_proba"]))
    return dict(
        roc_auc=round(np.mean(aucs), 4),
        roc_ci_lo=round(np.percentile(aucs, 2.5), 4),
        roc_ci_hi=round(np.percentile(aucs, 97.5), 4),
        pr_auc=round(np.mean(pr_aucs), 4),
        pr_ci_lo=round(np.percentile(pr_aucs, 2.5), 4),
        pr_ci_hi=round(np.percentile(pr_aucs, 97.5), 4),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== BINARY ALERT MODEL: Green vs Not-Green ===")

    paired = build_dataset(FEATURES_CSV, WF_PREDS_CSV)
    scored = walk_forward_classify(paired)

    valid  = scored.dropna(subset=["alert_proba", TARGET])
    y_true = valid[TARGET].values
    y_prob = valid["alert_proba"].values

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)

    # Persistence baseline: is_notgreen at t predicts is_notgreen at t+1
    persist = paired.sort_values(["county","date"]).copy()
    persist["persist_score"] = persist.groupby("county")[TARGET].shift(1)
    persist = persist.dropna(subset=["persist_score", TARGET])
    persist_roc = roc_auc_score(persist[TARGET], persist["persist_score"])

    # Threshold sweep — find best recall/precision tradeoffs
    thresholds = np.arange(0.05, 0.95, 0.05)
    sweep = []
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tp = ((pred==1)&(y_true==1)).sum()
        fp = ((pred==1)&(y_true==0)).sum()
        fn = ((pred==0)&(y_true==1)).sum()
        tn = ((pred==0)&(y_true==0)).sum()
        recall    = tp/(tp+fn) if (tp+fn)>0 else 0
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        fpr       = fp/(fp+tn) if (fp+tn)>0 else 0
        sweep.append(dict(threshold=round(thr,2), recall=round(recall,3),
                          precision=round(precision,3), f1=round(f1,3),
                          fpr=round(fpr,3), tp=int(tp), fp=int(fp),
                          fn=int(fn), tn=int(tn)))

    # Best threshold: Youden's J
    best = max(sweep, key=lambda x: x["recall"] - x["fpr"])

    # Statistical tests
    log.info("Running permutation test (500 permutations) …")
    perm = permutation_test(paired, n_perm=500)
    log.info("Running bootstrap CI (500 iterations) …")
    boot = bootstrap_ci(valid, n_boot=500)

    # DeLong vs persistence
    merged = valid.merge(
        persist[["county","date","persist_score"]],
        on=["county","date"], how="inner"
    ).dropna()
    if len(merged) > 0 and merged[TARGET].nunique() == 2:
        dl_auc1, dl_auc2, dl_z, dl_p = delong_test(
            merged[TARGET].values, merged["alert_proba"].values, merged["persist_score"].values
        )
    else:
        dl_auc1 = dl_auc2 = dl_z = dl_p = None

    # ── Print ─────────────────────────────────────────────────────────────────
    sep = "═"*68
    print(f"\n{sep}")
    print(f"  BINARY ALERT: Green vs Not-Green")
    print(f"  Features: ALL deployable (Trends + unemployment + demographics + baseline)")
    print(f"  Not-Green = deviation > county Yellow threshold ({YELLOW_PCT}th pctile of +devs)")
    print(f"{sep}")

    print(f"\n  Walk-forward results:")
    print(f"    n = {len(valid):,}  |  Not-Green = {int(y_true.sum()):,} ({100*y_true.mean():.1f}%)")
    print(f"    ROC AUC  : {roc_auc:.4f}")
    print(f"    PR  AUC  : {pr_auc:.4f}")
    print(f"    Persistence ROC AUC: {persist_roc:.4f}")

    print(f"\n  Best operating point (Youden's J):")
    b = best
    print(f"    Threshold : {b['threshold']}")
    print(f"    Recall    : {b['recall']:.3f}  ({b['tp']} caught / {b['tp']+b['fn']} true not-green)")
    print(f"    Precision : {b['precision']:.3f}  (1 true catch per {1/b['precision']:.1f} flags)")
    print(f"    FPR       : {b['fpr']:.3f}  ({b['fp']} false alarms)")
    print(f"    TP/FP/FN/TN: {b['tp']}/{b['fp']}/{b['fn']}/{b['tn']}")

    print(f"\n  Threshold sweep:")
    print(f"  {'Thr':>5}  {'Recall':>7}  {'Prec':>7}  {'F1':>6}  {'FPR':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*60}")
    for s in sweep[::2]:  # every other row
        print(f"  {s['threshold']:>5.2f}  {s['recall']:>7.3f}  {s['precision']:>7.3f}  "
              f"{s['f1']:>6.3f}  {s['fpr']:>6.3f}  {s['tp']:>5}  {s['fp']:>5}  {s['fn']:>5}")

    print(f"\n{sep}")
    print(f"  STATISTICAL VALIDATION")
    print(f"{sep}")

    print(f"\n  [1] Permutation test (n=500)")
    if perm:
        print(f"  Observed AUC : {perm['observed_auc']}")
        print(f"  Null AUC     : {perm['null_mean']} ± {perm['null_std']} (95th = {perm['null_95th']})")
        print(f"  p-value      : {perm['p_value']}  {'✓ significant' if perm['significant_at_05'] else '✗ not significant'}")

    print(f"\n  [2] Bootstrap CI (n=500, cluster by county)")
    print(f"  ROC AUC : {boot['roc_auc']}  (95% CI: {boot['roc_ci_lo']} – {boot['roc_ci_hi']})")
    print(f"  PR  AUC : {boot['pr_auc']}  (95% CI: {boot['pr_ci_lo']} – {boot['pr_ci_hi']})")

    print(f"\n  [3] DeLong test (Binary XGBoost vs Persistence)")
    if dl_auc1 is not None:
        sig = "✓" if dl_p < 0.05 else "✗"
        print(f"  AUC {dl_auc1:.4f} vs {dl_auc2:.4f}  Δ={dl_auc1-dl_auc2:+.4f}  "
              f"z={dl_z:.2f}  p={dl_p:.4f}  {sig}")

    print(f"\n{sep}\n")

    # Save JSON
    results = dict(
        model="Binary XGBoost — Green vs Not-Green",
        features=ALL_FEATURES,
        n_features=len(ALL_FEATURES),
        yellow_threshold_pct=YELLOW_PCT,
        n_total=len(valid),
        n_notgreen=int(y_true.sum()),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        persistence_roc_auc=round(persist_roc, 4),
        best_threshold=best,
        threshold_sweep=sweep,
        permutation_test=perm,
        bootstrap_ci=boot,
        delong_vs_persistence=dict(
            auc_model=round(dl_auc1, 4) if dl_auc1 else None,
            auc_persistence=round(dl_auc2, 4) if dl_auc2 else None,
            z=round(dl_z, 4) if dl_z else None,
            p_value=round(dl_p, 4) if dl_p else None,
        ) if dl_auc1 else None,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {OUT_JSON}")


if __name__ == "__main__":
    main()
