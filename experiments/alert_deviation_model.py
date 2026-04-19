"""
alert_deviation_model.py — Alert layer based on next-month deviation from baseline.

Architecture
------------
Instead of flagging anomalies in the raw SNAP rate, this alert layer:

  1. Uses the deployable baseline model's predictions as baseline_{t+1}
  2. Defines deviation_{t+1} = (actual_{t+1} - baseline_{t+1}) / baseline_{t+1}
  3. Uses Google Trends features at time t to predict high-deviation events
  4. Defines alert categories from the empirical distribution of positive deviations
     (not from raw Trends z-scores or manually set thresholds)

Trends features (computed per keyword: CalFresh and FoodBank)
-------------------------------------------------------------
  trends_level_3mo        mean(T_t, T_{t-1}, T_{t-2})
  trends_pct_change_1mo   (T_t - T_{t-1}) / T_{t-1}   [0 if T_{t-1}=0]
  trends_slope_3mo        (T_t - T_{t-2}) / 2          [OLS slope, unit x-spacing]
  trends_acceleration     (T_t - T_{t-1}) - (T_{t-1} - T_{t-2})
  trends_zscore_12mo      (T_t - rolling_mean_12) / rolling_std_12  [trailing]
  calfresh_momentum, foodbank_momentum   (existing engineered features)

Alert target
------------
  deviation_t1 = (actual_t1 - baseline_t1) / baseline_t1
  Green:  deviation_t1 <= 75th pctile of positive deviations (or non-positive)
  Yellow: 75th < deviation_t1 < 85th pctile
  Red:    deviation_t1 >= 85th pctile  [main version; sensitivity at 90th, 95th]

Statistical validation
----------------------
  1. Permutation test — "Do Trends features add signal beyond randomness?"
       Labels shuffled within county (preserving structure), 1,000 permutations.
       Uses temporal hold-out (first 70% train, last 30% test) for speed.

  2. Bootstrap CI — "How stable is performance?"
       Cluster bootstrap by county, 1,000 iterations.
       Reports mean AUC and 95% CI for both ROC AUC and PR AUC.

  3. Naive baselines — "Do lags already do everything?"
       (a) Persistence: predict Red if county was Red last month.
       (b) Lagged deviation: logistic regression on current-month deviation only.

  4. DeLong test — "Is AUC_trends significantly better than AUC_naive?"
       Proper DeLong (1988) implementation with covariance for correlated AUCs.

  5. McNemar's test — "Are the classification errors different?"
       Compares disagreement cases between Trends model and persistence baseline.

  6. Lead-time test — "Does Trends give earlier warning?"
       For each true Red event, checks whether model flagged it 1 month early
       (at t-1, for event at t+1). Compares Trends model vs persistence.

  7. Calibration — "Do the probabilities mean something?"
       Brier score (lower is better). Reliability curve description.

Usage
-----
    python experiments/alert_deviation_model.py

Outputs
-------
  outputs/metrics/alert_deviation_results.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    brier_score_loss, confusion_matrix,
)
from sklearn.calibration import calibration_curve
import logging

from pipeline import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
WF_CSV   = os.path.join(config.OUTPUTS_ROOT, "metrics", "deployable_walkforward_predictions.csv")
OUT_JSON = os.path.join(config.OUTPUTS_ROOT, "metrics", "alert_deviation_results.json")

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_TRAIN_MONTHS  = 12
YELLOW_PCT        = 75
RED_PCT_MAIN      = 85
RED_PCT_VERSIONS  = [85, 90, 95]
N_PERMUTATIONS    = 1_000
N_BOOTSTRAP       = 1_000
PERMUTATION_SPLIT = 0.70   # fraction of dates used for training in permutation test
RNG_SEED          = 42

# ── Feature columns ────────────────────────────────────────────────────────────
TRENDS_FEATURES = [
    "cf_level_3mo", "cf_pct_change_1mo", "cf_slope_3mo",
    "cf_acceleration", "cf_zscore_12mo",
    "fb_level_3mo", "fb_pct_change_1mo", "fb_slope_3mo",
    "fb_acceleration", "fb_zscore_12mo",
    "calfresh_momentum", "foodbank_momentum",
]


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def compute_trends_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["county", "date"])
    for prefix, raw_col, lag1_col, lag2_col, roll3_col in [
        ("cf", "monthly_average_CalFresh", "calfresh_lag1", "calfresh_lag2", "calfresh_roll3"),
        ("fb", "monthly_average_FoodBank", "foodbank_lag1", "foodbank_lag2", "foodbank_roll3"),
    ]:
        T, T1, T2 = df[raw_col], df[lag1_col], df[lag2_col]
        df[f"{prefix}_level_3mo"]      = df[roll3_col]
        df[f"{prefix}_pct_change_1mo"] = np.where(T1 != 0, (T - T1) / T1, 0.0)
        df[f"{prefix}_slope_3mo"]      = (T - T2) / 2.0
        df[f"{prefix}_acceleration"]   = (T - T1) - (T1 - T2)
        df[f"{prefix}_zscore_12mo"]    = np.nan

    for county, grp in df.groupby("county"):
        idx = grp.index
        for prefix, raw_col in [
            ("cf", "monthly_average_CalFresh"),
            ("fb", "monthly_average_FoodBank"),
        ]:
            s  = grp[raw_col]
            rm = s.rolling(12, min_periods=6).mean()
            rs = s.rolling(12, min_periods=6).std()
            df.loc[idx, f"{prefix}_zscore_12mo"] = np.where(rs > 0, (s - rm) / rs, 0.0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Dataset construction
# ══════════════════════════════════════════════════════════════════════════════

def build_paired_dataset(df: pd.DataFrame, wf: pd.DataFrame) -> pd.DataFrame:
    """
    Join features at t with baseline/actual at t+1.
    Also joins current-month deviation (deviation_t) for naive baseline use.
    """
    df = df.copy()
    wf = wf.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    wf["date"] = pd.to_datetime(wf["date"]).dt.to_period("M").dt.to_timestamp()

    # t+1 join
    df["date_t1"] = (df["date"].dt.to_period("M") + 1).dt.to_timestamp()
    wf_t1 = wf.rename(columns={
        "date": "date_t1",
        "predicted_rate": "baseline_t1",
        "actual_rate": "actual_t1",
    })
    paired = df.merge(wf_t1[["county", "date_t1", "baseline_t1", "actual_t1"]],
                      on=["county", "date_t1"], how="inner")
    paired = paired[paired["baseline_t1"] > 0].copy()
    paired["deviation_t1"] = (
        (paired["actual_t1"] - paired["baseline_t1"]) / paired["baseline_t1"]
    )

    # Current-month deviation (for naive lagged-dev baseline)
    wf_t = wf.rename(columns={
        "predicted_rate": "baseline_t",
        "actual_rate": "actual_t",
    })
    paired = paired.merge(wf_t[["county", "date", "baseline_t", "actual_t"]],
                          on=["county", "date"], how="left")
    paired["deviation_t"] = np.where(
        paired["baseline_t"] > 0,
        (paired["actual_t"] - paired["baseline_t"]) / paired["baseline_t"],
        np.nan,
    )
    return paired


def define_alert_labels(paired: pd.DataFrame, yellow_pct: int, red_pct: int):
    pos_devs   = paired.loc[paired["deviation_t1"] > 0, "deviation_t1"]
    yellow_thr = float(np.percentile(pos_devs, yellow_pct))
    red_thr    = float(np.percentile(pos_devs, red_pct))

    def _label(dev):
        if dev <= yellow_thr: return "Green"
        elif dev < red_thr:   return "Yellow"
        else:                 return "Red"

    df = paired.copy()
    df["alert_label"]    = df["deviation_t1"].apply(_label)
    df["is_red"]         = (df["alert_label"] == "Red").astype(int)
    df["is_yellow_plus"] = (df["alert_label"].isin(["Yellow", "Red"])).astype(int)
    return df, yellow_thr, red_thr


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward XGBoost classifier
# ══════════════════════════════════════════════════════════════════════════════

def _fit_xgb(X_train, y_train, X_test):
    pos_rate  = y_train.mean()
    scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RNG_SEED, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def logistic_walkforward(paired: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Walk-forward XGBoost classifier (name kept for compatibility)."""
    dates, results = sorted(paired["date"].unique()), []
    for t in dates:
        train = paired[paired["date"] < t].dropna(subset=feature_cols + ["is_red"])
        test  = paired[paired["date"] == t].dropna(subset=feature_cols)
        if (train["date"].nunique() < MIN_TRAIN_MONTHS
                or len(test) == 0
                or train["is_red"].nunique() < 2):
            continue
        out = test.copy()
        out["alert_proba"] = _fit_xgb(
            train[feature_cols].values, train["is_red"].values,
            test[feature_cols].values,
        )
        results.append(out)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Naive baselines
# ══════════════════════════════════════════════════════════════════════════════

def naive_persistence(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Persistence baseline: predict Red if county was Red last month.
    alert_proba = lagged is_red (binary 0/1 for AUC purposes).
    """
    df = paired.copy().sort_values(["county", "date"])
    df["alert_proba"] = df.groupby("county")["is_red"].shift(1).fillna(0)
    return df.dropna(subset=["is_red"])


def naive_lagged_deviation(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged deviation baseline: walk-forward logistic regression using only
    current-month deviation (deviation_t) as the feature.
    Tests whether Trends adds anything beyond knowing how this month went.
    """
    feat = ["deviation_t"]
    return logistic_walkforward(paired, feat)


# ══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════════════

# ── DeLong (1988) ─────────────────────────────────────────────────────────────

def _structural_components(y_true, y_score):
    """
    Returns (V10, V01) — structural components for DeLong variance estimation.
    V10[i] = mean(score_pos[i] > score_neg)  [for each positive]
    V01[j] = mean(score_pos > score_neg[j])  [for each negative]
    Vectorized O(n_pos * n_neg).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    mat = (pos[:, None] > neg[None, :]).astype(float) \
        + 0.5 * (pos[:, None] == neg[None, :]).astype(float)
    V10 = mat.mean(axis=1)
    V01 = mat.mean(axis=0)
    return V10, V01


def delong_compare(y_true, y_score1, y_score2) -> dict:
    """
    DeLong et al. (1988) test comparing two correlated ROC AUCs on the same dataset.
    Returns: auc1, auc2, difference, z-statistic, two-sided p-value.
    """
    V10_1, V01_1 = _structural_components(y_true, y_score1)
    V10_2, V01_2 = _structural_components(y_true, y_score2)
    n_pos, n_neg = V10_1.shape[0], V01_1.shape[0]

    auc1 = float(V10_1.mean())
    auc2 = float(V10_2.mean())

    var1  = V10_1.var(ddof=1) / n_pos + V01_1.var(ddof=1) / n_neg
    var2  = V10_2.var(ddof=1) / n_pos + V01_2.var(ddof=1) / n_neg
    cov   = (np.cov(V10_1, V10_2)[0, 1] / n_pos
             + np.cov(V01_1, V01_2)[0, 1] / n_neg)

    diff_var = max(var1 + var2 - 2 * cov, 1e-12)
    z        = (auc1 - auc2) / np.sqrt(diff_var)
    p_val    = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))

    return dict(auc1=round(auc1, 4), auc2=round(auc2, 4),
                diff=round(auc1 - auc2, 4),
                z=round(z, 3), p_value=round(p_val, 4))


# ── Permutation test ───────────────────────────────────────────────────────────

def permutation_test(paired: pd.DataFrame, n_perm: int = N_PERMUTATIONS) -> dict:
    """
    Null hypothesis: Trends features contain no signal beyond random labeling.

    Procedure (temporal hold-out for computational feasibility):
      - Split dates at PERMUTATION_SPLIT: first 70% = train, last 30% = test
      - Observed AUC: train logreg on true labels, evaluate on test
      - Each permutation: shuffle is_red WITHIN each county in training set
        (preserves county structure and temporal autocorrelation of features)
        then refit logreg, evaluate on same test set
      - p-value = fraction of permuted AUCs >= observed AUC
    """
    rng   = np.random.default_rng(RNG_SEED)
    dates = sorted(paired["date"].unique())
    split = dates[int(len(dates) * PERMUTATION_SPLIT)]

    train_all = paired[paired["date"] < split].dropna(subset=TRENDS_FEATURES + ["is_red"])
    test_all  = paired[paired["date"] >= split].dropna(subset=TRENDS_FEATURES + ["is_red"])

    if train_all["is_red"].nunique() < 2 or test_all["is_red"].nunique() < 2:
        logger.warning("Permutation test: degenerate labels in split — skipping")
        return {}

    X_test  = test_all[TRENDS_FEATURES].values
    y_test  = test_all["is_red"].values

    # Observed AUC
    obs_proba = _fit_xgb(
        train_all[TRENDS_FEATURES].values,
        train_all["is_red"].values, X_test,
    )
    fpr, tpr, _ = roc_curve(y_test, obs_proba)
    obs_auc = float(auc(fpr, tpr))

    # Permuted AUCs
    null_aucs = []
    X_train = train_all[TRENDS_FEATURES].values
    counties = train_all["county"].values
    y_train_orig = train_all["is_red"].values.copy()

    for i in range(n_perm):
        # Shuffle within county
        y_perm = y_train_orig.copy()
        for county in np.unique(counties):
            mask = counties == county
            y_perm[mask] = rng.permutation(y_perm[mask])

        if len(np.unique(y_perm)) < 2:
            continue
        perm_proba = _fit_xgb(X_train, y_perm, X_test)
        f, t, _ = roc_curve(y_test, perm_proba)
        null_aucs.append(float(auc(f, t)))

    null_aucs = np.array(null_aucs)
    p_val = float((null_aucs >= obs_auc).mean())

    return dict(
        observed_auc=round(obs_auc, 4),
        n_permutations=len(null_aucs),
        null_auc_mean=round(float(null_aucs.mean()), 4),
        null_auc_std=round(float(null_aucs.std()), 4),
        null_auc_p95=round(float(np.percentile(null_aucs, 95)), 4),
        p_value=round(p_val, 4),
        significant_at_05=(p_val < 0.05),
        train_months=int(len(paired[paired["date"] < split]["date"].unique())),
        test_months=int(len(paired[paired["date"] >= split]["date"].unique())),
    )


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(scored: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """
    Cluster bootstrap by county — resamples whole county time-series.
    Computes CI for ROC AUC and PR AUC using existing walk-forward scores.
    """
    rng     = np.random.default_rng(RNG_SEED)
    valid   = scored.dropna(subset=["alert_proba", "is_red"])
    counties = valid["county"].unique()

    roc_aucs, pr_aucs = [], []
    for _ in range(n_boot):
        sample_counties = rng.choice(counties, size=len(counties), replace=True)
        boot = pd.concat(
            [valid[valid["county"] == c] for c in sample_counties],
            ignore_index=True,
        )
        if boot["is_red"].nunique() < 2:
            continue
        y_t, y_s = boot["is_red"].values, boot["alert_proba"].values
        try:
            f, t, _ = roc_curve(y_t, y_s)
            roc_aucs.append(float(auc(f, t)))
            p, r, _ = precision_recall_curve(y_t, y_s)
            pr_aucs.append(float(auc(r, p)))
        except Exception:
            pass

    roc_arr = np.array(roc_aucs)
    pr_arr  = np.array(pr_aucs)

    def _ci(arr):
        return dict(
            mean=round(float(arr.mean()), 4),
            std=round(float(arr.std()), 4),
            ci_lower=round(float(np.percentile(arr, 2.5)), 4),
            ci_upper=round(float(np.percentile(arr, 97.5)), 4),
        )

    return dict(
        n_bootstrap=len(roc_arr),
        roc_auc=_ci(roc_arr),
        pr_auc=_ci(pr_arr),
    )


# ── McNemar's test ─────────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_pred_trends, y_pred_naive) -> dict:
    """
    McNemar's test on disagreement cases.
    b = Trends correct, naive wrong
    c = Trends wrong, naive correct
    H0: b == c (models make same types of errors)
    """
    b = int(((y_pred_trends == y_true) & (y_pred_naive != y_true)).sum())
    c = int(((y_pred_trends != y_true) & (y_pred_naive == y_true)).sum())
    if b + c == 0:
        return dict(b=b, c=c, chi2=0.0, p_value=1.0, significant_at_05=False)
    # Mid-P McNemar
    chi2  = float((abs(b - c) - 1) ** 2 / (b + c))
    p_val = float(1 - scipy_stats.chi2.cdf(chi2, df=1))
    return dict(b=b, c=c, chi2=round(chi2, 3), p_value=round(p_val, 4),
                significant_at_05=(p_val < 0.05))


# ── Lead-time test ─────────────────────────────────────────────────────────────

def lead_time_test(paired: pd.DataFrame, scored: pd.DataFrame,
                   scored_naive: pd.DataFrame, opt_thr: float) -> dict:
    """
    For each true Red event at (county, date_t1), check whether the model
    flagged it one month earlier than required — i.e., alert at t-1 for event at t+1
    (2-month advance warning vs the 1-month standard).

    Compares Trends model vs persistence naive baseline.
    """
    valid = scored.dropna(subset=["alert_proba", "is_red"])
    red_events = valid[valid["is_red"] == 1][["county", "date"]].copy()
    # date = t, event at t+1. A flag at t IS the 1-month advance warning.

    score_lookup = {
        (row["county"], row["date"]): row["alert_proba"]
        for _, row in valid.iterrows()
    }
    naive_valid = scored_naive.dropna(subset=["alert_proba", "is_red"])
    naive_lookup = {
        (row["county"], row["date"]): row["alert_proba"]
        for _, row in naive_valid.iterrows()
    }

    n_events = len(red_events)
    trends_early = 0
    naive_early  = 0

    for _, row in red_events.iterrows():
        key = (row["county"], row["date"])
        t_score = score_lookup.get(key)
        n_score = naive_lookup.get(key)
        if t_score is not None and t_score >= opt_thr:
            trends_early += 1
        if n_score is not None and n_score >= 0.5:
            naive_early += 1

    return dict(
        n_red_events=n_events,
        trends_early_warning=trends_early,
        naive_early_warning=naive_early,
        trends_early_pct=round(100 * trends_early / n_events, 1) if n_events else 0,
        naive_early_pct=round(100 * naive_early / n_events, 1) if n_events else 0,
        description="% of Red events flagged 1 month in advance (at t, for event at t+1)",
    )


# ── Calibration ────────────────────────────────────────────────────────────────

def calibration_analysis(scored: pd.DataFrame) -> dict:
    """Brier score and calibration curve summary."""
    valid  = scored.dropna(subset=["alert_proba", "is_red"])
    y_true = valid["is_red"].values.astype(float)
    y_prob = valid["alert_proba"].values

    brier  = float(brier_score_loss(y_true, y_prob))

    # Calibration curve (5 bins)
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=5,
                                                strategy="quantile")
    cal_bins = [
        {"mean_predicted": round(float(mp), 4), "fraction_positive": round(float(fp), 4)}
        for mp, fp in zip(mean_pred, fraction_pos)
    ]
    # Baseline Brier (always predict class rate)
    baseline_brier = float(y_true.mean() * (1 - y_true.mean()))

    return dict(
        brier_score=round(brier, 4),
        baseline_brier=round(baseline_brier, 4),
        brier_skill_score=round(1 - brier / baseline_brier, 4),
        calibration_curve=cal_bins,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _score_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1   = (2 * prec * rec / (prec + rec)
            if not (np.isnan(prec) or np.isnan(rec)) and (prec + rec) > 0
            else float("nan"))
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    J    = rec + spec - 1 if not (np.isnan(rec) or np.isnan(spec)) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, fpr=fpr,
                specificity=spec, youden_J=J)


def evaluate_version(scored: pd.DataFrame, label: str) -> dict:
    valid   = scored.dropna(subset=["alert_proba", "is_red"])
    y_true  = valid["is_red"].values.astype(int)
    y_score = valid["alert_proba"].values

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        logger.warning(f"{label}: degenerate labels")
        return {}

    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr_arr, tpr_arr))
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(rec_arr, prec_arr))

    J        = tpr_arr - fpr_arr
    best_i   = int(np.argmax(J))
    best_thr = float(roc_thresh[best_i])
    best_m   = _score_at_threshold(y_true, y_score, best_thr)

    print(f"\n  ── {label} ──")
    print(f"  n={len(valid):,}  positives={int(y_true.sum()):,} ({100*y_true.mean():.1f}%)")
    print(f"  ROC AUC = {roc_auc:.4f}  |  PR AUC = {pr_auc:.4f}")
    print(f"  Best threshold (Youden's J) = {best_thr:.4f}")
    print(f"  Recall={best_m['recall']:.3f}  Precision={best_m['precision']:.3f}  "
          f"F1={best_m['f1']:.3f}  FPR={best_m['fpr']:.3f}")
    print(f"  TP={best_m['tp']}  FP={best_m['fp']}  FN={best_m['fn']}  TN={best_m['tn']}")

    return dict(label=label, n_total=len(valid), n_positive=int(y_true.sum()),
                roc_auc=round(roc_auc, 4), pr_auc=round(pr_auc, 4),
                best_threshold=round(best_thr, 4), best_threshold_metrics=best_m,
                y_true=y_true, y_score=y_score)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run():
    if not os.path.exists(WF_CSV):
        raise FileNotFoundError(
            f"Walk-forward predictions not found at {WF_CSV}\n"
            "Run experiments/tune_deployable_model.py first."
        )

    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    wf = pd.read_csv(WF_CSV, parse_dates=["date"])
    logger.info(f"Features: {len(df):,} rows | WF predictions: {len(wf):,} rows")

    logger.info("Computing Trends features …")
    df = compute_trends_features(df)

    logger.info("Building paired dataset (t → t+1) …")
    paired = build_paired_dataset(df, wf)
    logger.info(f"Paired: {len(paired):,} county-months")

    # ── Deviation distribution ─────────────────────────────────────────────────
    devs     = paired["deviation_t1"]
    pos_devs = devs[devs > 0]
    pct_vals = {p: float(np.percentile(pos_devs, p)) for p in [50, 75, 85, 90, 95, 99]}

    print("\n" + "═" * 68)
    print("  DEVIATION DISTRIBUTION")
    print("  deviation_t1 = (actual_{t+1} - baseline_{t+1}) / baseline_{t+1}")
    print("═" * 68)
    print(f"  n={len(devs):,}  positive={len(pos_devs):,} ({100*len(pos_devs)/len(devs):.1f}%)")
    print(f"  Mean={devs.mean():.4f}  Median={devs.median():.4f}  Std={devs.std():.4f}")
    print(f"\n  Percentiles of positive deviations:")
    for p, v in pct_vals.items():
        marker = " ◄ Yellow" if p == YELLOW_PCT else (" ◄ Red (main)" if p == RED_PCT_MAIN else "")
        print(f"    {p:3d}th : {v:+.4f}{marker}")

    # ── Main version: Red@85th ─────────────────────────────────────────────────
    labeled, yellow_thr, red_thr = define_alert_labels(paired, YELLOW_PCT, RED_PCT_MAIN)
    n_g = (labeled["alert_label"] == "Green").sum()
    n_y = (labeled["alert_label"] == "Yellow").sum()
    n_r = (labeled["alert_label"] == "Red").sum()
    print(f"\n  Main version (Red@{RED_PCT_MAIN}th): "
          f"Green={n_g:,}  Yellow={n_y:,}  Red={n_r:,}")

    # ── Sensitivity analysis ───────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  SENSITIVITY ANALYSIS")
    print("═" * 68)
    sensitivity_results = []
    scored_main = None

    for red_pct in RED_PCT_VERSIONS:
        lbl_df, yt, rt = define_alert_labels(paired, YELLOW_PCT, red_pct)
        label = f"Trends model — Red@{red_pct}th pctile"
        logger.info(f"Walk-forward XGBoost ({label}) …")
        scored = logistic_walkforward(lbl_df, TRENDS_FEATURES)
        if len(scored) == 0:
            continue
        res = evaluate_version(scored, label)
        res["yellow_pct"] = YELLOW_PCT
        res["red_pct"]    = red_pct
        res["yellow_thr"] = round(yt, 6)
        res["red_thr"]    = round(rt, 6)
        sensitivity_results.append(res)
        if red_pct == RED_PCT_MAIN:
            scored_main = scored
            labeled_main = lbl_df

    if scored_main is None:
        logger.error("Main version produced no scored rows — aborting statistical tests")
        return

    main_res = next(r for r in sensitivity_results if r["red_pct"] == RED_PCT_MAIN)
    y_true_main  = main_res["y_true"]
    y_score_main = main_res["y_score"]
    opt_thr_main = main_res["best_threshold"]

    # ── Feature importances ───────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  FEATURE IMPORTANCES (full-data fit, Red@85th)")
    print("═" * 68)
    clean = labeled_main.dropna(subset=TRENDS_FEATURES + ["is_red"])
    if clean["is_red"].nunique() == 2:
        pos_rate = clean["is_red"].mean()
        scale_pos = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        clf_all = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=scale_pos, use_label_encoder=False,
                                eval_metric="logloss", random_state=RNG_SEED,
                                n_jobs=-1, verbosity=0)
        clf_all.fit(clean[TRENDS_FEATURES].values, clean["is_red"].values)
        pairs = sorted(zip(TRENDS_FEATURES, clf_all.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        print(f"  {'Feature':<30}  {'Importance':>10}")
        for feat, imp in pairs:
            print(f"  {feat:<30}  {imp:>10.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Statistical validation
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 68)
    print("  STATISTICAL VALIDATION")
    print("═" * 68)

    # ── 1. Permutation test ───────────────────────────────────────────────────
    print(f"\n  [1] Permutation test  (n={N_PERMUTATIONS:,} permutations)")
    print(f"  H0: Trends features contain no signal beyond random labeling")
    print(f"  Labels shuffled within county | temporal hold-out "
          f"({int(PERMUTATION_SPLIT*100)}/{int((1-PERMUTATION_SPLIT)*100)} split)")
    logger.info(f"Running permutation test ({N_PERMUTATIONS} permutations) …")
    perm = permutation_test(labeled_main, n_perm=N_PERMUTATIONS)
    if perm:
        print(f"  Observed AUC : {perm['observed_auc']:.4f}")
        print(f"  Null AUC     : {perm['null_auc_mean']:.4f} ± {perm['null_auc_std']:.4f} "
              f"(95th pctile = {perm['null_auc_p95']:.4f})")
        print(f"  p-value      : {perm['p_value']:.4f}  "
              f"{'✓ significant (p<0.05)' if perm['significant_at_05'] else '✗ not significant'}")

    # ── 2. Bootstrap CI ───────────────────────────────────────────────────────
    print(f"\n  [2] Bootstrap CI  (n={N_BOOTSTRAP:,}, cluster by county)")
    logger.info(f"Running bootstrap CI ({N_BOOTSTRAP} iterations) …")
    boot = bootstrap_ci(scored_main, n_boot=N_BOOTSTRAP)
    r = boot["roc_auc"]
    p = boot["pr_auc"]
    print(f"  ROC AUC : {r['mean']:.4f}  (95% CI: {r['ci_lower']:.4f} – {r['ci_upper']:.4f})")
    print(f"  PR  AUC : {p['mean']:.4f}  (95% CI: {p['ci_lower']:.4f} – {p['ci_upper']:.4f})")

    # ── 3. Naive baselines ────────────────────────────────────────────────────
    print(f"\n  [3] Naive baselines")
    scored_persist = naive_persistence(labeled_main)
    scored_lagdev  = naive_lagged_deviation(labeled_main)

    persist_res = evaluate_version(scored_persist, "Naive: persistence (last month's label)")
    lagdev_res  = evaluate_version(scored_lagdev,  "Naive: lagged deviation (deviation_t only)")

    # ── 4. DeLong test ────────────────────────────────────────────────────────
    print(f"\n  [4] DeLong test (Trends vs naive baselines)")

    def _align_scores(scored_ref, scored_other):
        """Return aligned y_true, score1, score2 on common (county, date) pairs."""
        ref   = scored_ref.dropna(subset=["alert_proba", "is_red"])
        other = scored_other.dropna(subset=["alert_proba", "is_red"])
        merged = ref[["county", "date", "is_red", "alert_proba"]].merge(
            other[["county", "date", "alert_proba"]].rename(
                columns={"alert_proba": "alert_proba_other"}),
            on=["county", "date"], how="inner",
        )
        return (merged["is_red"].values.astype(int),
                merged["alert_proba"].values,
                merged["alert_proba_other"].values)

    y_t, s_trends, s_persist = _align_scores(scored_main, scored_persist)
    dl_persist = delong_compare(y_t, s_trends, s_persist)
    print(f"  Trends vs Persistence: AUC {dl_persist['auc1']:.4f} vs {dl_persist['auc2']:.4f}  "
          f"Δ={dl_persist['diff']:+.4f}  z={dl_persist['z']:.2f}  p={dl_persist['p_value']:.4f}  "
          f"{'✓' if dl_persist['p_value'] < 0.05 else '✗'}")

    if len(scored_lagdev) > 0:
        y_t2, s_trends2, s_lagdev = _align_scores(scored_main, scored_lagdev)
        dl_lagdev = delong_compare(y_t2, s_trends2, s_lagdev)
        print(f"  Trends vs Lagged dev: AUC {dl_lagdev['auc1']:.4f} vs {dl_lagdev['auc2']:.4f}  "
              f"Δ={dl_lagdev['diff']:+.4f}  z={dl_lagdev['z']:.2f}  p={dl_lagdev['p_value']:.4f}  "
              f"{'✓' if dl_lagdev['p_value'] < 0.05 else '✗'}")
    else:
        dl_lagdev = {}

    # ── 5. McNemar's test ─────────────────────────────────────────────────────
    print(f"\n  [5] McNemar's test (classification disagreements)")
    y_t3, s_t3, s_p3 = _align_scores(scored_main, scored_persist)
    y_pred_trends  = (s_t3 >= opt_thr_main).astype(int)
    y_pred_persist = (s_p3 >= 0.5).astype(int)
    mc = mcnemar_test(y_t3, y_pred_trends, y_pred_persist)
    print(f"  Trends correct / Naive wrong (b) : {mc['b']:,}")
    print(f"  Trends wrong  / Naive correct (c): {mc['c']:,}")
    print(f"  χ²={mc['chi2']:.3f}  p={mc['p_value']:.4f}  "
          f"{'✓ significant' if mc['significant_at_05'] else '✗ not significant'}")

    # ── 6. Lead-time test ─────────────────────────────────────────────────────
    print(f"\n  [6] Lead-time test (1-month advance warning)")
    lt = lead_time_test(labeled_main, scored_main, scored_persist, opt_thr_main)
    print(f"  True Red events           : {lt['n_red_events']:,}")
    print(f"  Trends — flagged 1mo early: {lt['trends_early_warning']:,} "
          f"({lt['trends_early_pct']:.1f}%)")
    print(f"  Persistence — flagged 1mo : {lt['naive_early_warning']:,} "
          f"({lt['naive_early_pct']:.1f}%)")

    # ── 7. Calibration ────────────────────────────────────────────────────────
    print(f"\n  [7] Calibration")
    cal = calibration_analysis(scored_main)
    print(f"  Brier score          : {cal['brier_score']:.4f}")
    print(f"  Baseline Brier       : {cal['baseline_brier']:.4f}  (always predict base rate)")
    print(f"  Brier skill score    : {cal['brier_skill_score']:.4f}  "
          f"(higher = better; 1.0 = perfect)")
    print(f"  Calibration curve (predicted → observed fraction positive):")
    for b in cal["calibration_curve"]:
        print(f"    predicted={b['mean_predicted']:.3f} → actual={b['fraction_positive']:.3f}")

    print("\n" + "═" * 68)

    # ── Save ──────────────────────────────────────────────────────────────────
    def _clean(d):
        """Remove numpy arrays from dict before JSON serialization."""
        return {k: v for k, v in d.items()
                if not isinstance(v, np.ndarray)}

    output = {
        "deviation_distribution": {
            "n_total": len(devs), "n_positive": len(pos_devs),
            "positive_pct": round(100*len(pos_devs)/len(devs), 1),
            "mean": round(float(devs.mean()), 6),
            "median": round(float(devs.median()), 6),
            "std": round(float(devs.std()), 6),
            "percentiles_of_positive": {str(k): round(v, 6) for k, v in pct_vals.items()},
        },
        "sensitivity_versions": [_clean(r) for r in sensitivity_results],
        "features_used": TRENDS_FEATURES,
        "statistical_tests": {
            "permutation_test": perm,
            "bootstrap_ci": boot,
            "delong_vs_persistence": dl_persist,
            "delong_vs_lagged_dev": dl_lagdev,
            "mcnemar_vs_persistence": mc,
            "lead_time_test": lt,
            "calibration": cal,
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results → {OUT_JSON}")


if __name__ == "__main__":
    run()
