"""
Individual Search Term Analysis

Tests each of the 10 Google Trends search terms individually to find:
1. Which terms have the strongest predictive signal for SNAP applications
2. Why combining all 8 AllTerms hurts performance
3. The optimal subset of terms

For each term, runs walk-forward validation with:
  base features (Population, Median_Income, month) + that single term

Then tests the top-N combinations to find the best pair/triplet.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ───────────────────────────────────────────────────────────────────

ALLTERMS_DIR = "src/data/trends/AllTerms"
TRAINING_DATA = "src/data/aggregateTrends_scaled.csv"

FILENAME_TO_METRO = {
    'Bakersfield': 'Bakersfield',
    'ChicoRedding': 'ChicoRedding',
    'Eureka': 'Eureka',
    'FresnoVisalia': 'FresnoVisalia',
    'LosAngeles': 'LosAngeles',
    'MedfordKlamathFalls': 'MedfordKlamathFalls',
    'MontereySalinas': 'MontereySalinas',
    'PalmSprings': 'PalmSprings',
    'Reno': 'Reno',
    'SacramentoStocktonModesto': 'SacramentoStocktonModesto',
    'San Diego': 'SanDiego',
    'SanFranciscoOaklandSanJose': 'SanFranciscoOaklandSanJose',
    'SantaBarbaraSantaMariaSanLuisObispo': 'SantaBarbaraSantaMariaSanLuisObispo',
    'YumaElCentro': 'Yuma',
}

RAW_TERM_COLUMNS = [
    'food stamps', 'EBT card', 'SNAP benefits', 'food pantry near me',
    'food bank', 'CalFresh', 'apply for CalFresh',
    'Supplemental Nutrition Assistance Program',
]

BASE_FEATURES = ['Population', 'Median_Income', 'month']
ORIGINAL_TRENDS = ['monthly_average_CalFresh', 'monthly_average_FoodBank']


def safe_col(term):
    return 'trend_' + term.replace(' ', '_').lower()


TERM_FEATURE_COLS = [safe_col(t) for t in RAW_TERM_COLUMNS]
ALL_TERMS = dict(zip(RAW_TERM_COLUMNS, TERM_FEATURE_COLS))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_allterms():
    dfs = []
    for filename in sorted(os.listdir(ALLTERMS_DIR)):
        if not filename.endswith('.csv'):
            continue
        name = filename.replace('.csv', '')
        metro = FILENAME_TO_METRO.get(name)
        if metro is None:
            continue
        filepath = os.path.join(ALLTERMS_DIR, filename)
        df = pd.read_csv(filepath)
        df = df.rename(columns={'Time': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df['metro_area'] = metro
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_merged_data():
    df = pd.read_csv(TRAINING_DATA)
    df['date'] = pd.to_datetime(df['date'])

    allterms = load_allterms()
    merged = df.merge(allterms, on=['metro_area', 'date'], how='inner')

    # Normalize each new term by county population
    for raw_term in RAW_TERM_COLUMNS:
        col = safe_col(raw_term)
        merged[col] = merged[raw_term] / merged['Population']

    # Target: next month's SNAP rate
    merged = merged.sort_values(['county', 'date']).reset_index(drop=True)
    merged['SNAP_target'] = merged.groupby('county')['SNAP_Application_Rate'].shift(-1)
    merged['target_month'] = merged['date'] + pd.DateOffset(months=1)
    merged = merged.dropna(subset=['SNAP_target', 'target_month'])

    return merged


# ── Model & metrics ──────────────────────────────────────────────────────────

def make_model():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.01,
        min_child_weight=6, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=4, reg_alpha=0, random_state=42, n_jobs=-1,
    )


def calc_metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'smape': np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        'r2': r2_score(yt, yp),
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'mae': mean_absolute_error(yt, yp),
        'smape': np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100,
    }


def run_walk_forward(df, feature_cols):
    unique_months = sorted(df['target_month'].unique())
    n = len(unique_months)
    start = 3

    all_pred, all_actual = [], []

    for i in range(start, n):
        T = unique_months[i]
        train_df = df[df['target_month'] < T]
        test_df = df[df['target_month'] == T]

        X_train = train_df[feature_cols]
        y_train = train_df['SNAP_target']
        X_test = test_df[feature_cols]
        y_test = test_df['SNAP_target']

        trn_ok = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        tst_ok = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[trn_ok], y_train[trn_ok]
        X_test, y_test = X_test[tst_ok], y_test[tst_ok]

        if len(X_train) < 30 or len(X_test) < 3:
            continue

        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_pred.extend(y_pred)
        all_actual.extend(y_test.values)

    if not all_pred:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'smape': np.nan}, 0

    overall = calc_metrics(np.array(all_actual), np.array(all_pred))
    return overall, len(all_pred)


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_term_coverage(df):
    """Check how many metros each term has non-zero data in."""
    print("\n" + "=" * 80)
    print("TERM COVERAGE ANALYSIS (non-zero values per metro)")
    print("=" * 80)

    coverage = []
    for raw_term in RAW_TERM_COLUMNS:
        col = safe_col(raw_term)
        for metro in sorted(df['metro_area'].unique()):
            metro_data = df[df['metro_area'] == metro][col]
            total = len(metro_data)
            nonzero = (metro_data > 0).sum()
            pct = nonzero / total * 100 if total > 0 else 0
            coverage.append({
                'term': raw_term,
                'metro': metro,
                'total_rows': total,
                'nonzero_rows': nonzero,
                'pct_nonzero': pct,
            })

    cov_df = pd.DataFrame(coverage)

    # Summary per term
    print(f"\n{'Term':<45s} {'Metros w/ data':>15s} {'Avg % nonzero':>15s}")
    print("-" * 80)
    for raw_term in RAW_TERM_COLUMNS:
        term_data = cov_df[cov_df['term'] == raw_term]
        metros_with_data = (term_data['pct_nonzero'] > 5).sum()
        avg_pct = term_data['pct_nonzero'].mean()
        print(f"  {raw_term:<43s} {metros_with_data:>10d}/14   {avg_pct:>12.1f}%")

    # Also show original terms
    for orig_term in ORIGINAL_TRENDS:
        if orig_term in df.columns:
            nonzero_metros = 0
            for metro in df['metro_area'].unique():
                metro_data = df[df['metro_area'] == metro][orig_term]
                if (metro_data > 0).sum() / len(metro_data) > 0.05:
                    nonzero_metros += 1
            pct_all = (df[orig_term] > 0).sum() / len(df) * 100
            print(f"  {orig_term:<43s} {nonzero_metros:>10d}/14   {pct_all:>12.1f}%")

    return cov_df


def analyze_correlations(df):
    """Check raw correlation between each term and SNAP target."""
    print("\n" + "=" * 80)
    print("CORRELATION WITH SNAP TARGET (per-capita trend vs next-month SNAP rate)")
    print("=" * 80)

    corrs = []

    # AllTerms
    for raw_term in RAW_TERM_COLUMNS:
        col = safe_col(raw_term)
        if col in df.columns:
            r = df[col].corr(df['SNAP_target'])
            corrs.append({'term': raw_term, 'col': col, 'correlation': r, 'source': 'AllTerms'})

    # Original terms
    for orig in ORIGINAL_TRENDS:
        if orig in df.columns:
            r = df[orig].corr(df['SNAP_target'])
            corrs.append({'term': orig, 'col': orig, 'correlation': r, 'source': 'Original'})

    corrs_df = pd.DataFrame(corrs).sort_values('correlation', key=abs, ascending=False)

    print(f"\n{'Term':<50s} {'Correlation':>12s}  {'Source':<10s}")
    print("-" * 80)
    for _, row in corrs_df.iterrows():
        print(f"  {row['term']:<48s} {row['correlation']:>+10.4f}    {row['source']:<10s}")

    return corrs_df


def main():
    print("=" * 80)
    print("INDIVIDUAL SEARCH TERM ANALYSIS")
    print("=" * 80)

    df = load_merged_data()
    print(f"\nData: {len(df)} rows, {df['county'].nunique()} counties, {df['metro_area'].nunique()} metros")

    # ── Part 1: Coverage analysis ──
    coverage_df = analyze_term_coverage(df)

    # ── Part 2: Correlation analysis ──
    corrs_df = analyze_correlations(df)

    # ── Part 3: Baseline (no trends) ──
    print("\n" + "=" * 80)
    print("PART 3: BASELINE — No trend features")
    print("=" * 80)

    baseline_metrics, baseline_n = run_walk_forward(df, BASE_FEATURES)
    print(f"  Base only (Pop + Income + Month): R²={baseline_metrics['r2']:.4f}, MAE={baseline_metrics['mae']:.6f}, sMAPE={baseline_metrics['smape']:.2f}%")

    # ── Part 4: Each term individually ──
    print("\n" + "=" * 80)
    print("PART 4: INDIVIDUAL TERM PERFORMANCE")
    print("  Each term tested alone: base features + single term")
    print("=" * 80)

    individual_results = []

    # Test original 2 terms (each alone)
    for orig in ORIGINAL_TRENDS:
        features = BASE_FEATURES + [orig]
        metrics, n_pred = run_walk_forward(df, features)
        r = {
            'term': orig,
            'source': 'Original',
            'features': len(features),
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'smape': metrics['smape'],
            'n_predictions': n_pred,
            'r2_lift_vs_base': metrics['r2'] - baseline_metrics['r2'],
            'mae_improvement': baseline_metrics['mae'] - metrics['mae'],
        }
        individual_results.append(r)
        print(f"  {orig:<48s} R²={metrics['r2']:.4f} (lift: {r['r2_lift_vs_base']:+.4f})  MAE={metrics['mae']:.6f}")

    # Test each AllTerms term alone
    for raw_term in RAW_TERM_COLUMNS:
        col = safe_col(raw_term)
        features = BASE_FEATURES + [col]
        metrics, n_pred = run_walk_forward(df, features)
        r = {
            'term': raw_term,
            'source': 'AllTerms',
            'features': len(features),
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'smape': metrics['smape'],
            'n_predictions': n_pred,
            'r2_lift_vs_base': metrics['r2'] - baseline_metrics['r2'],
            'mae_improvement': baseline_metrics['mae'] - metrics['mae'],
        }
        individual_results.append(r)
        print(f"  {raw_term:<48s} R²={metrics['r2']:.4f} (lift: {r['r2_lift_vs_base']:+.4f})  MAE={metrics['mae']:.6f}")

    ind_df = pd.DataFrame(individual_results).sort_values('r2', ascending=False)

    # ── Part 5: Best pairs from top terms ──
    print("\n" + "=" * 80)
    print("PART 5: BEST PAIRS — Top terms combined")
    print("=" * 80)

    # Take top 5 terms by R² lift
    top_terms = ind_df.nlargest(5, 'r2')
    top_cols = [safe_col(t) if t in RAW_TERM_COLUMNS else t for t in top_terms['term']]

    pair_results = []
    for t1, t2 in combinations(top_cols, 2):
        features = BASE_FEATURES + [t1, t2]
        metrics, n_pred = run_walk_forward(df, features)
        name = f"{t1} + {t2}"
        pair_results.append({
            'pair': name,
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'smape': metrics['smape'],
        })
        print(f"  {name:<70s} R²={metrics['r2']:.4f}  MAE={metrics['mae']:.6f}")

    pair_df = pd.DataFrame(pair_results).sort_values('r2', ascending=False)

    # ── Part 6: Best triplet ──
    print("\n" + "=" * 80)
    print("PART 6: BEST TRIPLETS — Top 3 terms combined")
    print("=" * 80)

    top3_cols = top_cols[:4]  # take top 4, test all triplets
    triplet_results = []
    for combo in combinations(top3_cols, 3):
        features = BASE_FEATURES + list(combo)
        metrics, n_pred = run_walk_forward(df, features)
        name = " + ".join(combo)
        triplet_results.append({
            'triplet': name,
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'smape': metrics['smape'],
        })
        print(f"  {name:<70s} R²={metrics['r2']:.4f}  MAE={metrics['mae']:.6f}")

    # ── Save results ──
    os.makedirs('artifacts/experiments', exist_ok=True)

    ind_df.to_csv('artifacts/experiments/individual_terms_results.csv', index=False)
    if pair_results:
        pair_df.to_csv('artifacts/experiments/term_pairs_results.csv', index=False)
    corrs_df.to_csv('artifacts/experiments/term_correlations.csv', index=False)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBaseline (no trends): R²={baseline_metrics['r2']:.4f}, MAE={baseline_metrics['mae']:.6f}")
    print(f"\nIndividual term rankings (by R² lift over baseline):")
    print(f"{'Rank':<6s} {'Term':<48s} {'R²':>8s}  {'Lift':>8s}  {'Source':<10s}")
    print("-" * 85)
    for i, (_, row) in enumerate(ind_df.iterrows(), 1):
        print(f"  {i:<4d} {row['term']:<48s} {row['r2']:>8.4f}  {row['r2_lift_vs_base']:>+7.4f}  {row['source']:<10s}")

    if pair_results:
        best_pair = pair_df.iloc[0]
        print(f"\nBest pair: {best_pair['pair']}")
        print(f"  R²={best_pair['r2']:.4f}, MAE={best_pair['mae']:.6f}")

    print("\nResults saved to artifacts/experiments/")


if __name__ == '__main__':
    main()
