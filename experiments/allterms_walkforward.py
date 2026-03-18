"""
Walk-Forward Backtest — ALL TERMS EXPERIMENT

Tests whether adding more Google Trends search terms improves prediction.
The AllTerms folder has 8 terms vs the original 2 (CalFresh, FoodBank).

Compares three feature sets (all using XGBoost tuned):
  A) Original 2 terms only (baseline)
  B) All 8 new terms (replacing the original 2)
  C) All 8 new terms + original 2 terms (combined)

NOTE: AllTerms data starts 2023-05, so we restrict to that overlap period
for a fair comparison across all feature sets.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ───────────────────────────────────────────────────────────────────

ALLTERMS_DIR = "src/data/trends/AllTerms"
TRAINING_DATA = "src/data/aggregateTrends_scaled.csv"

# Map AllTerms CSV filenames (without .csv) to metro_area codes in training data
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


def safe_col(term):
    """Convert a raw term name to a safe column name."""
    return 'trend_' + term.replace(' ', '_').lower()


TERM_FEATURE_COLS = [safe_col(t) for t in RAW_TERM_COLUMNS]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_allterms():
    """Load all AllTerms CSVs into one DataFrame with metro_area column."""
    dfs = []
    for filename in sorted(os.listdir(ALLTERMS_DIR)):
        if not filename.endswith('.csv'):
            continue
        name = filename.replace('.csv', '')
        metro = FILENAME_TO_METRO.get(name)
        if metro is None:
            print(f"  Warning: No metro mapping for '{name}', skipping")
            continue

        filepath = os.path.join(ALLTERMS_DIR, filename)
        df = pd.read_csv(filepath)
        df = df.rename(columns={'Time': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df['metro_area'] = metro
        dfs.append(df)

    allterms = pd.concat(dfs, ignore_index=True)
    print(f"AllTerms loaded: {len(allterms)} rows, {allterms['metro_area'].nunique()} metros")
    print(f"  Date range: {allterms['date'].min().date()} to {allterms['date'].max().date()}")
    print(f"  Terms: {RAW_TERM_COLUMNS}")
    return allterms


def load_merged_data():
    """Load training data, merge with AllTerms, normalize by population."""
    # Load base training data
    df = pd.read_csv(TRAINING_DATA)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Training data: {len(df)} rows, {df['county'].nunique()} counties")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Load AllTerms
    allterms = load_allterms()

    # Merge on (metro_area, date) — inner join keeps only overlapping rows
    merged = df.merge(allterms, on=['metro_area', 'date'], how='inner')
    print(f"\nAfter inner join: {len(merged)} rows")
    print(f"  Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"  Counties: {merged['county'].nunique()}")
    # metro_area columns may duplicate from the merge; use the training one
    metro_col = 'metro_area'
    print(f"  Metros: {merged[metro_col].nunique()}")
    print(f"  Metro areas matched: {sorted(merged[metro_col].unique())}")

    # Normalize each new term by county population (same as existing trends)
    for raw_term in RAW_TERM_COLUMNS:
        col = safe_col(raw_term)
        merged[col] = merged[raw_term] / merged['Population']

    # Set up target: next month's SNAP rate
    merged = merged.sort_values(['county', 'date']).reset_index(drop=True)
    merged['SNAP_target'] = merged.groupby('county')['SNAP_Application_Rate'].shift(-1)
    merged['target_month'] = merged['date'] + pd.DateOffset(months=1)
    merged = merged.dropna(subset=['SNAP_target', 'target_month'])

    print(f"  After target alignment: {len(merged)} rows")
    return merged


# ── Feature sets ─────────────────────────────────────────────────────────────

def get_feature_sets():
    """Define the three feature sets to compare."""
    base = ['Population', 'Median_Income', 'month']
    original_trends = ['monthly_average_CalFresh', 'monthly_average_FoodBank']

    return {
        'A) Original 2 terms': base + original_trends,
        'B) All 8 new terms': base + TERM_FEATURE_COLS,
        'C) All 8 + original 2': base + original_trends + TERM_FEATURE_COLS,
    }


# ── Model ────────────────────────────────────────────────────────────────────

def make_xgb_tuned():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.01,
        min_child_weight=6, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=4, reg_alpha=0,
        random_state=42, n_jobs=-1,
    )


# ── Metrics ──────────────────────────────────────────────────────────────────

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


# ── Walk-forward loop ───────────────────────────────────────────────────────

def run_walk_forward(df, feature_cols, label):
    """Walk-forward backtest for XGBoost tuned with a given feature set."""
    unique_months = sorted(df['target_month'].unique())
    n = len(unique_months)
    start = 3  # AllTerms data is short, so use a smaller warm-up

    all_pred, all_actual = [], []
    per_month = []

    for i in range(start, n):
        T = unique_months[i]
        train_df = df[df['target_month'] < T]
        test_df = df[df['target_month'] == T]

        X_train = train_df[feature_cols]
        y_train = train_df['SNAP_target']
        X_test = test_df[feature_cols]
        y_test = test_df['SNAP_target']

        # Clean NaN / inf
        trn_ok = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        tst_ok = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[trn_ok], y_train[trn_ok]
        X_test, y_test = X_test[tst_ok], y_test[tst_ok]

        if len(X_train) < 30 or len(X_test) < 3:
            continue

        model = make_xgb_tuned()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = calc_metrics(y_test.values, y_pred)
        per_month.append({
            'month': pd.Timestamp(T).strftime('%Y-%m'),
            'train_size': len(X_train),
            'test_size': len(X_test),
            **m,
        })
        all_pred.extend(y_pred)
        all_actual.extend(y_test.values)

    if not all_pred:
        print(f"  WARNING: No predictions produced for '{label}'")
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'smape': np.nan}, pd.DataFrame(), {}

    overall = calc_metrics(np.array(all_actual), np.array(all_pred))
    per_month_df = pd.DataFrame(per_month)

    # Feature importance from last trained model
    importance = dict(zip(feature_cols, model.feature_importances_))

    return overall, per_month_df, importance


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_merged_data()
    feature_sets = get_feature_sets()

    print("\n" + "=" * 80)
    print("ALL-TERMS EXPERIMENT: Walk-Forward Backtest (XGBoost Tuned)")
    print("=" * 80)

    results = []
    all_importances = {}

    for fs_name, features in feature_sets.items():
        # Verify features exist
        available = [f for f in features if f in df.columns]
        missing = set(features) - set(available)
        if missing:
            print(f"\n  Warning: Missing features for '{fs_name}': {missing}")

        # Drop rows with NaN in these features
        df_clean = df.dropna(subset=available + ['SNAP_target'])

        print(f"\n{'─'*60}")
        print(f"  {fs_name}")
        print(f"  Features ({len(available)}): {available}")
        print(f"  Rows: {len(df_clean)}")
        print(f"{'─'*60}")

        overall, per_month_df, importance = run_walk_forward(df_clean, available, fs_name)

        results.append({
            'Feature_Set': fs_name,
            'Num_Features': len(available),
            'R2': round(overall['r2'], 4),
            'RMSE': round(overall['rmse'], 6),
            'MAE': round(overall['mae'], 6),
            'sMAPE': round(overall['smape'], 2),
            'Months_Tested': len(per_month_df),
            'Total_Predictions': int(per_month_df['test_size'].sum()) if not per_month_df.empty else 0,
        })

        all_importances[fs_name] = importance

        print(f"  R²:    {overall['r2']:.4f}")
        print(f"  RMSE:  {overall['rmse']:.6f}")
        print(f"  MAE:   {overall['mae']:.6f}")
        print(f"  sMAPE: {overall['smape']:.2f}%")

        if importance:
            print(f"\n  Feature importance (top 10):")
            for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
                print(f"    {feat:45s} {imp:.4f}")

    # ── Save results ─────────────────────────────────────────────────────

    os.makedirs('artifacts/experiments', exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv('artifacts/experiments/allterms_experiment_results.csv', index=False)

    # Convert float32 to native float for JSON serialization
    clean_importances = {
        k: {fk: float(fv) for fk, fv in v.items()}
        for k, v in all_importances.items()
    }
    with open('artifacts/experiments/allterms_feature_importance.json', 'w') as f:
        json.dump(clean_importances, f, indent=2)

    # ── Print comparison ─────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\nResults saved to artifacts/experiments/allterms_experiment_results.csv")


if __name__ == '__main__':
    main()
