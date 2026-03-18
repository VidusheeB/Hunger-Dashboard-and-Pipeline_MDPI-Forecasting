"""
Walk-Forward Backtest — PRODUCTION-REALISTIC

This uses ONLY the 5 features available at prediction time:
  Population, Median_Income, CalFresh_trend, FoodBank_trend, month

No SNAP lag features, no rolling means — because real-world predictions
don't have access to recent SNAP data (that's the whole point of the tool).

Tests: Random Forest, Gradient Boosting, XGBoost (default), XGBoost (tuned)
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data():
    """Load scaled training data and set up target alignment."""
    df = pd.read_csv("src/data/aggregateTrends_scaled.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['county', 'date']).reset_index(drop=True)

    # Target: next month's SNAP rate for this county
    df['SNAP_target'] = df.groupby('county')['SNAP_Application_Rate'].shift(-1)
    df['target_month'] = df['date'] + pd.DateOffset(months=1)

    # Drop rows with no target (last month per county)
    df = df.dropna(subset=['SNAP_target', 'target_month'])

    print(f"Loaded {len(df)} rows, {df['county'].nunique()} counties")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ── Feature sets ──────────────────────────────────────────────────────────────

PRODUCTION_FEATURES = [
    'Population',
    'Median_Income',
    'monthly_average_CalFresh',
    'monthly_average_FoodBank',
    'month',
]


# ── Models ────────────────────────────────────────────────────────────────────

def get_models():
    """Return dict of model_name → sklearn/xgb estimator."""
    return {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        'XGBoost (default)': xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1,
        ),
        'XGBoost (tuned)': xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.01,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.9,
            reg_lambda=4, reg_alpha=0,
            random_state=42, n_jobs=-1,
        ),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

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


# ── Walk-forward loop ────────────────────────────────────────────────────────

def run_walk_forward(df, feature_cols, model_factory, model_name):
    """Walk-forward backtest for a single model."""
    unique_months = sorted(df['target_month'].unique())
    n = len(unique_months)
    start = 5  # need some history before first prediction

    all_pred, all_actual = [], []
    per_month = []

    for i in range(start, n):
        T = unique_months[i]

        X_train = df[df['target_month'] < T][feature_cols]
        y_train = df[df['target_month'] < T]['SNAP_target']
        X_test = df[df['target_month'] == T][feature_cols]
        y_test = df[df['target_month'] == T]['SNAP_target']

        # Clean NaN / inf
        trn_ok = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        tst_ok = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[trn_ok], y_train[trn_ok]
        X_test, y_test = X_test[tst_ok], y_test[tst_ok]

        if len(X_train) < 50 or len(X_test) < 3:
            continue

        model = model_factory()
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

    overall = calc_metrics(np.array(all_actual), np.array(all_pred))
    per_month_df = pd.DataFrame(per_month)

    return overall, per_month_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    # Verify features exist
    missing = [f for f in PRODUCTION_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop rows with NaN in any production feature
    df = df.dropna(subset=PRODUCTION_FEATURES + ['SNAP_target'])

    print(f"\nAfter cleaning: {len(df)} rows")
    print(f"Features: {PRODUCTION_FEATURES}")
    print(f"Target: SNAP_target (next month's SNAP_Application_Rate)")

    models = get_models()
    results = {}

    print("\n" + "=" * 80)
    print("PRODUCTION-REALISTIC WALK-FORWARD BACKTEST")
    print("Features: ONLY the 5 available at prediction time (no SNAP lags)")
    print("=" * 80)

    for name, _ in models.items():
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")

        overall, per_month_df = run_walk_forward(
            df, PRODUCTION_FEATURES,
            model_factory=lambda n=name: get_models()[n],
            model_name=name,
        )

        results[name] = {
            'overall': overall,
            'per_month': per_month_df,
        }

        print(f"  Overall R²:    {overall['r2']:.4f}")
        print(f"  Overall RMSE:  {overall['rmse']:.6f}")
        print(f"  Overall MAE:   {overall['mae']:.6f}")
        print(f"  Overall sMAPE: {overall['smape']:.2f}%")
        print(f"  Months tested: {len(per_month_df)}")
        print(f"  Total predictions: {per_month_df['test_size'].sum()}")

        pos_r2 = (per_month_df['r2'] >= 0).sum()
        print(f"  Months with R² >= 0: {pos_r2}/{len(per_month_df)}")

    # ── Save results ──────────────────────────────────────────────────────

    os.makedirs('artifacts/experiments', exist_ok=True)

    # Summary comparison table
    comparison = []
    for name, res in results.items():
        o = res['overall']
        pm = res['per_month']
        comparison.append({
            'Model': name,
            'Overall_R2': round(o['r2'], 4),
            'Overall_RMSE': round(o['rmse'], 6),
            'Overall_MAE': round(o['mae'], 6),
            'Overall_sMAPE': round(o['smape'], 2),
            'R2_Mean': round(pm['r2'].mean(), 4),
            'R2_Std': round(pm['r2'].std(), 4),
            'MAE_Mean': round(pm['mae'].mean(), 6),
            'Months_Tested': len(pm),
            'Total_Predictions': int(pm['test_size'].sum()),
        })

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv('artifacts/experiments/production_walkforward_comparison.csv', index=False)

    # Per-model per-month details
    for name, res in results.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        res['per_month'].to_csv(
            f'artifacts/experiments/production_walkforward_{safe_name}.csv',
            index=False,
        )

    # JSON summary
    summary = {name: res['overall'] for name, res in results.items()}
    with open('artifacts/experiments/production_walkforward_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print final comparison table ──────────────────────────────────────

    print("\n" + "=" * 80)
    print("FINAL COMPARISON — PRODUCTION FEATURES ONLY (walk-forward)")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("\nResults saved to artifacts/experiments/production_walkforward_*.csv")


if __name__ == '__main__':
    main()
