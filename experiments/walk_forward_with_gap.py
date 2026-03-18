"""
Walk-Forward Backtest with Realistic SNAP Data Gap

In production, SNAP data lags several months behind Google Trends.
E.g., training data ends March 2025 but we predict August 2025 (5-month gap).

This script tests how model performance degrades as the gap increases:
  gap=0: train up to T-1, predict T  (current walk-forward, unrealistic)
  gap=1: train up to T-2, predict T
  ...
  gap=5: train up to T-6, predict T  (realistic production scenario)
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

PRODUCTION_FEATURES = [
    'Population', 'Median_Income',
    'monthly_average_CalFresh', 'monthly_average_FoodBank',
    'month',
]


def load_data():
    df = pd.read_csv("src/data/aggregateTrends_scaled.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['county', 'date']).reset_index(drop=True)
    df['SNAP_target'] = df.groupby('county')['SNAP_Application_Rate'].shift(-1)
    df['target_month'] = df['date'] + pd.DateOffset(months=1)
    df = df.dropna(subset=['SNAP_target', 'target_month'] + PRODUCTION_FEATURES)
    return df


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


def make_model():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.01,
        min_child_weight=6, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=4, reg_alpha=0,
        random_state=42, n_jobs=-1,
    )


def run_walk_forward_with_gap(df, gap_months):
    """
    Walk-forward backtest with a gap between training and prediction.

    For each target month T:
      - Train on rows where target_month <= T - (1 + gap_months)
      - Predict rows where target_month == T

    gap=0: standard walk-forward (train up to T-1, predict T)
    gap=5: realistic production gap (train up to T-6, predict T)
    """
    unique_months = sorted(df['target_month'].unique())
    n = len(unique_months)
    start = 5 + gap_months  # need enough history + gap buffer

    all_pred, all_actual = [], []

    for i in range(start, n):
        T = unique_months[i]
        # Training cutoff: only use data from gap_months before T
        cutoff = unique_months[i - gap_months] if gap_months > 0 else T

        train_df = df[df['target_month'] < cutoff]
        test_df = df[df['target_month'] == T]

        X_train = train_df[PRODUCTION_FEATURES]
        y_train = train_df['SNAP_target']
        X_test = test_df[PRODUCTION_FEATURES]
        y_test = test_df['SNAP_target']

        # Clean
        trn_ok = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        tst_ok = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[trn_ok], y_train[trn_ok]
        X_test, y_test = X_test[tst_ok], y_test[tst_ok]

        if len(X_train) < 50 or len(X_test) < 3:
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


def main():
    df = load_data()
    print(f"Data: {len(df)} rows, {df['county'].nunique()} counties, {df['metro_area'].nunique()} metros")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    gaps = [0, 1, 2, 3, 4, 5]
    results = []

    print("\n" + "=" * 80)
    print("WALK-FORWARD WITH REALISTIC SNAP DATA GAP")
    print("=" * 80)

    for gap in gaps:
        overall, n_preds = run_walk_forward_with_gap(df, gap)
        results.append({
            'Gap_Months': gap,
            'R2': round(overall['r2'], 4),
            'RMSE': round(overall['rmse'], 6),
            'MAE': round(overall['mae'], 6),
            'sMAPE': round(overall['smape'], 2),
            'Predictions': n_preds,
        })
        label = "no gap (unrealistic)" if gap == 0 else f"{gap}-month gap"
        if gap == 5:
            label += " (production)"
        print(f"  Gap={gap} ({label}): R²={overall['r2']:.4f}, MAE={overall['mae']:.6f}, sMAPE={overall['smape']:.2f}%")

    results_df = pd.DataFrame(results)

    os.makedirs('artifacts/experiments', exist_ok=True)
    results_df.to_csv('artifacts/experiments/walkforward_gap_analysis.csv', index=False)

    print("\n" + "=" * 80)
    print("GAP ANALYSIS RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\nSaved to artifacts/experiments/walkforward_gap_analysis.csv")


if __name__ == '__main__':
    main()
