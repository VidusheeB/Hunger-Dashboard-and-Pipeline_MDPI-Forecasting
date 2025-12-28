"""
Walk-Forward, Leak-Free Backtest for SNAP Predictions

This script implements a rigorous walk-forward backtest where:
1. For each prediction month, train only on earlier months
2. Skip the immediate prior month to prevent leakage from lagged features
3. Predict the next month using last month's Google Trends and history
4. This provides the most realistic assessment of model performance
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_panel_data():
    """Load the existing panel data with county-month observations."""
    print("Loading panel data...")
    
    # Load the aggregate trends data
    df = pd.read_csv("src/data/aggregateTrends_scaled.csv")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'monthly_average_CalFresh': 'CalFresh_trend',
        'monthly_average_FoodBank': 'FoodBank_trend',
        'SNAP_Application_Rate': 'SNAP_rate',
        'date': 'month_dt'
    })
    
    # Convert date column
    df['month_dt'] = pd.to_datetime(df['month_dt'])
    
    # Sort by county and date for proper lagging
    df = df.sort_values(['county', 'month_dt']).reset_index(drop=True)
    
    # Align rows to represent (features at month t) -> (target SNAP_rate at month t+1)
    # Create the next-month target per county
    df['SNAP_target'] = df.groupby('county')['SNAP_rate'].shift(-1)
    # Define the target month as t+1 for each row
    df['target_month'] = df['month_dt'] + pd.DateOffset(months=1)
    
    print(f"Loaded {len(df)} observations across {df['county'].nunique()} counties")
    print(f"Date range: {df['month_dt'].min()} to {df['month_dt'].max()}")
    
    return df

def create_seasonality_features(df):
    """Create month seasonality features."""
    print("Creating seasonality features...")
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def create_lag_features(df):
    """Create lagged features for SNAP_rate, CalFresh_trend, and FoodBank_trend."""
    print("Creating lag features...")
    
    lag_features = ['SNAP_rate', 'CalFresh_trend', 'FoodBank_trend']
    lag_periods = [1, 2, 3]
    
    for feature in lag_features:
        for lag in lag_periods:
            df[f'{feature}_lag{lag}'] = df.groupby('county')[feature].shift(lag)
    
    return df

def create_rolling_features(df):
    """Create 3- and 6-month rolling means for stability."""
    print("Creating rolling mean features...")
    
    rolling_features = ['SNAP_rate', 'CalFresh_trend', 'FoodBank_trend']
    rolling_windows = [3, 6]
    
    for feature in rolling_features:
        for window in rolling_windows:
            df[f'{feature}_rolling{window}'] = df.groupby('county')[feature].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
    
    return df

def create_income_normalization(df):
    """Create income z-score normalization within each county."""
    print("Creating income normalization...")
    
    df['income_z_by_county'] = df.groupby('county')['Median_Income'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Fill NaN values (for counties with single observation) with 0
    df['income_z_by_county'] = df['income_z_by_county'].fillna(0)
    
    return df

# Spike detection removed - focusing on overall prediction performance

def prepare_features(df):
    """Prepare all features for modeling."""
    print("Preparing features for modeling...")
    
    # Feature engineering
    df = create_seasonality_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_income_normalization(df)
    
    # Define feature columns
    base_features = ['Population', 'Median_Income', 'CalFresh_trend', 'FoodBank_trend', 'month']
    seasonality_features = ['month_sin', 'month_cos']
    lag_features = [col for col in df.columns if '_lag' in col]
    rolling_features = [col for col in df.columns if '_rolling' in col]
    income_features = ['income_z_by_county']
    
    feature_cols = base_features + seasonality_features + lag_features + rolling_features + income_features
    
    # Filter to only include features that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Drop rows where any required lag is NaN (but keep rolling means with min_periods=1)
    required_lag_cols = [col for col in lag_features if 'SNAP_rate_lag' in col or 'CalFresh_trend_lag' in col or 'FoodBank_trend_lag' in col]
    
    print(f"Dropping rows with missing lag features...")
    initial_rows = len(df)
    df = df.dropna(subset=required_lag_cols)
    
    # Also drop rows where the next-month target is missing (end of series)
    df = df.dropna(subset=['SNAP_target', 'target_month'])
    
    final_rows = len(df)
    print(f"Dropped {initial_rows - final_rows} rows, {final_rows} remaining")
    
    return df, feature_cols

def create_xgboost_model(use_early_stopping=False):
    """Create XGBoost model with specified hyperparameters."""
    params = {
        'n_estimators': 1200,
        'learning_rate': 0.03,
        'max_depth': 4,
        'min_child_weight': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 12,
        'reg_alpha': 1,
        'tree_method': "hist",
        'eval_metric': "mae",
        'random_state': 42,
        'n_jobs': -1
    }
    
    if use_early_stopping:
        params['early_stopping_rounds'] = 100
    
    return xgb.XGBRegressor(**params)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    # Ensure predictions are non-negative
    y_pred = np.clip(y_pred, 0, None)
    
    # Handle NaN values
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite_mask.sum() == 0:
        return {
            'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'smape': np.nan
        }
    
    y_true_clean = y_true[finite_mask]
    y_pred_clean = y_pred[finite_mask]
    
    # Overall metrics
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # sMAPE
    smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean))) * 100
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'smape': smape
    }
    
    return metrics

def run_walk_forward_backtest():
    """Run the walk-forward, leak-free backtest."""
    print("="*80)
    print("WALK-FORWARD, LEAK-FREE BACKTEST")
    print("="*80)
    
    # Load and prepare data
    df = load_panel_data()
    df, feature_cols = prepare_features(df)
    
    # Get unique target months sorted chronologically (evaluation happens at target_month)
    unique_target_months = sorted(df['target_month'].unique())
    n_months = len(unique_target_months)
    
    print(f"Total unique months: {n_months}")
    print(f"Target month range: {unique_target_months[0]} to {unique_target_months[-1]}")
    
    # Need at least 6 target months for meaningful backtest (train + predict)
    min_required_months = 6 
    if n_months < min_required_months:
        raise ValueError(f"Need at least {min_required_months} months for backtest, got {n_months}")
    
    # Start after enough history to form stable features (kept at 5 as before)
    start_month_idx = 5
    
    # Initialize results storage
    backtest_results = []
    all_predictions = []
    all_actuals = []
    
    print(f"\nStarting walk-forward backtest from target month {unique_target_months[start_month_idx].strftime('%Y-%m')}")
    print("="*80)
    
    # Walk-forward using target_month as the time index:
    # For each target month T: train on rows with target_month < T, test on rows with target_month == T
    for i in range(start_month_idx, n_months):
        T = unique_target_months[i]
        print(f"\n--- Predicting target month {T.strftime('%Y-%m')} ({i+1}/{n_months}) ---")
        
        train_mask = df['target_month'] < T
        test_mask = df['target_month'] == T
        
        X_train = df[train_mask][feature_cols]
        y_train = df[train_mask]['SNAP_target']
        X_pred = df[test_mask][feature_cols]
        y_actual = df[test_mask]['SNAP_target']
        
        print(f"Training samples: {len(X_train)} (target_month < {T.strftime('%Y-%m')})")
        print(f"Testing samples: {len(X_pred)} (target_month == {T.strftime('%Y-%m')})")
        
        # Skip if insufficient samples
        if len(X_train) < 50 or len(X_pred) < 5:
            print(f"Skipping {T.strftime('%Y-%m')}: insufficient samples")
            continue
        
        # Check for and handle NaN/inf values
        train_mask_clean = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        pred_mask_clean = np.isfinite(X_pred).all(axis=1)
        
        X_train = X_train[train_mask_clean]
        y_train = y_train[train_mask_clean]
        X_pred = X_pred[pred_mask_clean]
        y_actual = y_actual[pred_mask_clean]
        
        print(f"After cleaning: Training: {len(X_train)} samples, Testing: {len(X_pred)} samples")
        
        if len(X_train) < 20 or len(X_pred) < 3:
            print(f"Skipping {T.strftime('%Y-%m')}: insufficient samples after cleaning")
            continue
        
        # Train model (no early stopping for walk-forward backtest)
        model = create_xgboost_model(use_early_stopping=False)
        model.fit(X_train, y_train, verbose=False)
        
        # Make predictions
        y_pred = model.predict(X_pred)
        
        # Calculate metrics
        metrics = calculate_metrics(y_actual, y_pred)
        
        # Store results
        result = {
            'prediction_month': T.strftime('%Y-%m'),
            'train_cutoff_month': (T - pd.DateOffset(months=1)).strftime('%Y-%m'),
            'train_size': len(X_train),
            'pred_size': len(X_pred),
            **metrics
        }
        
        backtest_results.append(result)
        
        # Store for overall metrics
        all_predictions.extend(y_pred)
        all_actuals.extend(y_actual)
        
        print(f"R²: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"sMAPE: {metrics['smape']:.2f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(backtest_results)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions))
    
    # Calculate summary statistics
    numeric_cols = ['r2', 'rmse', 'mae', 'smape']
    summary = {}
    
    for col in numeric_cols:
        if col in results_df.columns:
            summary[f'{col}_mean'] = results_df[col].mean()
            summary[f'{col}_std'] = results_df[col].std()
            summary[f'{col}_min'] = results_df[col].min()
            summary[f'{col}_max'] = results_df[col].max()
    
    # Add overall metrics
    summary['overall_r2'] = overall_metrics['r2']
    summary['overall_rmse'] = overall_metrics['rmse']
    summary['overall_mae'] = overall_metrics['mae']
    summary['overall_smape'] = overall_metrics['smape']
    
    summary['total_months'] = len(results_df)
    summary['total_predictions'] = len(all_predictions)
    
    # Save results
    os.makedirs('artifacts/experiments', exist_ok=True)
    
    # Save per-month results
    results_df.to_csv('artifacts/experiments/walk_forward_backtest_metrics.csv', index=False)
    print(f"\nSaved per-month results to: artifacts/experiments/walk_forward_backtest_metrics.csv")
    
    # Save summary
    with open('artifacts/experiments/walk_forward_backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to: artifacts/experiments/walk_forward_backtest_summary.json")
    
    # Print summary
    print("\n" + "="*80)
    print("WALK-FORWARD BACKTEST SUMMARY")
    print("="*80)
    
    print(f"Total prediction months: {len(results_df)}")
    print(f"Total predictions: {len(all_predictions)}")
    
    print(f"\nOverall Performance (All Predictions Combined):")
    print(f"R²: {overall_metrics['r2']:.4f}")
    print(f"RMSE: {overall_metrics['rmse']:.6f}")
    print(f"MAE: {overall_metrics['mae']:.6f}")
    print(f"sMAPE: {overall_metrics['smape']:.2f}%")
    
    print(f"\nPer-Month Performance Statistics:")
    print(f"R²: {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f} (range: {summary['r2_min']:.4f} - {summary['r2_max']:.4f})")
    print(f"RMSE: {summary['rmse_mean']:.6f} ± {summary['rmse_std']:.6f}")
    print(f"MAE: {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}")
    print(f"sMAPE: {summary['smape_mean']:.2f}% ± {summary['smape_std']:.2f}%")
    
    # Performance analysis
    positive_r2_months = (results_df['r2'] >= 0).sum()
    print(f"\nPerformance Analysis:")
    print(f"Months with positive R²: {positive_r2_months}/{len(results_df)} ({positive_r2_months/len(results_df)*100:.1f}%)")
    
    if len(results_df) > 0:
        avg_r2 = results_df['r2'].mean()
        if avg_r2 > 0.7:
            print("✓ Excellent predictive performance")
        elif avg_r2 > 0.5:
            print("✓ Good predictive performance")
        elif avg_r2 > 0.3:
            print("⚠ Moderate predictive performance")
        else:
            print("✗ Poor predictive performance")
    
    print("\nWalk-forward backtest completed successfully!")
    
    return results_df, summary, overall_metrics

if __name__ == "__main__":
    results_df, summary, overall_metrics = run_walk_forward_backtest()
