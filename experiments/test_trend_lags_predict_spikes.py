"""
Experiment: Test whether lagged Google Trends + rolling means improve next-month SNAP rate predictions,
especially during spikes.

This script tests the hypothesis that including lagged features and rolling averages
can improve prediction accuracy, particularly for spike months.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
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

def identify_spike_months(df):
    """Identify spike months based on month-over-month changes in SNAP_rate."""
    print("Identifying spike months...")
    
    # Calculate month-over-month change within each county
    df['SNAP_rate_mom_change'] = df.groupby('county')['SNAP_rate'].pct_change()
    
    # Calculate 80th percentile threshold for positive changes across all data
    positive_changes = df[df['SNAP_rate_mom_change'] > 0]['SNAP_rate_mom_change'].dropna()
    spike_threshold = positive_changes.quantile(0.8)
    
    # Mark spike months
    df['is_spike'] = (df['SNAP_rate_mom_change'] >= spike_threshold).astype(int)
    
    spike_count = df['is_spike'].sum()
    total_months = len(df)
    
    print(f"Spike threshold (80th percentile): {spike_threshold:.4f}")
    print(f"Identified {spike_count} spike months out of {total_months} total ({spike_count/total_months*100:.1f}%)")
    
    return df, spike_threshold

def prepare_features(df):
    """Prepare all features for modeling."""
    print("Preparing features for modeling...")
    
    # Feature engineering
    df = create_seasonality_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_income_normalization(df)
    
    # Identify spike months
    df, spike_threshold = identify_spike_months(df)
    
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
    final_rows = len(df)
    print(f"Dropped {initial_rows - final_rows} rows, {final_rows} remaining")
    
    return df, feature_cols, spike_threshold

def create_purged_rolling_timeseries_cv(df, n_folds=5, purge_gap=1):
    """
    Create Purged Rolling TimeSeries CV (grouped by month, across all counties).
    
    For each fold:
    - Choose a validation month near the end of the series
    - Training window = all rows with month ≤ (val_month - 1 - purge_gap)
    - Validation window = all rows with month = val_month across all counties
    - No shuffling, strictly past→future splits
    """
    print("Creating Purged Rolling TimeSeries CV folds...")
    
    # Get unique months sorted chronologically
    unique_months = sorted(df['month_dt'].unique())
    n_months = len(unique_months)
    
    print(f"Total unique months: {n_months}")
    print(f"Month range: {unique_months[0]} to {unique_months[-1]}")
    print(f"Purge gap: {purge_gap} month(s)")
    
    # Need at least 3 months for meaningful CV (train + purge + validation)
    min_required_months = 3 + purge_gap
    if n_months < min_required_months:
        raise ValueError(f"Need at least {min_required_months} months for CV, got {n_months}")
    
    folds = []
    
    # Select validation months from the latter part of the series
    # Start from month index that allows for sufficient training data
    min_val_month_idx = max(2 + purge_gap, n_months // 3)  # At least 1/3 through the series
    
    # Create n_folds validation months, evenly spaced in the latter part
    val_month_indices = np.linspace(min_val_month_idx, n_months - 1, n_folds, dtype=int)
    
    for fold, val_month_idx in enumerate(val_month_indices):
        val_month = unique_months[val_month_idx]
        
        # Training window: all months ≤ (val_month - 1 - purge_gap)
        train_cutoff_idx = val_month_idx - 1 - purge_gap
        
        if train_cutoff_idx < 0:
            print(f"Skipping fold {fold}: insufficient training data before purge")
            continue
            
        train_months = unique_months[:train_cutoff_idx + 1]
        
        # Count samples
        train_mask = df['month_dt'].isin(train_months)
        val_mask = df['month_dt'] == val_month
        
        train_size = train_mask.sum()
        val_size = val_mask.sum()
        
        # Skip if insufficient samples
        if train_size < 50 or val_size < 10:
            print(f"Skipping fold {fold}: insufficient samples (train={train_size}, val={val_size})")
            continue
        
        # Create fold info
        fold_info = {
            'fold': fold,
            'train_months': train_months,
            'val_month': val_month,
            'train_cutoff_month': unique_months[train_cutoff_idx],
            'purge_months': unique_months[train_cutoff_idx + 1:val_month_idx] if val_month_idx > train_cutoff_idx + 1 else [],
            'train_size': train_size,
            'val_size': val_size,
            'purge_gap': purge_gap
        }
        
        folds.append(fold_info)
        
        purge_info = f", Purge: {len(fold_info['purge_months'])} months" if fold_info['purge_months'] else ""
        print(f"Fold {fold}: Train through {fold_info['train_cutoff_month'].strftime('%Y-%m')} "
              f"(n={train_size}), Validate on {val_month.strftime('%Y-%m')} (n={val_size}){purge_info}")
    
    print(f"Created {len(folds)} valid folds")
    return folds

def create_xgboost_model():
    """Create XGBoost model with specified hyperparameters."""
    return xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=12,
        reg_alpha=1,
        tree_method="hist",
        eval_metric="mae",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1
    )

def calculate_metrics(y_true, y_pred, is_spike=None):
    """Calculate comprehensive metrics."""
    # Ensure predictions are non-negative
    y_pred = np.clip(y_pred, 0, None)
    
    # Overall metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # sMAPE
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'smape': smape
    }
    
    # Spike-specific metrics
    if is_spike is not None:
        spike_mask = is_spike == 1
        if spike_mask.sum() > 0:
            spike_r2 = r2_score(y_true[spike_mask], y_pred[spike_mask])
            spike_mae = mean_absolute_error(y_true[spike_mask], y_pred[spike_mask])
            metrics.update({
                'r2_spike': spike_r2,
                'mae_spike': spike_mae,
                'n_spikes': spike_mask.sum()
            })
        else:
            metrics.update({
                'r2_spike': np.nan,
                'mae_spike': np.nan,
                'n_spikes': 0
            })
    
    return metrics

def run_experiment():
    """Run the complete experiment."""
    print("="*80)
    print("EXPERIMENT: Lagged Google Trends + Rolling Means for SNAP Predictions")
    print("="*80)
    
    # Load and prepare data
    df = load_panel_data()
    df, feature_cols, spike_threshold = prepare_features(df)
    
    # Create purged rolling timeseries CV folds
    folds = create_purged_rolling_timeseries_cv(df, n_folds=5, purge_gap=1)
    
    # Initialize results storage
    fold_results = []
    
    print("\n" + "="*80)
    print("RUNNING CROSS-VALIDATION")
    print("="*80)
    
    for fold_info in folds:
        print(f"\n--- Fold {fold_info['fold']} ---")
        
        # Split data
        train_mask = df['month_dt'].isin(fold_info['train_months'])
        val_mask = df['month_dt'] == fold_info['val_month']
        
        X_train = df[train_mask][feature_cols]
        y_train = df[train_mask]['SNAP_rate']
        X_val = df[val_mask][feature_cols]
        y_val = df[val_mask]['SNAP_rate']
        is_spike_val = df[val_mask]['is_spike'].values
        
        print(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
        print(f"Validation month: {fold_info['val_month'].strftime('%Y-%m')}")
        print(f"Spikes in validation: {is_spike_val.sum()}")
        
        # Scale target for training stability and handle NaN/inf values
        y_train_scaled = y_train * 10000
        y_val_scaled = y_val * 10000
        
        # Check for and handle NaN/inf values
        train_mask_clean = np.isfinite(y_train_scaled) & np.isfinite(X_train).all(axis=1)
        val_mask_clean = np.isfinite(y_val_scaled) & np.isfinite(X_val).all(axis=1)
        
        X_train = X_train[train_mask_clean]
        y_train_scaled = y_train_scaled[train_mask_clean]
        X_val = X_val[val_mask_clean]
        y_val_scaled = y_val_scaled[val_mask_clean]
        y_val = y_val[val_mask_clean]
        is_spike_val = is_spike_val[val_mask_clean]
        
        print(f"After cleaning: Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
        
        # Skip fold if not enough samples
        if len(X_train) < 10 or len(X_val) < 5:
            print(f"Skipping fold {fold_info['fold']} - insufficient samples after cleaning")
            continue
        
        # Train model
        model = create_xgboost_model()
        model.fit(
            X_train, y_train_scaled,
            eval_set=[(X_val, y_val_scaled)],
            verbose=False
        )
        
        # Make predictions and scale back
        y_pred_scaled = model.predict(X_val)
        y_pred = y_pred_scaled / 10000
        
        # Calculate metrics
        metrics = calculate_metrics(y_val, y_pred, is_spike_val)
        
        # Store results
        result = {
            'fold': fold_info['fold'],
            'val_month': fold_info['val_month'].strftime('%Y-%m'),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'n_spikes': is_spike_val.sum(),
            **metrics
        }
        
        fold_results.append(result)
        
        print(f"R²: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"sMAPE: {metrics['smape']:.2f}%")
        if 'r2_spike' in metrics and not np.isnan(metrics['r2_spike']):
            print(f"R² (spikes): {metrics['r2_spike']:.4f}")
            print(f"MAE (spikes): {metrics['mae_spike']:.6f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(fold_results)
    
    # Calculate summary statistics
    numeric_cols = ['r2', 'rmse', 'mae', 'smape', 'r2_spike', 'mae_spike']
    summary = {}
    
    for col in numeric_cols:
        if col in results_df.columns:
            summary[f'{col}_mean'] = results_df[col].mean()
            summary[f'{col}_std'] = results_df[col].std()
    
    summary['spike_threshold'] = spike_threshold
    summary['total_folds'] = len(folds)
    summary['total_spikes'] = results_df['n_spikes'].sum()
    summary['total_validation_samples'] = results_df['val_size'].sum()
    
    # Save results
    os.makedirs('artifacts/experiments', exist_ok=True)
    
    # Save per-fold results
    results_df.to_csv('artifacts/experiments/trend_lags_cv_metrics.csv', index=False)
    print(f"\nSaved per-fold results to: artifacts/experiments/trend_lags_cv_metrics.csv")
    
    # Save summary
    with open('artifacts/experiments/trend_lags_cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to: artifacts/experiments/trend_lags_cv_summary.json")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"Total folds: {len(folds)}")
    print(f"Total validation samples: {results_df['val_size'].sum()}")
    print(f"Total spike samples: {results_df['n_spikes'].sum()}")
    print(f"Spike threshold: {spike_threshold:.4f}")
    
    print(f"\nOverall Performance:")
    print(f"R²: {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    print(f"RMSE: {summary['rmse_mean']:.6f} ± {summary['rmse_std']:.6f}")
    print(f"MAE: {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}")
    print(f"sMAPE: {summary['smape_mean']:.2f}% ± {summary['smape_std']:.2f}%")
    
    if 'r2_spike_mean' in summary and not np.isnan(summary['r2_spike_mean']):
        print(f"\nSpike Performance:")
        print(f"R² (spikes): {summary['r2_spike_mean']:.4f} ± {summary['r2_spike_std']:.4f}")
        print(f"MAE (spikes): {summary['mae_spike_mean']:.6f} ± {summary['mae_spike_std']:.6f}")
    
    # Check acceptance criteria
    print(f"\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    non_negative_r2_folds = (results_df['r2'] >= 0).sum()
    print(f"Folds with non-negative R²: {non_negative_r2_folds}/{len(folds)}")
    
    if 'r2_spike_mean' in summary and not np.isnan(summary['r2_spike_mean']):
        print(f"Average R² on spikes: {summary['r2_spike_mean']:.4f}")
        if summary['r2_spike_mean'] > 0:
            print("✓ Model shows positive performance on spike detection")
        else:
            print("✗ Model struggles with spike detection")
    else:
        print("No spike samples in validation sets")
    
    print("\nExperiment completed successfully!")
    
    return results_df, summary

if __name__ == "__main__":
    results_df, summary = run_experiment()
