import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends_scaled.csv"
MODELS_DIR = "county_models"

def load_and_prepare_data():
    """Load and prepare the training data."""
    df = pd.read_csv(AGGREGATE_TRENDS_FILE)
    
    # Use Population, trend columns, month, and median income features
    feature_cols = ["Population", "Median_Income"] + [col for col in df.columns if col.startswith('monthly_average_')]
    
    # Add minimal month feature (avoid overfitting)
    month_features = ['month']
    
    # Combine all features
    feature_cols = feature_cols + month_features
    
    # Filter to only include features that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df["SNAP_Application_Rate"]
    
    # Drop rows with missing values in any feature or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Ensure target values are positive (SNAP application rates cannot be negative)
    y = y.clip(lower=0)
    
    return X, y, feature_cols

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics."""
    # Ensure predictions are non-negative
    y_pred = np.clip(y_pred, 0, None)
    
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error) - handle division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    
    # Additional metrics
    mape_alt = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Standard MAPE
    
    # Mean Absolute Scaled Error (MASE) - using naive forecast as baseline
    naive_forecast_error = np.mean(np.abs(np.diff(y_true)))
    mase = mae / naive_forecast_error if naive_forecast_error > 0 else float('inf')
    
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(y_true - np.mean(y_true)))
    
    # Coefficient of Variation of RMSE (CV-RMSE)
    cv_rmse = (rmse / np.mean(y_true)) * 100
    
    # Coefficient of Variation of MAE (CV-MAE)
    cv_mae = (mae / np.mean(y_true)) * 100
    
    return {
        'R²': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MAPE_Alt': mape_alt,
        'sMAPE': smape,
        'MASE': mase,
        'MAD': mad,
        'CV-RMSE': cv_rmse,
        'CV-MAE': cv_mae
    }

def evaluate_model():
    """Evaluate the trained XGBoost model comprehensively."""
    print("Loading and preparing data...")
    X, y, feature_cols = load_and_prepare_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Target range: {y.min():.6f} - {y.max():.6f}")
    print(f"Target mean: {y.mean():.6f}, Target std: {y.std():.6f}")
    
    # Load the trained model
    print("\nLoading trained XGBoost model...")
    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "rb") as f:
        model_info = pickle.load(f)
    
    model = model_info["model"]
    print(f"Model type: {model_info['type']}")
    
    # Split data for evaluation
    print("\nSplitting data for evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics for training set
    print("\n" + "="*80)
    print("TRAINING SET EVALUATION")
    print("="*80)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    for metric, value in train_metrics.items():
        print(f"{metric:12}: {value:.6f}")
    
    # Calculate metrics for test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    for metric, value in test_metrics.items():
        print(f"{metric:12}: {value:.6f}")
    
    # Cross-validation evaluation
    print("\n" + "="*80)
    print("CROSS-VALIDATION EVALUATION")
    print("="*80)
    
    # R² Cross-validation
    cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_r2_mean = cv_r2_scores.mean()
    cv_r2_std = cv_r2_scores.std()
    
    # RMSE Cross-validation
    cv_rmse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse_mean = cv_rmse_scores.mean()
    cv_rmse_std = cv_rmse_scores.std()
    
    # MAE Cross-validation
    cv_mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae_mean = cv_mae_scores.mean()
    cv_mae_std = cv_mae_scores.std()
    
    print(f"CV R²       : {cv_r2_mean:.6f} ± {cv_r2_std:.6f}")
    print(f"CV RMSE     : {cv_rmse_mean:.6f} ± {cv_rmse_std:.6f}")
    print(f"CV MAE      : {cv_mae_mean:.6f} ± {cv_mae_std:.6f}")
    
    # Individual fold scores
    print(f"\nIndividual CV R² scores: {[f'{score:.4f}' for score in cv_r2_scores]}")
    print(f"Individual CV RMSE scores: {[f'{score:.6f}' for score in cv_rmse_scores]}")
    print(f"Individual CV MAE scores: {[f'{score:.6f}' for score in cv_mae_scores]}")
    
    # Feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_importance_pairs = list(zip(feature_cols, feature_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance_pairs:
            print(f"{feature:25}: {importance:.6f}")
    
    # Model summary
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"Model Type          : XGBoost")
    print(f"Number of Estimators: {model.n_estimators}")
    print(f"Max Depth           : {model.max_depth}")
    print(f"Learning Rate       : {model.learning_rate}")
    print(f"Training R²         : {train_metrics['R²']:.6f}")
    print(f"Test R²             : {test_metrics['R²']:.6f}")
    print(f"Cross-validation R² : {cv_r2_mean:.6f} ± {cv_r2_std:.6f}")
    print(f"Test RMSE           : {test_metrics['RMSE']:.6f}")
    print(f"Test MAE            : {test_metrics['MAE']:.6f}")
    print(f"Test MAPE           : {test_metrics['MAPE']:.2f}%")
    print(f"Test sMAPE          : {test_metrics['sMAPE']:.2f}%")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Metric': list(train_metrics.keys()) + ['CV_R2_Mean', 'CV_R2_Std', 'CV_RMSE_Mean', 'CV_RMSE_Std', 'CV_MAE_Mean', 'CV_MAE_Std'],
        'Training': list(train_metrics.values()) + [cv_r2_mean, cv_r2_std, cv_rmse_mean, cv_rmse_std, cv_mae_mean, cv_mae_std],
        'Test': list(test_metrics.values()) + ['', '', '', '', '', '']
    })
    
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print(f"\nDetailed results saved to: model_evaluation_results.csv")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_r2': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'cv_rmse': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'cv_mae': cv_mae_mean,
        'cv_mae_std': cv_mae_std
    }

if __name__ == "__main__":
    print("Starting comprehensive model evaluation...")
    results = evaluate_model()
    print("\nModel evaluation completed!")
