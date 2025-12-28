import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends_scaled.csv"
MODELS_DIR = "county_models"
TUNING_RESULTS_DIR = "xgboost_tuning_results"
os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)

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
    
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # Mean Absolute Scaled Error (MASE) - using naive forecast as baseline
    naive_forecast_error = np.mean(np.abs(np.diff(y_true)))
    mase = mae / naive_forecast_error if naive_forecast_error > 0 else float('inf')
    
    return {
        'R²': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'sMAPE': smape,
        'MASE': mase
    }

def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, X_full, y_full, model_name="Model"):
    """Comprehensively evaluate a model."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Cross-validation
    cv_r2_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='r2')
    cv_rmse_scores = -cross_val_score(model, X_full, y_full, cv=5, scoring='neg_root_mean_squared_error')
    cv_mae_scores = -cross_val_score(model, X_full, y_full, cv=5, scoring='neg_mean_absolute_error')
    
    print(f"\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric:8}: {value:.6f}")
    
    print(f"\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric:8}: {value:.6f}")
    
    print(f"\nCross-Validation:")
    print(f"  CV R²   : {cv_r2_scores.mean():.6f} ± {cv_r2_scores.std():.6f}")
    print(f"  CV RMSE : {cv_rmse_scores.mean():.6f} ± {cv_rmse_scores.std():.6f}")
    print(f"  CV MAE  : {cv_mae_scores.mean():.6f} ± {cv_mae_scores.std():.6f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop 5 Feature Importance:")
        feature_importance = model.feature_importances_
        feature_names = X_train.columns
        feature_importance_pairs = list(zip(feature_names, feature_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance_pairs[:5]:
            print(f"  {feature:25}: {importance:.6f}")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std(),
        'cv_mae_mean': cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std()
    }

def tune_xgboost():
    """Tune XGBoost hyperparameters using GridSearchCV and RandomizedSearchCV."""
    print("="*80)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("="*80)
    
    # Load data
    print("\n1. Loading and preparing data...")
    X, y, feature_cols = load_and_prepare_data()
    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {feature_cols}")
    
    # Split data
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Load current model for comparison
    print("\n3. Loading current XGBoost model for comparison...")
    current_model_path = os.path.join(MODELS_DIR, "global_model.pkl")
    if os.path.exists(current_model_path):
        with open(current_model_path, "rb") as f:
            current_model_info = pickle.load(f)
        current_model = current_model_info["model"]
        print(f"   Current model loaded: n_estimators={current_model.n_estimators}, "
              f"max_depth={current_model.max_depth}, learning_rate={current_model.learning_rate}")
    else:
        print("   No current model found, skipping comparison")
        current_model = None
    
    # Define parameter grid for tuning
    print("\n4. Setting up hyperparameter search space...")
    
    # First, do a broader RandomizedSearchCV to explore the space
    print("\n   Phase 1: Randomized Search (exploring parameter space)...")
    param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1, 5],
        'reg_lambda': [0, 0.1, 0.5, 1, 5, 10]
    }
    
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=50,  # Try 50 random combinations
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("   Running randomized search (this may take a few minutes)...")
    random_search.fit(X_train, y_train)
    
    print(f"\n   Best parameters from randomized search:")
    for param, value in random_search.best_params_.items():
        print(f"     {param}: {value}")
    print(f"   Best CV score: {random_search.best_score_:.6f}")
    
    # Refine with GridSearchCV around the best parameters
    print("\n   Phase 2: Grid Search (refining around best parameters)...")
    best_params = random_search.best_params_
    
    # Create a refined grid around the best parameters
    refined_grid = {
        'n_estimators': sorted(list(set([
            max(50, best_params['n_estimators'] - 50),
            best_params['n_estimators'],
            best_params['n_estimators'] + 50
        ]))),
        'max_depth': sorted(list(set([
            max(3, best_params['max_depth'] - 1),
            best_params['max_depth'],
            min(8, best_params['max_depth'] + 1)
        ]))),
        'learning_rate': sorted(list(set([
            max(0.01, best_params['learning_rate'] - 0.02),
            best_params['learning_rate'],
            min(0.2, best_params['learning_rate'] + 0.02)
        ]))),
        'min_child_weight': sorted(list(set([
            max(1, best_params['min_child_weight'] - 1),
            best_params['min_child_weight'],
            best_params['min_child_weight'] + 1
        ]))),
        'subsample': sorted(list(set([
            max(0.6, best_params['subsample'] - 0.1),
            best_params['subsample'],
            min(1.0, best_params['subsample'] + 0.1)
        ]))),
        'colsample_bytree': sorted(list(set([
            max(0.6, best_params['colsample_bytree'] - 0.1),
            best_params['colsample_bytree'],
            min(1.0, best_params['colsample_bytree'] + 0.1)
        ]))),
        'reg_alpha': sorted(list(set([
            max(0, best_params['reg_alpha'] - 0.5),
            best_params['reg_alpha'],
            best_params['reg_alpha'] + 0.5
        ]))),
        'reg_lambda': sorted(list(set([
            max(0, best_params['reg_lambda'] - 1),
            best_params['reg_lambda'],
            best_params['reg_lambda'] + 1
        ])))
    }
    
    # Ensure all parameters have at least one value
    for key in refined_grid:
        if len(refined_grid[key]) == 0:
            refined_grid[key] = [best_params[key]]
    
    grid_search = GridSearchCV(
        base_model,
        refined_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    print("   Running grid search refinement...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n   Best parameters from grid search:")
    for param, value in grid_search.best_params_.items():
        print(f"     {param}: {value}")
    print(f"   Best CV score: {grid_search.best_score_:.6f}")
    
    # Get the best tuned model
    tuned_model = grid_search.best_estimator_
    
    # Evaluate both models
    print("\n5. Evaluating models...")
    
    if current_model:
        print("\n   Evaluating CURRENT XGBoost model:")
        current_results = evaluate_model_comprehensive(
            current_model, X_train, X_test, y_train, y_test, X, y, "Current XGBoost"
        )
    
    print("\n   Evaluating TUNED XGBoost model:")
    tuned_results = evaluate_model_comprehensive(
        tuned_model, X_train, X_test, y_train, y_test, X, y, "Tuned XGBoost"
    )
    
    # Compare models
    if current_model:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"{'Metric':<20} {'Current':<20} {'Tuned':<20} {'Improvement':<20}")
        print("-"*80)
        
        metrics_to_compare = [
            ('Test R²', 'R²'),
            ('Test RMSE', 'RMSE'),
            ('Test MAE', 'MAE'),
            ('CV R²', 'cv_r2_mean'),
            ('CV RMSE', 'cv_rmse_mean'),
            ('CV MAE', 'cv_mae_mean')
        ]
        
        for metric_name, metric_key in metrics_to_compare:
            if metric_key in ['R²', 'RMSE', 'MAE']:
                current_val = current_results['test_metrics'][metric_key]
                tuned_val = tuned_results['test_metrics'][metric_key]
            else:
                current_val = current_results[metric_key]
                tuned_val = tuned_results[metric_key]
            
            if metric_name in ['Test R²', 'CV R²']:
                improvement = ((tuned_val - current_val) / abs(current_val)) * 100 if current_val != 0 else 0
                improvement_str = f"{improvement:+.2f}%"
            else:
                improvement = ((current_val - tuned_val) / current_val) * 100 if current_val != 0 else 0
                improvement_str = f"{improvement:+.2f}%"
            
            print(f"{metric_name:<20} {current_val:<20.6f} {tuned_val:<20.6f} {improvement_str:<20}")
    
    # Save results
    print("\n6. Saving results...")
    
    # Save tuned model
    tuned_model_path = os.path.join(TUNING_RESULTS_DIR, "tuned_xgboost_model.pkl")
    with open(tuned_model_path, "wb") as f:
        pickle.dump({
            "model": tuned_model,
            "features": feature_cols,
            "type": "xgboost_tuned",
            "best_params": grid_search.best_params_,
            "cv_score": grid_search.best_score_,
            "results": tuned_results
        }, f)
    print(f"   Tuned model saved to: {tuned_model_path}")
    
    # Save comparison results
    comparison_data = {
        'Model': ['Current XGBoost', 'Tuned XGBoost'],
        'Test_R2': [current_results['test_metrics']['R²'] if current_model else None, tuned_results['test_metrics']['R²']],
        'Test_RMSE': [current_results['test_metrics']['RMSE'] if current_model else None, tuned_results['test_metrics']['RMSE']],
        'Test_MAE': [current_results['test_metrics']['MAE'] if current_model else None, tuned_results['test_metrics']['MAE']],
        'CV_R2_Mean': [current_results['cv_r2_mean'] if current_model else None, tuned_results['cv_r2_mean']],
        'CV_R2_Std': [current_results['cv_r2_std'] if current_model else None, tuned_results['cv_r2_std']],
        'CV_RMSE_Mean': [current_results['cv_rmse_mean'] if current_model else None, tuned_results['cv_rmse_mean']],
        'CV_MAE_Mean': [current_results['cv_mae_mean'] if current_model else None, tuned_results['cv_mae_mean']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(TUNING_RESULTS_DIR, "tuning_comparison.csv"), index=False)
    print(f"   Comparison results saved to: {TUNING_RESULTS_DIR}/tuning_comparison.csv")
    
    # Save detailed report
    report_path = os.path.join(TUNING_RESULTS_DIR, "TUNING_REPORT.md")
    with open(report_path, 'w') as f:
        f.write("# XGBoost Hyperparameter Tuning Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Best Hyperparameters\n\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"- **{param}**: {value}\n")
        f.write(f"\n**Best CV R² Score**: {grid_search.best_score_:.6f}\n\n")
        
        f.write("## Model Performance\n\n")
        f.write("### Tuned XGBoost\n\n")
        f.write("**Training Metrics:**\n")
        for metric, value in tuned_results['train_metrics'].items():
            f.write(f"- {metric}: {value:.6f}\n")
        f.write("\n**Test Metrics:**\n")
        for metric, value in tuned_results['test_metrics'].items():
            f.write(f"- {metric}: {value:.6f}\n")
        f.write(f"\n**Cross-Validation:**\n")
        f.write(f"- CV R²: {tuned_results['cv_r2_mean']:.6f} ± {tuned_results['cv_r2_std']:.6f}\n")
        f.write(f"- CV RMSE: {tuned_results['cv_rmse_mean']:.6f} ± {tuned_results['cv_rmse_std']:.6f}\n")
        f.write(f"- CV MAE: {tuned_results['cv_mae_mean']:.6f} ± {tuned_results['cv_mae_std']:.6f}\n")
        
        if current_model:
            f.write("\n### Current XGBoost\n\n")
            f.write("**Training Metrics:**\n")
            for metric, value in current_results['train_metrics'].items():
                f.write(f"- {metric}: {value:.6f}\n")
            f.write("\n**Test Metrics:**\n")
            for metric, value in current_results['test_metrics'].items():
                f.write(f"- {metric}: {value:.6f}\n")
            f.write(f"\n**Cross-Validation:**\n")
            f.write(f"- CV R²: {current_results['cv_r2_mean']:.6f} ± {current_results['cv_r2_std']:.6f}\n")
            f.write(f"- CV RMSE: {current_results['cv_rmse_mean']:.6f} ± {current_results['cv_rmse_std']:.6f}\n")
            f.write(f"- CV MAE: {current_results['cv_mae_mean']:.6f} ± {current_results['cv_mae_std']:.6f}\n")
    
    print(f"   Detailed report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    
    return tuned_model, tuned_results, current_results if current_model else None

if __name__ == "__main__":
    print("Starting XGBoost hyperparameter tuning...")
    tuned_model, tuned_results, current_results = tune_xgboost()
    print("\nTuning process completed!")

