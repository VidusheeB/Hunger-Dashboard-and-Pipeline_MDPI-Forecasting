import os
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, but don't fail if it's not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - skipping XGBoost model")

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends_scaled.csv"
RESULTS_DIR = "model_comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def detailed_kfold_analysis(model, model_name, X, y, cv_folds=5):
    """Perform detailed k-fold cross-validation analysis."""
    print(f"\n{'='*80}")
    print(f"Detailed K-Fold Analysis: {model_name}")
    print(f"{'='*80}")
    
    # Create different CV strategies
    cv_strategies = {
        'KFold': KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        'StratifiedKFold': StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    }
    
    results = {}
    
    for cv_name, cv in cv_strategies.items():
        print(f"\n--- {cv_name} ---")
        
        # Get cross-validation scores
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        
        # Calculate statistics
        r2_mean = r2_scores.mean()
        r2_std = r2_scores.std()
        r2_min = r2_scores.min()
        r2_max = r2_scores.max()
        
        mse_mean = mse_scores.mean()
        mse_std = mse_scores.std()
        
        mae_mean = mae_scores.mean()
        mae_std = mae_scores.std()
        
        # Print detailed results
        print(f"R² Scores: {[f'{score:.4f}' for score in r2_scores]}")
        print(f"R² Mean: {r2_mean:.4f} ± {r2_std:.4f}")
        print(f"R² Range: [{r2_min:.4f}, {r2_max:.4f}]")
        print(f"MSE Mean: {mse_mean:.6f} ± {mse_std:.6f}")
        print(f"MAE Mean: {mae_mean:.6f} ± {mae_std:.6f}")
        
        # Store results
        results[cv_name] = {
            'r2_scores': r2_scores,
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'r2_min': r2_min,
            'r2_max': r2_max,
            'mse_mean': mse_mean,
            'mse_std': mse_std,
            'mae_mean': mae_mean,
            'mae_std': mae_std
        }
    
    return results

def compare_models_with_detailed_kfold():
    """Compare different machine learning models with detailed k-fold analysis."""
    print("Loading and preparing data...")
    X, y, feature_cols = load_and_prepare_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Target range: {y.min():.6f} - {y.max():.6f}")
    print(f"Target mean: {y.mean():.6f}, Target std: {y.std():.6f}")
    
    # Define models to compare - using the same configuration as your original Random Forest
    models = {
        'Random Forest (Original Config)': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Random Forest (Tuned)': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting (Tuned)': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.01),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        models['XGBoost (Tuned)'] = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    
    # Store all results for comparison
    all_kfold_results = {}
    
    # Perform detailed k-fold analysis for each model
    for model_name, model in models.items():
        try:
            print(f"\n{'#'*100}")
            print(f"ANALYZING: {model_name}")
            print(f"{'#'*100}")
            
            kfold_results = detailed_kfold_analysis(model, model_name, X, y, cv_folds=5)
            all_kfold_results[model_name] = kfold_results
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            continue
    
    # Create summary comparison
    create_kfold_summary_report(all_kfold_results)
    
    return all_kfold_results

def create_kfold_summary_report(all_kfold_results):
    """Create a summary report of k-fold analysis."""
    
    report_path = os.path.join(RESULTS_DIR, 'DETAILED_KFOLD_ANALYSIS.md')
    
    with open(report_path, 'w') as f:
        f.write("# Detailed K-Fold Cross-Validation Analysis\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report provides detailed k-fold cross-validation analysis for different machine learning models.\n\n")
        
        f.write("## K-Fold Analysis Results\n\n")
        
        # Create summary table
        f.write("### K-Fold Cross-Validation Summary\n\n")
        f.write("| Model | CV Strategy | R² Mean ± Std | R² Min | R² Max | MSE Mean | MAE Mean |\n")
        f.write("|-------|-------------|---------------|--------|--------|----------|----------|\n")
        
        for model_name, cv_results in all_kfold_results.items():
            for cv_name, results in cv_results.items():
                f.write(f"| {model_name} | {cv_name} | {results['r2_mean']:.4f} ± {results['r2_std']:.4f} | {results['r2_min']:.4f} | {results['r2_max']:.4f} | {results['mse_mean']:.6f} | {results['mae_mean']:.6f} |\n")
        
        f.write("\n## Detailed Results by Model\n\n")
        
        for model_name, cv_results in all_kfold_results.items():
            f.write(f"### {model_name}\n\n")
            
            for cv_name, results in cv_results.items():
                f.write(f"#### {cv_name}\n\n")
                f.write(f"- **R² Scores**: {[f'{score:.4f}' for score in results['r2_scores']]}\n")
                f.write(f"- **R² Mean**: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}\n")
                f.write(f"- **R² Range**: [{results['r2_min']:.4f}, {results['r2_max']:.4f}]\n")
                f.write(f"- **MSE Mean**: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}\n")
                f.write(f"- **MAE Mean**: {results['mae_mean']:.6f} ± {results['mae_std']:.6f}\n\n")
        
        f.write("## Key Insights\n\n")
        
        # Find best performing models
        best_models = []
        for model_name, cv_results in all_kfold_results.items():
            kfold_results = cv_results.get('KFold', {})
            if kfold_results:
                best_models.append((model_name, kfold_results['r2_mean'], kfold_results['r2_std']))
        
        best_models.sort(key=lambda x: x[1], reverse=True)
        
        f.write("### Top Performing Models (by K-Fold R² Mean):\n\n")
        for i, (model_name, r2_mean, r2_std) in enumerate(best_models[:5], 1):
            f.write(f"{i}. **{model_name}**: {r2_mean:.4f} ± {r2_std:.4f}\n")
        
        f.write("\n### Stability Analysis:\n\n")
        f.write("- Models with lower standard deviation are more stable across folds\n")
        f.write("- Models with consistent R² scores across folds are more reliable\n")
        f.write("- Large differences between min and max R² indicate potential overfitting\n\n")
    
    print(f"\nDetailed k-fold analysis report saved to: {report_path}")
    
    # Also create a CSV summary
    csv_path = os.path.join(RESULTS_DIR, 'kfold_analysis_summary.csv')
    
    summary_data = []
    for model_name, cv_results in all_kfold_results.items():
        for cv_name, results in cv_results.items():
            summary_data.append({
                'Model': model_name,
                'CV_Strategy': cv_name,
                'R2_Mean': results['r2_mean'],
                'R2_Std': results['r2_std'],
                'R2_Min': results['r2_min'],
                'R2_Max': results['r2_max'],
                'MSE_Mean': results['mse_mean'],
                'MSE_Std': results['mse_std'],
                'MAE_Mean': results['mae_mean'],
                'MAE_Std': results['mae_std']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_path, index=False)
    print(f"K-fold analysis summary CSV saved to: {csv_path}")

if __name__ == "__main__":
    print("Starting detailed k-fold analysis...")
    kfold_results = compare_models_with_detailed_kfold()
    print("\nDetailed k-fold analysis completed!")
