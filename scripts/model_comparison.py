import os
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
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

def train_and_evaluate_model(model, model_name, X, y, feature_cols):
    """Train and evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Ensure predictions are non-negative
    y_pred_train = np.clip(y_pred_train, 0, None)
    y_pred_test = np.clip(y_pred_test, 0, None)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_)
    
    results = {
        'model_name': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'model': model
    }
    
    # Print results
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training MAE: {train_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    if feature_importance is not None:
        print("\n=== FEATURE IMPORTANCE ===")
        feature_importance_pairs = list(zip(feature_cols, feature_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance_pairs:
            print(f"{feature}: {importance:.4f}")
    
    return results

def compare_models():
    """Compare different machine learning models."""
    print("Loading and preparing data...")
    X, y, feature_cols = load_and_prepare_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Target range: {y.min():.6f} - {y.max():.6f}")
    
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.01),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    # Train and evaluate all models
    all_results = []
    
    for model_name, model in models.items():
        try:
            results = train_and_evaluate_model(model, model_name, X, y, feature_cols)
            all_results.append(results)
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': result['model_name'],
            'Train_R2': result['train_r2'],
            'Test_R2': result['test_r2'],
            'Train_MSE': result['train_mse'],
            'Test_MSE': result['test_mse'],
            'Train_MAE': result['train_mae'],
            'Test_MAE': result['test_mae'],
            'CV_R2_Mean': result['cv_r2_mean'],
            'CV_R2_Std': result['cv_r2_std']
        }
        for result in all_results
    ])
    
    # Sort by test R² score
    comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
    
    # Save results
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_results.csv'), index=False)
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(comparison_df.round(4))
    
    # Save the best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = next(r for r in all_results if r['model_name'] == best_model_name)
    
    with open(os.path.join(RESULTS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump({
            'model': best_result['model'],
            'features': feature_cols,
            'type': best_model_name.lower().replace(' ', '_'),
            'results': best_result
        }, f)
    
    print(f"\nBest model: {best_model_name}")
    print(f"Test R²: {best_result['test_r2']:.4f}")
    print(f"Cross-validation R²: {best_result['cv_r2_mean']:.4f} (+/- {best_result['cv_r2_std'] * 2:.4f})")
    print(f"Best model saved to: {RESULTS_DIR}/best_model.pkl")
    
    # Create detailed report
    create_detailed_report(all_results, comparison_df)
    
    return comparison_df, all_results

def create_detailed_report(all_results, comparison_df):
    """Create a detailed markdown report of the model comparison."""
    
    report_path = os.path.join(RESULTS_DIR, 'MODEL_COMPARISON_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report compares different machine learning models for predicting SNAP application rates.\n\n")
        
        f.write("## Dataset Information\n\n")
        f.write(f"- **Total samples**: {len(comparison_df)}\n")
        f.write(f"- **Features**: Population, Median_Income, monthly trends (CalFresh, FoodBank), month\n")
        f.write(f"- **Target**: SNAP_Application_Rate\n\n")
        
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model | Test R² | Train R² | Test MSE | Test MAE | CV R² (Mean ± Std) |\n")
        f.write("|-------|---------|----------|----------|----------|---------------------|\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['Model']} | {row['Test_R2']:.4f} | {row['Train_R2']:.4f} | {row['Test_MSE']:.6f} | {row['Test_MAE']:.6f} | {row['CV_R2_Mean']:.4f} ± {row['CV_R2_Std']:.4f} |\n")
        
        f.write("\n## Model Details\n\n")
        
        for result in all_results:
            f.write(f"### {result['model_name']}\n\n")
            f.write(f"- **Training R²**: {result['train_r2']:.4f}\n")
            f.write(f"- **Test R²**: {result['test_r2']:.4f}\n")
            f.write(f"- **Training MSE**: {result['train_mse']:.6f}\n")
            f.write(f"- **Test MSE**: {result['test_mse']:.6f}\n")
            f.write(f"- **Training MAE**: {result['train_mae']:.6f}\n")
            f.write(f"- **Test MAE**: {result['test_mae']:.6f}\n")
            f.write(f"- **Cross-validation R²**: {result['cv_r2_mean']:.4f} (± {result['cv_r2_std']:.4f})\n\n")
            
            if result['feature_importance'] is not None:
                f.write("**Top 5 Most Important Features:**\n")
                feature_importance_pairs = list(zip(['Population', 'Median_Income', 'monthly_average_FoodBank', 'monthly_average_CalFresh', 'month'], result['feature_importance']))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                for feature, importance in feature_importance_pairs[:5]:
                    f.write(f"- {feature}: {importance:.4f}\n")
                f.write("\n")
        
        f.write("## Recommendations\n\n")
        best_model = comparison_df.iloc[0]
        f.write(f"**Best performing model**: {best_model['Model']}\n\n")
        f.write(f"- Test R²: {best_model['Test_R2']:.4f}\n")
        f.write(f"- Cross-validation R²: {best_model['CV_R2_Mean']:.4f} (± {best_model['CV_R2_Std']:.4f})\n\n")
        
        f.write("### Key Insights:\n\n")
        f.write("1. **Overfitting Check**: Models with large gaps between training and test R² scores may be overfitting.\n")
        f.write("2. **Cross-validation**: The CV scores provide a more robust estimate of model performance.\n")
        f.write("3. **Feature Importance**: Understanding which features drive predictions helps with model interpretability.\n")
        f.write("4. **Model Selection**: Choose based on test performance, cross-validation stability, and business requirements.\n\n")
    
    print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    print("Starting model comparison...")
    comparison_df, all_results = compare_models()
    print("\nModel comparison completed!")
