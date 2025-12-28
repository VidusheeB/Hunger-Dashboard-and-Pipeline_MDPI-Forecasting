import os
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
import numpy as np
import xgboost as xgb

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends_scaled.csv"
MODELS_DIR = "county_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_global_model():
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
    # Use SNAP application rates as target variable instead of absolute numbers
    y = df["SNAP_Application_Rate"]
    # Drop rows with missing values in any feature or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Ensure target values are positive (SNAP application rates cannot be negative)
    y = y.clip(lower=0)

    # Fit XGBoost model with constraints to prevent negative predictions
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Check for negative predictions on training data
    train_predictions = model.predict(X)
    negative_count = np.sum(train_predictions < 0)
    if negative_count > 0:
        print(f"⚠️  Warning: {negative_count} negative predictions in training data")
        print("   This will be handled by clipping predictions to 0 in production")
    
    r2 = model.score(X, y)
    print(f"Trained XGBoost model (predicting rates) | R^2: {r2:.3f}")

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Feature importance
    feature_importance = model.feature_importances_
    print("\n=== FEATURE IMPORTANCE ===")
    for feature, importance in zip(feature_cols, feature_importance):
        print(f"{feature}: {importance:.4f}")

    # Model info
    print(f"\n=== MODEL INFO ===")
    print(f"Number of estimators: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Model type: XGBoost (predicting SNAP application rates)")
    print(f"Target variable: SNAP_Application_Rate (applications per population)")

    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": feature_cols, "type": "xgboost"}, f)

if __name__ == "__main__":
    train_global_model()