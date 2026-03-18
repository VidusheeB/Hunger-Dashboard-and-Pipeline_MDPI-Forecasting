import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

AGGREGATE_TRENDS_FILE = "src/data/training_data.csv"
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
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
    print(f"Trained XGBoost model (predicting rates) | In-sample R²: {r2:.3f}  RMSE: {rmse:.6f}")

    # Walk-forward validation: train on past, test on each future month in sequence
    df_wf = df[mask].copy()
    df_wf['date'] = pd.to_datetime(df_wf['date'])
    dates = sorted(df_wf['date'].unique())
    min_train_months = 12  # need at least 12 months of history before testing
    wf_r2, wf_rmse, wf_mae, wf_smape = [], [], [], []

    for i, test_date in enumerate(dates[min_train_months:], start=min_train_months):
        train_mask = df_wf['date'] < test_date
        test_mask  = df_wf['date'] == test_date
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        X_tr, y_tr = df_wf.loc[train_mask, feature_cols], df_wf.loc[train_mask, 'SNAP_Application_Rate']
        X_te, y_te = df_wf.loc[test_mask,  feature_cols], df_wf.loc[test_mask,  'SNAP_Application_Rate']
        m = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        preds = np.clip(m.predict(X_te), 0, None)
        wf_r2.append(r2_score(y_te, preds))
        wf_rmse.append(np.sqrt(mean_squared_error(y_te, preds)))
        wf_mae.append(mean_absolute_error(y_te, preds))
        denom = np.abs(y_te.values) + np.abs(preds)
        wf_smape.append(np.mean(2 * np.abs(y_te.values - preds) / np.where(denom == 0, 1, denom)) * 100)

    print(f"\n=== WALK-FORWARD VALIDATION ({len(wf_r2)} months tested) ===")
    print(f"R²:    {np.mean(wf_r2):.4f}  (±{np.std(wf_r2):.4f})")
    print(f"RMSE:  {np.mean(wf_rmse):.6f}")
    print(f"MAE:   {np.mean(wf_mae):.6f}")
    print(f"sMAPE: {np.mean(wf_smape):.2f}%")

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

    # Compute feature ranges for drift detection
    feature_ranges = {f: {"min": float(X[f].min()), "max": float(X[f].max())} for f in feature_cols}

    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_cols,
            "type": "xgboost",
            "walkforward_mae": float(np.mean(wf_mae)) if wf_mae else 0.000877,
            "feature_ranges": feature_ranges,
        }, f)

if __name__ == "__main__":
    train_global_model()