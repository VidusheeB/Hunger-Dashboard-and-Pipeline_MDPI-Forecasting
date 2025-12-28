# XGBoost Hyperparameter Tuning Report

Generated on: 2025-11-12 17:44:05

## Best Hyperparameters

- **colsample_bytree**: 0.9
- **learning_rate**: 0.01
- **max_depth**: 8
- **min_child_weight**: 6
- **n_estimators**: 500
- **reg_alpha**: 0
- **reg_lambda**: 4
- **subsample**: 0.7999999999999999

**Best CV R² Score**: 0.643264

## Model Performance

### Tuned XGBoost

**Training Metrics:**
- R²: 0.737117
- MSE: 0.000002
- RMSE: 0.001318
- MAE: 0.000537
- MAPE: 8.627286
- sMAPE: 8.400397
- MASE: 0.228222

**Test Metrics:**
- R²: 0.674811
- MSE: 0.000001
- RMSE: 0.001075
- MAE: 0.000683
- MAPE: 11.570133
- sMAPE: 10.909896
- MASE: 0.326523

**Cross-Validation:**
- CV R²: 0.258770 ± 0.159234
- CV RMSE: 0.001986 ± 0.000727
- CV MAE: 0.001219 ± 0.000197

### Current XGBoost

**Training Metrics:**
- R²: 0.920565
- MSE: 0.000001
- RMSE: 0.000725
- MAE: 0.000503
- MAPE: 8.414819
- sMAPE: 8.202724
- MASE: 0.213670

**Test Metrics:**
- R²: 0.884732
- MSE: 0.000000
- RMSE: 0.000640
- MAE: 0.000481
- MAPE: 8.352517
- sMAPE: 8.127593
- MASE: 0.229911

**Cross-Validation:**
- CV R²: -0.193065 ± 1.046732
- CV RMSE: 0.002218 ± 0.000800
- CV MAE: 0.001208 ± 0.000172
