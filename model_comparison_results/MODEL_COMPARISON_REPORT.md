# Model Comparison Report

Generated on: 2025-09-11 13:02:34

## Summary

This report compares different machine learning models for predicting SNAP application rates.

## Dataset Information

- **Total samples**: 8
- **Features**: Population, Median_Income, monthly trends (CalFresh, FoodBank), month
- **Target**: SNAP_Application_Rate

## Model Performance Comparison

| Model | Test R² | Train R² | Test MSE | Test MAE | CV R² (Mean ± Std) |
|-------|---------|----------|----------|----------|---------------------|
| Gradient Boosting | 0.7464 | 0.8489 | 0.000001 | 0.000666 | 0.3185 ± 0.1602 |
| XGBoost | 0.7059 | 0.9278 | 0.000001 | 0.000641 | -0.3450 ± 1.3167 |
| Random Forest | 0.5594 | 0.9042 | 0.000002 | 0.000696 | 0.2838 ± 0.1978 |
| Lasso Regression | 0.3470 | 0.2246 | 0.000002 | 0.001129 | 0.1700 ± 0.1080 |
| Linear Regression | 0.3390 | 0.2551 | 0.000002 | 0.001157 | -0.0029 ± 0.3223 |
| Ridge Regression | 0.3255 | 0.2446 | 0.000002 | 0.001163 | 0.1805 ± 0.1119 |
| Support Vector Regression | -216.2832 | -115.8482 | 0.000772 | 0.027716 | -169.3305 ± 96.0960 |
| Neural Network (MLP) | -759050528.9759 | -310550549.2400 | 2695.910668 | 19.137718 | -2224109113.6683 ± 1921236829.0173 |

## Model Details

### Linear Regression

- **Training R²**: 0.2551
- **Test R²**: 0.3390
- **Training MSE**: 0.000005
- **Test MSE**: 0.000002
- **Training MAE**: 0.001228
- **Test MAE**: 0.001157
- **Cross-validation R²**: -0.0029 (± 0.3223)

**Top 5 Most Important Features:**
- monthly_average_FoodBank: 0.2577
- monthly_average_CalFresh: 0.2365
- month: 0.0001
- Median_Income: 0.0000
- Population: 0.0000

### Ridge Regression

- **Training R²**: 0.2446
- **Test R²**: 0.3255
- **Training MSE**: 0.000005
- **Test MSE**: 0.000002
- **Training MAE**: 0.001240
- **Test MAE**: 0.001163
- **Cross-validation R²**: 0.1805 (± 0.1119)

**Top 5 Most Important Features:**
- monthly_average_FoodBank: 0.0010
- monthly_average_CalFresh: 0.0006
- month: 0.0001
- Median_Income: 0.0000
- Population: 0.0000

### Lasso Regression

- **Training R²**: 0.2246
- **Test R²**: 0.3470
- **Training MSE**: 0.000005
- **Test MSE**: 0.000002
- **Training MAE**: 0.001266
- **Test MAE**: 0.001129
- **Cross-validation R²**: 0.1700 (± 0.1080)

**Top 5 Most Important Features:**
- Median_Income: 0.0000
- Population: 0.0000
- monthly_average_FoodBank: 0.0000
- monthly_average_CalFresh: 0.0000
- month: 0.0000

### Support Vector Regression

- **Training R²**: -115.8482
- **Test R²**: -216.2832
- **Training MSE**: 0.000773
- **Test MSE**: 0.000772
- **Training MAE**: 0.027722
- **Test MAE**: 0.027716
- **Cross-validation R²**: -169.3305 (± 96.0960)

### Random Forest

- **Training R²**: 0.9042
- **Test R²**: 0.5594
- **Training MSE**: 0.000001
- **Test MSE**: 0.000002
- **Training MAE**: 0.000276
- **Test MAE**: 0.000696
- **Cross-validation R²**: 0.2838 (± 0.1978)

**Top 5 Most Important Features:**
- Median_Income: 0.4342
- monthly_average_FoodBank: 0.1632
- Population: 0.1550
- monthly_average_CalFresh: 0.1529
- month: 0.0948

### Gradient Boosting

- **Training R²**: 0.8489
- **Test R²**: 0.7464
- **Training MSE**: 0.000001
- **Test MSE**: 0.000001
- **Training MAE**: 0.000619
- **Test MAE**: 0.000666
- **Cross-validation R²**: 0.3185 (± 0.1602)

**Top 5 Most Important Features:**
- Median_Income: 0.4791
- Population: 0.3096
- month: 0.0769
- monthly_average_CalFresh: 0.0687
- monthly_average_FoodBank: 0.0658

### Neural Network (MLP)

- **Training R²**: -310550549.2400
- **Test R²**: -759050528.9759
- **Training MSE**: 2053.397078
- **Test MSE**: 2695.910668
- **Training MAE**: 17.715884
- **Test MAE**: 19.137718
- **Cross-validation R²**: -2224109113.6683 (± 1921236829.0173)

### XGBoost

- **Training R²**: 0.9278
- **Test R²**: 0.7059
- **Training MSE**: 0.000000
- **Test MSE**: 0.000001
- **Training MAE**: 0.000490
- **Test MAE**: 0.000641
- **Cross-validation R²**: -0.3450 (± 1.3167)

**Top 5 Most Important Features:**
- Median_Income: 0.3344
- Population: 0.2575
- monthly_average_CalFresh: 0.2451
- month: 0.0887
- monthly_average_FoodBank: 0.0743

## Recommendations

**Best performing model**: Gradient Boosting

- Test R²: 0.7464
- Cross-validation R²: 0.3185 (± 0.1602)

### Key Insights:

1. **Overfitting Check**: Models with large gaps between training and test R² scores may be overfitting.
2. **Cross-validation**: The CV scores provide a more robust estimate of model performance.
3. **Feature Importance**: Understanding which features drive predictions helps with model interpretability.
4. **Model Selection**: Choose based on test performance, cross-validation stability, and business requirements.

