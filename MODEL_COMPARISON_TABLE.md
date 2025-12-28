# Model Performance Comparison Table

## Comprehensive Statistics - Complete Table

| Metric | XGBoost (Current) | Random Forest | Tuned XGBoost | Gradient Boosting ⭐ |
|--------|-------------------|--------------|---------------|---------------------|
| **Training R²** | 0.9206 | 0.9042 | 0.7371 | 0.8489 |
| **Test R²** | 0.8847 | 0.5594 | 0.6748 | **0.7464** |
| **Test RMSE** | 0.000640 | 0.001251 | 0.001075 | 0.000949 |
| **Test MAE** | 0.000481 | 0.000696 | 0.000683 | 0.000666 |
| **CV R² (Mean)** | -0.1931 | 0.2838 | 0.2588 | **0.3185** |
| **CV R² (Std)** | ±1.0467 | ±0.1978 | ±0.1592 | ±0.1602 |
| **CV RMSE (Mean)** | 0.002218 | 0.001941 | 0.001986 | **0.001916** |
| **CV RMSE (Std)** | ±0.000800 | ±0.000747 | ±0.000727 | ±0.000761 |
| **CV MAE (Mean)** | 0.001208 | - | 0.001219 | - |
| **CV MAE (Std)** | ±0.000172 | - | ±0.000197 | - |
| **Training MSE** | 0.0000005 | 0.0000006 | 0.000002 | 0.000001 |
| **Test MSE** | 0.0000004 | 0.0000016 | 0.000001 | 0.0000009 |
| **Training MAE** | 0.000503 | 0.000276 | 0.000537 | 0.000619 |
| **Overfitting Gap** | 0.0359 | 0.3448 | 0.0623 | **0.1025** |
| **Generalization** | ❌ Poor | ✅ Good | ✅ Good | ✅ **Best** |

## Key Metrics Breakdown

### Test Performance (Higher is Better for R², Lower is Better for Errors)

| Model | Test R² | Test RMSE | Test MAE | Rank |
|-------|---------|-----------|----------|------|
| **Gradient Boosting** | **0.7464** | **0.000949** | 0.000666 | 🥇 1st |
| XGBoost (Current) | 0.7059 | 0.001022 | **0.000641** | 🥈 2nd |
| Tuned XGBoost | 0.6748 | 0.001075 | 0.000683 | 🥉 3rd |
| Random Forest | 0.5594 | 0.001251 | 0.000696 | 4th |

### Cross-Validation Performance (Generalization Ability)

| Model | CV R² Mean | CV R² Std | CV RMSE | CV RMSE Std | Stability | Rank |
|-------|------------|----------|---------|-------------|-----------|------|
| **Gradient Boosting** | **0.3185** | ±0.1602 | **0.001916** | ±0.000761 | ✅ Very Stable | 🥇 1st |
| Random Forest | 0.2838 | ±0.1978 | 0.001941 | ±0.000747 | ✅ Stable | 🥈 2nd |
| Tuned XGBoost | 0.2588 | ±0.1592 | 0.001986 | ±0.000727 | ✅ Very Stable | 🥉 3rd |
| XGBoost (Current) | -0.3450 | ±1.3167 | 0.002218 | ±0.000800 | ❌ Unstable | 4th |

### Overfitting Analysis

| Model | Train R² | Test R² | Gap | Status |
|-------|----------|---------|-----|--------|
| **Gradient Boosting** | 0.8489 | 0.7464 | 0.1025 | ✅ Healthy |
| Tuned XGBoost | 0.7371 | 0.6748 | 0.0623 | ✅ Very Healthy |
| Random Forest | 0.9042 | 0.5594 | 0.3448 | ⚠️ Some Overfitting |
| XGBoost (Current) | 0.9278 | 0.7059 | 0.2219 | ⚠️ Overfitting |

## Summary & Recommendations

### 🏆 Best Overall: **Gradient Boosting**
- **Highest Test R²**: 0.7464
- **Best CV Performance**: 0.3185 ± 0.1602
- **Good Generalization**: Low overfitting gap (0.1025)
- **Stable**: Low CV standard deviation

### 🥈 Second Best: **Tuned XGBoost**
- **Good Test R²**: 0.6748
- **Positive CV R²**: 0.2588 (fixed negative CV from current XGBoost)
- **Best Overfitting Control**: Smallest gap (0.0623)
- **Very Stable**: Low CV standard deviation (±0.1592)

### 🥉 Third: **XGBoost (Current)**
- **Good Test R²**: 0.7059
- **Poor Generalization**: Negative CV R² (-0.3450)
- **Unstable**: High CV standard deviation (±1.3167)
- **Overfitting**: Large gap between train/test

### 4th: **Random Forest**
- **Moderate Test R²**: 0.5594
- **Good CV Performance**: 0.2838 ± 0.1978
- **Some Overfitting**: Gap of 0.3448
- **Stable**: Moderate CV standard deviation

## Notes

- **Overfitting Gap** = Training R² - Test R² (smaller is better)
- **CV R²** measures generalization across different data splits (positive is good)
- **CV Std** measures stability (lower is better)
- ⭐ indicates best performing model overall



## Walk-Forward (Time-Series) Backtest Summary

| Metric | XGBoost | Gradient Boosting |
|--------|---------|-------------------|
| Overall R² | 0.8424 | 0.1001 |
| Overall RMSE | 0.000747 | 0.001784 |
| Overall MAE | 0.000397 | 0.000456 |
| sMAPE | 6.66% | 7.15% |
| Per-month R² (mean ± std) | 0.8386 ± 0.2853 | 0.1320 ± 3.8026 |
| Per-month RMSE (mean ± std) | 0.000620 ± 0.000447 | 0.000889 ± 0.001637 |
| Months evaluated | 26 | 25 |
| Predictions total | 1313 | 1248 |


## Walk-Forward Backtest (Additional Models)

| Metric | Random Forest | Linear Regression | Tuned XGBoost |
|--------|----------------|-------------------|---------------|
| Overall R² | 0.5348 | 1.0000 | 0.8493 |
| Overall RMSE | 0.001283 | 0.000000 | 0.000730 |
| Overall MAE | 0.000432 | 0.000000 | 0.000417 |
| sMAPE | 6.85% | 0.00% | 7.04% |
| Per-month R² (mean ± std) | 0.5377 ± 1.6401 | 1.0000 ± 0.0000 | 0.8482 ± 0.2004 |
| Per-month RMSE (mean ± std) | 0.000809 ± 0.001060 | 0.000000 ± 0.000000 | 0.000640 ± 0.000378 |
| Months evaluated | 25 | 25 | 25 |
| Predictions total | 1248 | 1248 | 1248 |



## Walk-Forward (Corrected target_month split)

| Metric | XGBoost (Base) | Random Forest | Gradient Boosting | Tuned XGBoost |
|--------|----------------|---------------|-------------------|---------------|
| Overall R² | -0.0275 | 0.4923 | 0.2981 | 0.6214 |
| Overall RMSE | 0.001903 | 0.001338 | 0.001573 | 0.001155 |
| Overall MAE | 0.001504 | 0.000775 | 0.000788 | 0.000772 |
| sMAPE | 26.12% | 12.37% | 12.44% | 12.57% |
| Per-month R² (mean ± std) | -0.1607 ± 0.1693 | 0.4094 ± 0.7060 | 0.1096 ± 1.7425 | 0.5922 ± 0.2370 |
| Per-month RMSE (mean ± std) | 0.001891 ± 0.000277 | 0.001242 ± 0.000586 | 0.001374 ± 0.000880 | 0.001103 ± 0.000388 |
| Months evaluated | 25 | 25 | 25 | 25 |
| Predictions total | 1242 | 1242 | 1242 | 1242 |


## Walk-Forward Validation (Corrected Method) - Complete Results

### Method Overview
The corrected walk-forward validation uses proper temporal alignment:
- **Each row**: Features at month t → Target (SNAP rate) at month t+1
- **Time index**: `target_month` (the month being predicted)
- **Splitting**: For target month T, train on `target_month < T`, test on `target_month == T`
- **No leakage**: Training data strictly before test period

### Complete Performance Comparison

| Model | Overall R² | Overall RMSE | Overall MAE | Overall sMAPE | Months | Predictions |
|-------|------------|--------------|-------------|--------------|--------|-------------|
| **Tuned XGBoost** ⭐ | **0.6214** | **0.001155** | **0.000772** | **12.57%** | 25 | 1,242 |
| Random Forest | 0.4923 | 0.001338 | 0.000775 | 12.37% | 25 | 1,242 |
| Gradient Boosting | 0.2981 | 0.001573 | 0.000788 | 12.44% | 25 | 1,242 |
| Base XGBoost | -0.0275 | 0.001903 | 0.001504 | 26.12% | 25 | 1,242 |

### Per-Month Statistics

| Model | R² Mean ± Std | RMSE Mean ± Std | MAE Mean ± Std | sMAPE Mean ± Std |
|-------|---------------|-----------------|----------------|------------------|
| **Tuned XGBoost** ⭐ | **0.5922 ± 0.2370** | **0.001103 ± 0.000388** | **0.000776 ± 0.000253** | **12.67% ± 3.52%** |
| Random Forest | 0.4094 ± 0.7060 | 0.001242 ± 0.000586 | 0.000782 ± 0.000256 | 12.46% ± 3.73% |
| Gradient Boosting | 0.1096 ± 1.7425 | 0.001374 ± 0.000880 | 0.000796 ± 0.000230 | 12.54% ± 3.30% |
| Base XGBoost | -0.1607 ± 0.1693 | 0.001891 ± 0.000277 | 0.001510 ± 0.000210 | 26.23% ± 1.96% |

### Key Findings

1. **Tuned XGBoost is the clear winner** with R² = 0.6214 and lowest errors
2. **Hyperparameter tuning is critical**: Base XGBoost (R² = -0.03) vs Tuned (R² = 0.62)
3. **Random Forest provides solid baseline** performance (R² = 0.49)
4. **Gradient Boosting shows high variance** (std = 1.74) indicating instability
5. **All models evaluated on same 25 months** (2023-02 to 2025-02) for fair comparison
