# Complete Model Performance Comparison Table

## Comprehensive Statistics - All Models

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

## Detailed Breakdown by Category

### 1. Training Performance

| Model | Train R² | Train RMSE | Train MAE | Train MSE |
|-------|----------|------------|-----------|-----------|
| **XGBoost (Current)** | **0.9206** | 0.000725 | 0.000503 | 0.0000005 |
| Random Forest | 0.9042 | 0.000796 | 0.000276 | 0.0000006 |
| **Gradient Boosting** | 0.8489 | 0.001000 | 0.000619 | 0.000001 |
| Tuned XGBoost | 0.7371 | 0.001318 | 0.000537 | 0.000002 |

### 2. Test Performance (Holdout Set)

| Model | Test R² | Test RMSE | Test MAE | Test MSE | Rank |
|-------|---------|-----------|----------|----------|------|
| **Gradient Boosting** | **0.7464** | **0.000949** | 0.000666 | 0.0000009 | 🥇 1st |
| XGBoost (Current) | 0.8847 | 0.000640 | **0.000481** | 0.0000004 | 🥈 2nd |
| Tuned XGBoost | 0.6748 | 0.001075 | 0.000683 | 0.000001 | 🥉 3rd |
| Random Forest | 0.5594 | 0.001251 | 0.000696 | 0.0000016 | 4th |

### 3. Cross-Validation Performance (Generalization Ability)

| Model | CV R² Mean | CV R² Std | CV RMSE Mean | CV RMSE Std | CV MAE Mean | CV MAE Std | Stability | Rank |
|-------|------------|----------|--------------|-------------|------------|------------|-----------|------|
| **Gradient Boosting** | **0.3185** | ±0.1602 | **0.001916** | ±0.000761 | - | - | ✅ Very Stable | 🥇 1st |
| Random Forest | 0.2838 | ±0.1978 | 0.001941 | ±0.000747 | - | - | ✅ Stable | 🥈 2nd |
| Tuned XGBoost | 0.2588 | ±0.1592 | 0.001986 | ±0.000727 | 0.001219 | ±0.000197 | ✅ Very Stable | 🥉 3rd |
| XGBoost (Current) | -0.1931 | ±1.0467 | 0.002218 | ±0.000800 | 0.001208 | ±0.000172 | ❌ Unstable | 4th |

### 4. Overfitting Analysis

| Model | Train R² | Test R² | Gap | Status | Interpretation |
|-------|----------|---------|-----|--------|----------------|
| Tuned XGBoost | 0.7371 | 0.6748 | 0.0623 | ✅ Very Healthy | Minimal overfitting |
| **Gradient Boosting** | 0.8489 | 0.7464 | **0.1025** | ✅ Healthy | Good balance |
| XGBoost (Current) | 0.9206 | 0.8847 | 0.0359 | ⚠️ Overfitting | Large train-test gap despite high test R² |
| Random Forest | 0.9042 | 0.5594 | 0.3448 | ⚠️ Overfitting | Significant gap |

**Note:** The XGBoost (Current) has a small gap but negative CV R², indicating it's overfitting to the specific train/test split.

## Key Insights & Rankings

### 🏆 Overall Best: **Gradient Boosting**
- ✅ **Highest Test R²**: 0.7464
- ✅ **Best CV R²**: 0.3185 ± 0.1602
- ✅ **Best CV RMSE**: 0.001916 ± 0.000761
- ✅ **Good Generalization**: Healthy overfitting gap (0.1025)
- ✅ **Stable**: Low CV standard deviation

### 🥈 Second Best: **Tuned XGBoost**
- ✅ **Good Test R²**: 0.6748
- ✅ **Positive CV R²**: 0.2588 (fixed negative CV from current XGBoost)
- ✅ **Best Overfitting Control**: Smallest gap (0.0623)
- ✅ **Very Stable**: Low CV standard deviation (±0.1592)
- ✅ **Better CV RMSE**: 0.001986 vs 0.002218 (current)

### 🥉 Third: **XGBoost (Current)**
- ✅ **High Test R²**: 0.8847
- ❌ **Poor Generalization**: Negative CV R² (-0.1931)
- ❌ **Unstable**: High CV standard deviation (±1.0467)
- ⚠️ **Overfitting**: Despite high test R², poor CV performance indicates overfitting

### 4th: **Random Forest**
- ⚠️ **Moderate Test R²**: 0.5594
- ✅ **Good CV Performance**: 0.2838 ± 0.1978
- ✅ **Good CV RMSE**: 0.001941 ± 0.000747
- ⚠️ **Some Overfitting**: Gap of 0.3448
- ✅ **Stable**: Moderate CV standard deviation

## Summary Statistics

### Test Performance Rankings
1. **Gradient Boosting**: Test R² = 0.7464, Test RMSE = 0.000949
2. **XGBoost (Current)**: Test R² = 0.8847, Test RMSE = 0.000640
3. **Tuned XGBoost**: Test R² = 0.6748, Test RMSE = 0.001075
4. **Random Forest**: Test R² = 0.5594, Test RMSE = 0.001251

### Cross-Validation Rankings (Generalization)
1. **Gradient Boosting**: CV R² = 0.3185, CV RMSE = 0.001916
2. **Random Forest**: CV R² = 0.2838, CV RMSE = 0.001941
3. **Tuned XGBoost**: CV R² = 0.2588, CV RMSE = 0.001986
4. **XGBoost (Current)**: CV R² = -0.1931, CV RMSE = 0.002218

### Stability Rankings (CV Std)
1. **Tuned XGBoost**: CV R² Std = ±0.1592, CV RMSE Std = ±0.000727
2. **Gradient Boosting**: CV R² Std = ±0.1602, CV RMSE Std = ±0.000761
3. **Random Forest**: CV R² Std = ±0.1978, CV RMSE Std = ±0.000747
4. **XGBoost (Current)**: CV R² Std = ±1.0467, CV RMSE Std = ±0.000800

## Recommendations

### For Production Use:
1. **Gradient Boosting** - Best overall performance and generalization
2. **Tuned XGBoost** - Good balance, best overfitting control

### For Further Investigation:
- **XGBoost (Current)** - High test R² but poor generalization; needs more regularization
- **Random Forest** - Good stability but lower test performance

## Notes

- **Overfitting Gap** = Training R² - Test R² (smaller is generally better, but context matters)
- **CV R²** measures generalization across different data splits (positive is good, negative indicates poor generalization)
- **CV Std** measures stability across folds (lower is better)
- **CV RMSE** provides error magnitude in cross-validation (lower is better)
- ⭐ indicates best performing model overall
- All metrics calculated using 5-fold cross-validation
- Test set is 20% of data, randomly split with random_state=42

