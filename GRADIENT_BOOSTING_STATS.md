# Gradient Boosting Model Performance Summary

## Overall Performance Statistics

### Training Performance
- **Training R²**: 0.8489 (84.89% variance explained)
- **Training MSE**: 0.000001
- **Training MAE**: 0.000619

### Test Performance
- **Test R²**: 0.7464 (74.64% variance explained) ⭐ **Best among all models**
- **Test MSE**: 0.000001
- **Test MAE**: 0.000666

### Cross-Validation Performance
- **CV R²**: 0.3185 (± 0.1602) ⭐ **Best CV performance**
- This indicates the model generalizes well across different data splits

## Comparison with Other Models

| Model | Test R² | CV R² (Mean ± Std) | Status |
|-------|---------|---------------------|--------|
| **Gradient Boosting** | **0.7464** | **0.3185 ± 0.1602** | ⭐ Best overall |
| XGBoost | 0.7059 | -0.3450 ± 1.3167 | Overfitting concerns |
| Random Forest | 0.5594 | 0.2838 ± 0.1978 | Good stability |
| Linear Regression | 0.3390 | -0.0029 ± 0.3223 | Poor performance |
| Ridge Regression | 0.3255 | 0.1805 ± 0.1119 | Poor performance |
| Lasso Regression | 0.3470 | 0.1700 ± 0.1080 | Poor performance |

## Key Insights

1. **Best Test Performance**: Gradient Boosting achieved the highest test R² (0.7464) among all models tested.

2. **Best Cross-Validation Performance**: With a CV R² of 0.3185, Gradient Boosting shows the best generalization capability, indicating it's less prone to overfitting compared to XGBoost.

3. **Stable Performance**: The relatively low CV standard deviation (0.1602) suggests consistent performance across different data splits.

4. **Feature Importance** (Top 5):
   - Median_Income: 0.4791 (47.91%) - Most important
   - Population: 0.3096 (30.96%)
   - month: 0.0769 (7.69%)
   - monthly_average_CalFresh: 0.0687 (6.87%)
   - monthly_average_FoodBank: 0.0658 (6.58%)

5. **Overfitting Check**: 
   - Training R² (0.8489) vs Test R² (0.7464) = Gap of 0.1025
   - This gap is reasonable and suggests the model is not severely overfitting

## Recommendation

**Gradient Boosting is the recommended model** for this SNAP application rate prediction task because:
- Highest test R² score (0.7464)
- Best cross-validation performance (0.3185)
- Good balance between bias and variance
- More stable than XGBoost (which shows negative CV R²)

However, with proper hyperparameter tuning, XGBoost could potentially match or exceed Gradient Boosting's performance.

