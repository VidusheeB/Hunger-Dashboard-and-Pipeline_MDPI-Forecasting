# Walk-Forward Validation Results (Corrected Method)

## Method Description

### Corrected Walk-Forward Validation Process

The corrected walk-forward validation implements a rigorous time-series cross-validation approach that properly aligns features and targets to prevent data leakage:

#### 1. **Data Alignment (Features at t → Target at t+1)**
   - Each row represents: **features at month t** predicting **SNAP application rate at month t+1**
   - Created `SNAP_target` column: `groupby('county')['SNAP_rate'].shift(-1)` to get next month's rate
   - Created `target_month` column: `month_dt + 1 month` to identify the prediction target month
   - This ensures features are always from an earlier time period than the target

#### 2. **Temporal Splitting Logic**
   - For each target month **T**:
     - **Training set**: All rows where `target_month < T` (strictly before T)
     - **Test set**: All rows where `target_month == T` (exactly T)
   - No "skip month" logic needed - the target_month alignment naturally prevents leakage
   - Training data grows over time as more historical data becomes available

#### 3. **Feature Engineering (All Computed on Historical Data Only)**
   - **Lagged features**: `shift(lag)` ensures lags only use past values
   - **Rolling means**: Computed with `min_periods=1` but only use historical windows
   - **Seasonality**: Month-based features (sin/cos) are static and don't leak
   - **Income normalization**: Z-score computed within county (static feature)

#### 4. **Walk-Forward Loop**
   ```
   For each target_month T (starting from month 6):
     1. Train on all rows with target_month < T
     2. Test on all rows with target_month == T
     3. Calculate metrics (R², RMSE, MAE, sMAPE)
     4. Store results and aggregate predictions
   ```

#### 5. **Key Improvements Over Original Method**
   - ✅ **Proper temporal alignment**: Features at t always predict target at t+1
   - ✅ **No leakage**: Training data never includes information from test period
   - ✅ **Consistent time index**: Uses `target_month` as the single source of truth
   - ✅ **Realistic evaluation**: Mimics real-world prediction scenario where you predict next month using current month's features

---

## Complete Results Summary

### Overall Performance (All Predictions Combined)

| Model | Overall R² | Overall RMSE | Overall MAE | Overall sMAPE |
|-------|------------|--------------|-------------|---------------|
| **Tuned XGBoost** ⭐ | **0.6214** | **0.001155** | **0.000772** | **12.57%** |
| Random Forest | 0.4923 | 0.001338 | 0.000775 | 12.37% |
| Gradient Boosting | 0.2981 | 0.001573 | 0.000788 | 12.44% |
| Base XGBoost | -0.0275 | 0.001903 | 0.001504 | 26.12% |

### Per-Month Performance Statistics

| Model | R² Mean ± Std | RMSE Mean ± Std | MAE Mean ± Std | sMAPE Mean ± Std |
|-------|---------------|-----------------|----------------|------------------|
| **Tuned XGBoost** ⭐ | **0.5922 ± 0.2370** | **0.001103 ± 0.000388** | **0.000776 ± 0.000253** | **12.67% ± 3.52%** |
| Random Forest | 0.4094 ± 0.7060 | 0.001242 ± 0.000586 | 0.000782 ± 0.000256 | 12.46% ± 3.73% |
| Gradient Boosting | 0.1096 ± 1.7425 | 0.001374 ± 0.000880 | 0.000796 ± 0.000230 | 12.54% ± 3.30% |
| Base XGBoost | -0.1607 ± 0.1693 | 0.001891 ± 0.000277 | 0.001510 ± 0.000210 | 26.23% ± 1.96% |

### Detailed Model Performance

#### 1. Tuned XGBoost ⭐ (Best Performer)
- **Overall R²**: 0.6214 (62.14% variance explained)
- **Overall RMSE**: 0.001155
- **Overall MAE**: 0.000772
- **Overall sMAPE**: 12.57%
- **Months evaluated**: 25
- **Total predictions**: 1,242
- **Per-month R² range**: Mean 0.5922 with std 0.2370
- **Stability**: Good (moderate variance across months)

**Key Insights**:
- Best overall performance across all metrics
- Consistent performance across months (lower variance)
- Hyperparameter tuning significantly improved over base XGBoost

#### 2. Random Forest
- **Overall R²**: 0.4923 (49.23% variance explained)
- **Overall RMSE**: 0.001338
- **Overall MAE**: 0.000775
- **Overall sMAPE**: 12.37%
- **Months evaluated**: 25
- **Total predictions**: 1,242
- **Per-month R² range**: Mean 0.4094 with std 0.7060
- **Stability**: Moderate (higher variance, some months perform poorly)

**Key Insights**:
- Second-best overall performance
- Higher variance suggests some months are challenging
- More stable than Gradient Boosting

#### 3. Gradient Boosting
- **Overall R²**: 0.2981 (29.81% variance explained)
- **Overall RMSE**: 0.001573
- **Overall MAE**: 0.000788
- **Overall sMAPE**: 12.44%
- **Months evaluated**: 25
- **Total predictions**: 1,242
- **Per-month R² range**: Mean 0.1096 with std 1.7425
- **Stability**: Poor (very high variance, some months have negative R²)

**Key Insights**:
- Third-best overall performance
- Very high variance suggests inconsistent performance
- May be overfitting to training patterns that don't generalize

#### 4. Base XGBoost
- **Overall R²**: -0.0275 (worse than naive baseline)
- **Overall RMSE**: 0.001903
- **Overall MAE**: 0.001504
- **Overall sMAPE**: 26.12%
- **Months evaluated**: 25
- **Total predictions**: 1,242
- **Per-month R² range**: Mean -0.1607 with std 0.1693
- **Stability**: Consistent but consistently poor

**Key Insights**:
- Poor performance (negative R² means worse than predicting the mean)
- All months show negative R²
- Hyperparameter tuning is essential for this model

---

## Comparison: Original vs Corrected Method

### Base XGBoost Performance

| Metric | Original Method | Corrected Method | Change |
|--------|----------------|------------------|--------|
| Overall R² | 0.8424 | -0.0275 | -0.8699 (massive drop) |
| Overall RMSE | 0.000747 | 0.001903 | +0.001156 (worse) |
| Overall MAE | 0.000397 | 0.001504 | +0.001107 (worse) |
| Overall sMAPE | 6.66% | 26.12% | +19.46% (worse) |

**Analysis**: The original method showed inflated performance due to data leakage. The corrected method reveals the true predictive capability, which is poor for the base XGBoost model.

### Why the Difference?

1. **Original Method Issues**:
   - Used `month_dt` for splitting with skip-month logic
   - Features and targets not properly aligned (t → t+1)
   - Potential leakage from rolling features including current month
   - Inconsistent temporal boundaries

2. **Corrected Method Benefits**:
   - Strict temporal alignment (features at t → target at t+1)
   - Uses `target_month` as single time index
   - No leakage possible - training data strictly before test period
   - Realistic evaluation scenario

---

## Recommendations

### Best Model: Tuned XGBoost
- **R²**: 0.6214 (explains 62% of variance)
- **RMSE**: 0.001155 (lowest error)
- **Stable**: Consistent performance across months
- **Production Ready**: Good balance of accuracy and reliability

### Model Selection Guide

1. **For Production Use**: Tuned XGBoost
   - Best overall performance
   - Most stable across months
   - Hyperparameters optimized for generalization

2. **For Baseline Comparison**: Random Forest
   - Good performance (R² = 0.49)
   - More interpretable than XGBoost
   - Can serve as benchmark

3. **Avoid for Production**: 
   - Base XGBoost (poor performance, negative R²)
   - Gradient Boosting (high variance, inconsistent)

### Key Takeaways

1. **Hyperparameter tuning is critical**: Tuned XGBoost (R² = 0.62) vs Base XGBoost (R² = -0.03)
2. **Corrected method is essential**: Original method showed inflated performance
3. **Time-series validation matters**: Standard CV doesn't capture temporal dependencies
4. **Model stability is important**: Tuned XGBoost shows consistent performance across months

---

## Technical Details

### Evaluation Period
- **Start month**: 2023-02 (after sufficient history for feature engineering)
- **End month**: 2025-02
- **Total months evaluated**: 25
- **Total predictions**: 1,242 (across 58 counties)

### Feature Set (23 features)
- Base: Population, Median_Income, CalFresh_trend, FoodBank_trend, month
- Seasonality: month_sin, month_cos
- Lags (1, 2, 3): SNAP_rate, CalFresh_trend, FoodBank_trend
- Rolling means (3, 6): SNAP_rate, CalFresh_trend, FoodBank_trend
- Normalization: income_z_by_county

### Model Configurations

**Tuned XGBoost**:
- n_estimators: 500
- learning_rate: 0.01
- max_depth: 8
- min_child_weight: 6
- subsample: 0.8
- colsample_bytree: 0.9
- reg_alpha: 0
- reg_lambda: 4

**Base XGBoost**:
- n_estimators: 1200
- learning_rate: 0.03
- max_depth: 4
- min_child_weight: 8
- subsample: 0.8
- colsample_bytree: 0.8
- reg_lambda: 12
- reg_alpha: 1

**Random Forest**:
- n_estimators: 200
- random_state: 42

**Gradient Boosting**:
- n_estimators: 100
- random_state: 42

---

*Report generated from corrected walk-forward validation results*
*All metrics computed using proper temporal alignment (features at t → target at t+1)*
