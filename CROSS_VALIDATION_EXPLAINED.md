# Cross-Validation: A Comprehensive Explanation

## Table of Contents
1. [What is Cross-Validation?](#what-is-cross-validation)
2. [Why Use Cross-Validation?](#why-use-cross-validation)
3. [Types of Cross-Validation](#types-of-cross-validation)
4. [Standard K-Fold Cross-Validation](#standard-k-fold-cross-validation)
5. [Time-Series Cross-Validation](#time-series-cross-validation)
6. [Walk-Forward Validation (Our Implementation)](#walk-forward-validation-our-implementation)
7. [Comparison of Methods Used in This Project](#comparison-of-methods-used-in-this-project)
8. [Detailed Step-by-Step Process](#detailed-step-by-step-process)

---

## What is Cross-Validation?

**Cross-validation** is a statistical method used to assess how well a machine learning model generalizes to an independent dataset. Instead of using a single train-test split, cross-validation:

1. **Divides the data** into multiple subsets (folds)
2. **Trains the model** on some folds
3. **Tests the model** on the remaining fold(s)
4. **Repeats this process** multiple times with different train/test combinations
5. **Aggregates the results** to get a more robust estimate of model performance

### Key Concept: Generalization

The goal is to estimate how well your model will perform on **new, unseen data** - not just data it was trained on. Cross-validation helps prevent:
- **Overfitting**: Model memorizes training data but fails on new data
- **Underfitting**: Model is too simple to capture patterns
- **Optimistic bias**: Single train-test split might be "lucky" or "unlucky"

---

## Why Use Cross-Validation?

### Problems with Single Train-Test Split

```
┌─────────────────────────────────────┐
│         All Data                     │
│  ┌──────────────┬─────────────────┐  │
│  │   Training   │     Test        │  │
│  │   (80%)      │     (20%)       │  │
│  └──────────────┴─────────────────┘  │
└─────────────────────────────────────┘
```

**Issues:**
1. **High variance**: Results depend heavily on which 20% is selected
2. **Limited data**: Only 20% of data used for evaluation
3. **No stability measure**: Can't assess consistency across different splits
4. **Temporal issues**: For time-series, random split destroys temporal order

### Benefits of Cross-Validation

1. **More robust estimates**: Uses all data for both training and testing
2. **Variance reduction**: Multiple evaluations reduce impact of "lucky" splits
3. **Stability assessment**: Can measure consistency (standard deviation)
4. **Better data utilization**: Every data point is used for both training and testing
5. **Hyperparameter tuning**: Can compare different model configurations fairly

---

## Types of Cross-Validation

### 1. K-Fold Cross-Validation (Standard)

**How it works:**
- Data is randomly shuffled
- Divided into K equal-sized folds
- Model trained K times, each time using K-1 folds for training and 1 fold for testing
- Results averaged across all K iterations

**Example with K=5:**
```
Iteration 1: Train on Folds 2,3,4,5 → Test on Fold 1
Iteration 2: Train on Folds 1,3,4,5 → Test on Fold 2
Iteration 3: Train on Folds 1,2,4,5 → Test on Fold 3
Iteration 4: Train on Folds 1,2,3,5 → Test on Fold 4
Iteration 5: Train on Folds 1,2,3,4 → Test on Fold 5
```

**Pros:**
- ✅ Uses all data efficiently
- ✅ Good for independent, identically distributed (IID) data
- ✅ Provides variance estimate

**Cons:**
- ❌ Assumes data is IID (independent and identically distributed)
- ❌ **Breaks temporal order** - not suitable for time-series
- ❌ Can leak future information into past predictions

### 2. Time-Series Cross-Validation

**How it works:**
- Respects temporal order
- Training data always comes before test data
- Multiple train-test splits, each with expanding or sliding window

**Example with Expanding Window:**
```
Split 1: Train on [1-10]  → Test on [11]
Split 2: Train on [1-11]  → Test on [12]
Split 3: Train on [1-12]  → Test on [13]
Split 4: Train on [1-13]  → Test on [14]
...
```

**Example with Sliding Window:**
```
Split 1: Train on [1-10]  → Test on [11]
Split 2: Train on [2-11] → Test on [12]
Split 3: Train on [3-12] → Test on [13]
...
```

**Pros:**
- ✅ Respects temporal order
- ✅ No future information leakage
- ✅ Realistic evaluation scenario

**Cons:**
- ❌ Less data for early splits
- ❌ Computationally more expensive
- ❌ More complex to implement correctly

### 3. Walk-Forward Validation (Our Method)

A specific type of time-series cross-validation that:
- Uses a **single time index** (target_month)
- Ensures **strict temporal separation**
- Mimics **real-world prediction scenario**

---

## Standard K-Fold Cross-Validation

### How It Was Used in This Project

**Implementation:**
```python
from sklearn.model_selection import cross_val_score

# Standard K-Fold (K=5)
cv_scores = cross_val_score(
    model, 
    X, 
    y, 
    cv=5,           # 5 folds
    scoring='r2'    # R² score
)

# Results
cv_r2_mean = cv_scores.mean()    # Average R² across folds
cv_r2_std = cv_scores.std()      # Standard deviation
```

### Process Flow

```
1. Load all data (1784 rows after cleaning)
2. Shuffle data randomly (if not time-series)
3. Split into 5 equal folds (~357 rows each)
4. For each fold i:
   a. Use fold i as test set
   b. Use folds 1,2,3,4,5 (except i) as training set
   c. Train model on training set
   d. Evaluate on test set
   e. Record metric (R², RMSE, MAE)
5. Calculate mean and std across all 5 iterations
```

### Visual Representation

```
All Data (1784 rows)
│
├─ Fold 1 (357 rows) ──┐
├─ Fold 2 (357 rows) ──┤
├─ Fold 3 (357 rows) ──┼─→ Train on 4 folds (1427 rows)
├─ Fold 4 (357 rows) ──┤   Test on 1 fold (357 rows)
└─ Fold 5 (357 rows) ──┘

Iteration 1: Train [2,3,4,5] → Test [1]
Iteration 2: Train [1,3,4,5] → Test [2]
Iteration 3: Train [1,2,4,5] → Test [3]
Iteration 4: Train [1,2,3,5] → Test [4]
Iteration 5: Train [1,2,3,4] → Test [5]
```

### Results from This Project

**Example (Gradient Boosting):**
- CV R² Mean: 0.3185
- CV R² Std: ±0.1602
- Interpretation: Model explains 31.85% of variance on average, with ±16% variation across folds

**Limitations for Time-Series:**
- ❌ Random shuffling breaks temporal order
- ❌ Future data can appear in training set before past data
- ❌ Not realistic for time-series prediction

---

## Time-Series Cross-Validation

### Why Standard K-Fold Fails for Time-Series

**Problem Example:**
```
Standard K-Fold might do:
Train on: [2023-01, 2023-03, 2023-05, 2024-02, 2024-04]
Test on:   [2023-02, 2023-04, 2024-01, 2024-03]

This is WRONG because:
- Training includes 2024-02, 2024-04 (future)
- Testing includes 2023-02, 2023-04 (past)
- Model "sees the future" before predicting the past!
```

### Time-Series Approach

**Correct Approach:**
```
Train on: [2022-01, 2022-02, ..., 2023-01]  (past)
Test on:   [2023-02]                          (future)

Train on: [2022-01, 2022-02, ..., 2023-02]  (past)
Test on:   [2023-03]                          (future)
```

**Key Principle:** Training data must always be **strictly before** test data in time.

---

## Walk-Forward Validation (Our Implementation)

### Overview

Walk-forward validation is a rigorous time-series cross-validation method that:
1. **Aligns data properly**: Features at time t → Target at time t+1
2. **Uses temporal index**: `target_month` as the single source of truth
3. **Prevents leakage**: Training data strictly before test period
4. **Mimics reality**: Predicts next month using current month's features

### Detailed Process

#### Step 1: Data Alignment

**Original Data Structure:**
```
Row: county, month_dt, SNAP_rate, CalFresh_trend, FoodBank_trend, ...
```

**Problem:** Features and target are from the same month (temporal misalignment)

**Solution:** Create target alignment
```python
# Shift target to next month
df['SNAP_target'] = df.groupby('county')['SNAP_rate'].shift(-1)

# Create target month identifier
df['target_month'] = df['month_dt'] + pd.DateOffset(months=1)
```

**Result:**
```
Row structure:
- Features: All from month_dt (e.g., 2023-01)
- Target: SNAP_rate from target_month (e.g., 2023-02)
- target_month: 2023-02 (the month being predicted)
```

**Visual:**
```
Month t (2023-01): Features → Predict → Month t+1 (2023-02): Target
```

#### Step 2: Feature Engineering (Historical Only)

All features must use only historical data:

**Lagged Features:**
```python
# Lag 1: Value from 1 month ago
df['SNAP_rate_lag1'] = df.groupby('county')['SNAP_rate'].shift(1)

# Lag 2: Value from 2 months ago
df['SNAP_rate_lag2'] = df.groupby('county')['SNAP_rate'].shift(2)
```

**Rolling Means:**
```python
# 3-month rolling mean (uses only past 3 months)
df['SNAP_rate_rolling3'] = df.groupby('county')['SNAP_rate'].rolling(
    window=3, min_periods=1
).mean()
```

**Key Point:** All features are computed using `shift()` or `rolling()` which only look backward in time.

#### Step 3: Temporal Splitting Logic

**For each target month T:**

```python
# Training set: All rows where target_month < T
train_mask = df['target_month'] < T
X_train = df[train_mask][feature_cols]
y_train = df[train_mask]['SNAP_target']

# Test set: All rows where target_month == T
test_mask = df['target_month'] == T
X_test = df[test_mask][feature_cols]
y_test = df[test_mask]['SNAP_target']
```

**Visual Timeline:**
```
Time →
│─────────────────────────────────────────────────────────────│
│  Training Data (target_month < T)  │  Test (target_month = T) │
│─────────────────────────────────────────────────────────────│
│  All past months                  │  Month T only            │
│  (growing over time)               │  (single month)         │
└─────────────────────────────────────────────────────────────┘
```

#### Step 4: Walk-Forward Loop

```python
# Get unique target months (sorted chronologically)
unique_target_months = sorted(df['target_month'].unique())
# Example: [2022-09, 2022-10, ..., 2025-02]

# Start after sufficient history (month 6)
start_month_idx = 5  # Skip first 5 months for feature stability

# Walk forward through time
for i in range(start_month_idx, len(unique_target_months)):
    T = unique_target_months[i]  # Current target month
    
    # Split: train on past, test on current
    train_mask = df['target_month'] < T
    test_mask = df['target_month'] == T
    
    # Prepare data
    X_train = df[train_mask][feature_cols]
    y_train = df[train_mask]['SNAP_target']
    X_test = df[test_mask][feature_cols]
    y_test = df[test_mask]['SNAP_target']
    
    # Clean data (remove NaN/inf)
    # ... cleaning code ...
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred)
    
    # Store results
    results.append({
        'target_month': T,
        'r2': metrics['r2'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'smape': metrics['smape']
    })
```

#### Step 5: Example Walk-Through

**Month-by-Month Process:**

```
Month 1 (2023-02):
  Training: target_month < 2023-02 → 266 rows
  Testing:  target_month == 2023-02 → 50 rows
  Result: R² = -0.0388

Month 2 (2023-03):
  Training: target_month < 2023-03 → 316 rows (grew by 50)
  Testing:  target_month == 2023-03 → 46 rows
  Result: R² = -0.7217

Month 3 (2023-04):
  Training: target_month < 2023-04 → 362 rows (grew by 46)
  Testing:  target_month == 2023-04 → 48 rows
  Result: R² = -0.0806

...

Month 25 (2025-02):
  Training: target_month < 2025-02 → 1513 rows (all previous data)
  Testing:  target_month == 2025-02 → 54 rows
  Result: R² = -0.4770
```

**Key Observations:**
- Training set grows over time (expanding window)
- Each month is evaluated independently
- No future information ever leaks into training

#### Step 6: Aggregation

**Per-Month Metrics:**
- Calculate mean and std across all 25 months
- Example: R² Mean = 0.5922, R² Std = ±0.2370

**Overall Metrics:**
- Combine all predictions and actuals from all months
- Calculate overall R², RMSE, MAE, sMAPE
- Example: Overall R² = 0.6214

---

## Comparison of Methods Used in This Project

### 1. Standard K-Fold Cross-Validation

**Used for:** Initial model comparison (non-temporal)

**Implementation:**
```python
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
```

**Results Example:**
- Gradient Boosting: CV R² = 0.3185 ± 0.1602
- Random Forest: CV R² = 0.2838 ± 0.1978
- Tuned XGBoost: CV R² = 0.2588 ± 0.1592

**Characteristics:**
- ✅ Fast and simple
- ✅ Good for comparing models
- ❌ Breaks temporal order
- ❌ Not suitable for time-series evaluation

### 2. Walk-Forward Validation (Corrected)

**Used for:** Final time-series evaluation

**Implementation:**
- Custom loop with `target_month` splitting
- Features at t → Target at t+1 alignment

**Results Example:**
- Tuned XGBoost: Overall R² = 0.6214
- Random Forest: Overall R² = 0.4923
- Gradient Boosting: Overall R² = 0.2981
- Base XGBoost: Overall R² = -0.0275

**Characteristics:**
- ✅ Respects temporal order
- ✅ No data leakage
- ✅ Realistic evaluation
- ❌ More computationally expensive
- ❌ More complex implementation

### Key Differences

| Aspect | Standard K-Fold | Walk-Forward |
|--------|----------------|--------------|
| **Data Order** | Random/Shuffled | Temporal (chronological) |
| **Train/Test Split** | Random | Temporal (past/future) |
| **Leakage Risk** | High (for time-series) | None (properly implemented) |
| **Realism** | Low (for time-series) | High (mimics real scenario) |
| **Speed** | Fast | Slower (sequential) |
| **Use Case** | IID data, model comparison | Time-series, final evaluation |

---

## Detailed Step-by-Step Process

### Walk-Forward Validation: Complete Example

Let's trace through a complete example with actual data:

#### Initial Setup

```python
# Load data
df = pd.read_csv("src/data/aggregateTrends_scaled.csv")
# 2030 rows, 58 counties, 35 months

# After feature engineering and cleaning
# 1567 rows remain (after dropping rows with missing lags)
```

#### Month 1: Predicting 2023-02

```python
T = 2023-02  # Target month

# Training set
train_mask = df['target_month'] < pd.Timestamp('2023-02')
# Result: 266 rows
# These are rows where target_month is 2022-09, 2022-10, ..., 2023-01

# Test set
test_mask = df['target_month'] == pd.Timestamp('2023-02')
# Result: 50 rows
# These are rows where target_month is exactly 2023-02

# Features in training set:
# - All from months 2022-08 to 2023-01 (feature months)
# - Predicting targets from months 2022-09 to 2023-01 (target months)

# Features in test set:
# - All from month 2023-01 (feature month)
# - Predicting targets from month 2023-02 (target month)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)  # -0.0388
rmse = sqrt(mean_squared_error(y_test, y_pred))  # 0.002018
```

#### Month 2: Predicting 2023-03

```python
T = 2023-03  # Target month

# Training set (now includes previous test month!)
train_mask = df['target_month'] < pd.Timestamp('2023-03')
# Result: 316 rows (grew from 266 to 316)
# Now includes: 2022-09, 2022-10, ..., 2023-01, 2023-02

# Test set
test_mask = df['target_month'] == pd.Timestamp('2023-03')
# Result: 46 rows

# Key point: Training data now includes 2023-02 data
# This is correct because in real-world, after predicting 2023-02,
# we would have the actual 2023-02 data available for training
```

#### Final Month: Predicting 2025-02

```python
T = 2025-02  # Target month

# Training set (maximum size)
train_mask = df['target_month'] < pd.Timestamp('2025-02')
# Result: 1513 rows
# Includes all months from 2022-09 to 2025-01

# Test set
test_mask = df['target_month'] == pd.Timestamp('2025-02')
# Result: 54 rows
```

### Aggregation Process

```python
# After all 25 months evaluated:

# Per-month statistics
r2_scores = [-0.0388, -0.7217, -0.0806, ..., -0.4770]  # 25 values
r2_mean = np.mean(r2_scores)  # 0.5922
r2_std = np.std(r2_scores)    # 0.2370

# Overall statistics (combine all predictions)
all_predictions = [pred1, pred2, ..., pred1242]  # All predictions
all_actuals = [actual1, actual2, ..., actual1242]  # All actuals

overall_r2 = r2_score(all_actuals, all_predictions)  # 0.6214
overall_rmse = sqrt(mean_squared_error(all_actuals, all_predictions))  # 0.001155
```

---

## Key Takeaways

### 1. Cross-Validation Purpose
- Estimate model performance on unseen data
- Reduce variance in performance estimates
- Detect overfitting/underfitting

### 2. Method Selection
- **Standard K-Fold**: For IID data, model comparison
- **Time-Series/Walk-Forward**: For temporal data, realistic evaluation

### 3. Walk-Forward Advantages
- ✅ No data leakage
- ✅ Realistic evaluation scenario
- ✅ Respects temporal order
- ✅ Growing training set mimics real-world

### 4. Implementation Requirements
- Proper temporal alignment (features at t → target at t+1)
- Historical-only feature engineering
- Strict temporal splitting (past < future)
- Consistent time index (target_month)

### 5. Results Interpretation
- **Per-month metrics**: Show consistency across time
- **Overall metrics**: Aggregate performance across all predictions
- **Standard deviation**: Indicates stability/variance

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Random Shuffling for Time-Series
```python
# WRONG
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# This breaks temporal order!
```

### ❌ Pitfall 2: Future Data in Training
```python
# WRONG
train = df[df['month'] <= '2023-06']
test = df[df['month'] == '2023-07']
# But features might include future information!
```

### ❌ Pitfall 3: Target Leakage
```python
# WRONG
df['target_rolling'] = df['target'].rolling(3).mean()
# Using target to predict target!
```

### ✅ Correct Approach
```python
# CORRECT
df['target'] = df.groupby('county')['SNAP_rate'].shift(-1)
df['target_month'] = df['month_dt'] + pd.DateOffset(months=1)
train_mask = df['target_month'] < T
test_mask = df['target_month'] == T
```

---

*This document explains the cross-validation methods used in the SNAP application prediction project. For implementation details, see `experiments/walk_forward_backtest.py`.*

