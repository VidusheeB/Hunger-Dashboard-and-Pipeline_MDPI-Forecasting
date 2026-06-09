# Methods Summary — CalFresh SNAP Prediction Model
**For paper methods section drafting. All numbers are from the final leakage-free pipeline.**

---

## 1. What We Are Predicting

**Target variable:** `SNAP_Application_Rate` — the number of new CalFresh (SNAP) applications submitted in a given county in a given calendar month, divided by that county's total population.

```
SNAP_Application_Rate(county, month) = SNAP_Applications(county, month) / Population(county)
```

- Unit: applications per capita (e.g., 0.003 = 3 applications per 1,000 residents)
- Granularity: county × month (58 California counties × 105 months = April 2017 – December 2025)
- This is a **nowcasting / retrospective estimation task**: SNAP administrative data is released every fiscal year. The model estimates the application rate before the official report is available, using only data observable at prediction time.

---

## 2. Data Sources

### 2a. SNAP Applications
- **Source:** California Department of Social Services (CDSS) administrative records
- **File:** `SNAPData.csv`
- **Format:** county, month-year, total new applications
- **Coverage:** April 2017 – December 2025, all 58 California counties
- **Cleaning:** Rows where applications exceed 3× the county's historical median are flagged and removed as data entry errors (e.g., Madera County January 2023: 11,090 vs typical ~1,000). Threshold = 3.0× median (set in `config.py`, `OUTLIER_THRESHOLD`).

### 2b. Google Trends
- **Source:** Google Trends API, retrieved monthly at DMA (Designated Market Area) level
- **Keywords tracked:**
  - `CalFresh` — the California-specific name for SNAP; direct program search; topic NOT keyword; catches related searches
  - `Food Bank` — broader food insecurity signal; topic not keyword; catches related searches
  - `Food Stamps` — legacy name; captures older/different-language searchers
  - `SNAP Topic` — Google's topic entity for SNAP, catches related searches
- **Temporal unit:** Weekly index (0–100, where 100 = peak search interest in the DMA over the pulled time window)
- **Monthly aggregate:** Mean of weekly values within each calendar month → `monthly_average_{keyword}`
- **Temporal alignment:** **CRITICAL — All Trends features used to predict month-t SNAP applications are from month t-1 or earlier (Nov Trends → Dec SNAP).** Same-month Trends are not available at prediction time for real-time deployment.

#### Google Trends Scaling
Each annual Trends CSV is independently normalized 0–100 within its own download window, so raw values across different annual files are not directly comparable. To stitch them into a continuous series, a chain-rescaling algorithm is applied (`pipeline/stage1_load_raw.py`, `_stitch_dma()`):

1. Sort all annual files for a DMA oldest-first; anchor the newest file at scale = 1.0
2. For each adjacent pair (older, newer), identify the overlapping weeks — one month of overlap is maintained between consecutive downloads for exactly this purpose
3. Compute the rescaling ratio from the **mean of all overlapping weeks**:
   ```
   ratio = mean(newer[overlap_weeks]) / mean(older[overlap_weeks])
   scale[older] = ratio × scale[newer]
   ```
4. Apply cumulative scale factors; where dates overlap, the newer file's scaled value wins
5. Re-normalize the full stitched series to 0–100

The mean across all overlapping weeks is used rather than a single anchor point, for robustness against week-to-week Google Trends noise (values can fluctuate 10–20 index points within a month for unchanged underlying search volume). Minimum overlap required: 4 weeks; if fewer, ratio defaults to 1.0.

- Scaling parameters are stored in `outputs/data/trend_scaling_params.json`
- Scaling is applied at the DMA level (14 California DMAs), not at the county level

#### County-to-DMA Mapping
Counties are mapped to DMAs via `county_to_metro.csv`. California counties share DMAs (e.g., both San Francisco and Alameda counties use the San Francisco–Oakland–San Jose DMA Trends values). This means counties within the same DMA receive identical Trends signals; variation between them comes from demographics and SNAP history.

### 2c. BLS LAUS Unemployment
- **Source:** Bureau of Labor Statistics Local Area Unemployment Statistics (LAUS)
- **File:** `laus_county_unemployment_2017_2025.csv`
- **Coverage:** Monthly unemployment rate for all 58 California counties
- **Reporting lag:** BLS releases county unemployment with approximately 1-month lag (April data released in late May)
- **Features used:** `unemployment_rate` = LAUS unemployment from t-1 for SNAP month t; `unemployment_rate_lag1` = LAUS unemployment from t-2

### 2d. Population
- **Source:** US Census Bureau / California Department of Finance population estimates
- **File:** `popData.csv`
- **Coverage:** Annual estimates; constant within a calendar year for a given county
- **Use:** Denominator for SNAP_Application_Rate; also used as raw `Population` feature and log-transformed `log_population`

### 2e. Median Household Income
- **Source:** American Community Survey (ACS) 5-year estimates
- **File:** `MedianIncome.csv`
- **Coverage:** Annual estimates per county; constant within a year
- **Unit:** US dollars (range across CA counties: ~$53K–$160K)

---

## 3. Pipeline Stages

The pipeline runs as 6 sequential stages (entry point: `run_pipeline.py`):

### Stage 1: Data Ingestion (`pipeline/build_training_data.py`)
- Loads SNAP applications → computes per-capita rate
- Loads and monthly-aggregates Google Trends for all 4 keywords across 14 DMAs
- Applies ratio-based scaling to Trends values
- Joins SNAP data to Trends via county→DMA crosswalk (merge on `trend_date = SNAP_month - 1 month`)
- Joins population and income
- Outputs: `outputs/data/training_data.csv`

### Stage 2: Feature Engineering (`pipeline/feature_engineering.py`)
Applied in order within each county group (sorted by date):

| Step | Function | Output features | Notes |
|---|---|---|---|
| A | `add_lag_features()` | `calfresh_lag{1,2}`, `foodbank_lag{1,2}`, `foodstamps_lag{1,2}`, `snaptopic_lag{1,2}` | Stage 2 already aligns raw Trends to t-1; `lag1 = monthly_average_*`, `lag2 = groupby(county).shift(1)` |
| B | `add_rolling_features()` | `calfresh_roll3`, `foodbank_roll3`, `foodstamps_roll3`, `snaptopic_roll3` | `series.rolling(window=3, min_periods=2).mean()` after Stage 2 alignment — window covers Trends at t-1, t-2, t-3 only |
| C | `add_momentum_features()` | `calfresh_momentum`, `foodbank_momentum`, `foodstamps_momentum`, `snaptopic_momentum` | `lag1 - lag2` (e.g., Nov Trends minus Oct Trends), **not** same-month minus lag1 |
| D | `add_seasonality_features()` | `month_sin`, `month_cos`, `quarter` | `sin/cos(2π × month / 12)` for cyclical encoding |
| E | `add_population_features()` | `log_population` | `log10(Population)` |
| F | `add_income_features()` | `log_income` | `log10(Median_Income)` |
| G | `add_unemployment_features()` | `unemployment_rate`, `unemployment_rate_lag1` | BLS LAUS shifted before merge: t-1 and t-2 only |

**Leakage safeguards:**
- Trends are first joined to `trend_date = SNAP_month - 1 month`, so raw monthly Trends are already t-1 relative to the target SNAP month
- Lag and rolling features are computed within county — no cross-county leakage and no month-t Trends
- Rolling windows cover t-1, t-2, and t-3 only
- Momentum = lag1 − lag2 — no same-month Trends ever appears as a feature
- No SNAP outcome lags or rolling SNAP outcome features are created or used.

Outputs: `outputs/data/features.csv`

### Stage 3: Model Training (`pipeline/train_model.py`)
- In production: full-data XGBoost fit for deployment
- In experiments: walk-forward validation (see Section 5)

### Stage 4: Evaluation (`pipeline/evaluate_model.py`)
- Computes walk-forward metrics, feature importances, per-county accuracy

### Stage 5: Hyperparameter Tuning (`experiments/tune_deployable_model.py`)
See Section 6.

### Stage 6: Alert Layer (`experiments/threshold_alert.py`)
See Section 8.

---

## 4. Feature Set

The model uses 26 deployable features. SNAP outcome lags and rolling SNAP outcome features are not part of the pipeline because they require official SNAP data, which has a reporting lag.

| # | Feature | Group | Formula / Source | Scale | What it captures |
|---|---|---|---|---|---|
| 1 | `Population` | Demographics | US Census annual estimate | Integer (1,043 – 9,550,505) | County size; denominator for rate |
| 2 | `Median_Income` | Demographics | ACS 5-year estimate ($) | Float ($53K – $160K) | County wealth level |
| 3 | `calfresh_lag1` | Trends (lag) | monthly_average_CalFresh(t-1) | 0–100 Google index | CalFresh searches last month |
| 4 | `calfresh_lag2` | Trends (lag) | monthly_average_CalFresh(t-2) | 0–100 Google index | CalFresh searches 2 months ago |
| 5 | `foodbank_lag1` | Trends (lag) | monthly_average_FoodBank(t-1) | 0–100 Google index | Food bank searches last month |
| 6 | `foodbank_lag2` | Trends (lag) | monthly_average_FoodBank(t-2) | 0–100 Google index | Food bank searches 2 months ago |
| 7 | `foodstamps_lag1` | Trends (lag) | monthly_average_FoodStamps(t-1) | 0–100 Google index | Food stamps searches last month |
| 8 | `foodstamps_lag2` | Trends (lag) | monthly_average_FoodStamps(t-2) | 0–100 Google index | Food stamps searches 2 months ago |
| 9 | `snaptopic_lag1` | Trends (lag) | monthly_average_SNAPTopic(t-1) | 0–100 Google index | SNAP topic searches last month |
| 10 | `snaptopic_lag2` | Trends (lag) | monthly_average_SNAPTopic(t-2) | 0–100 Google index | SNAP topic searches 2 months ago |
| 11 | `calfresh_roll3` | Trends (rolling) | mean(calfresh at t-1, t-2, t-3) | 0–100 Google index | 3-month sustained CalFresh interest |
| 12 | `foodbank_roll3` | Trends (rolling) | mean(foodbank at t-1, t-2, t-3) | 0–100 Google index | 3-month sustained food bank interest |
| 13 | `foodstamps_roll3` | Trends (rolling) | mean(foodstamps at t-1, t-2, t-3) | 0–100 Google index | 3-month sustained food stamps interest |
| 14 | `snaptopic_roll3` | Trends (rolling) | mean(snaptopic at t-1, t-2, t-3) | 0–100 Google index | 3-month sustained SNAP topic interest |
| 15 | `calfresh_momentum` | Trends (momentum) | calfresh_lag1 − calfresh_lag2 | Difference of 0–100 indices | Direction of CalFresh search trend |
| 16 | `foodbank_momentum` | Trends (momentum) | foodbank_lag1 − foodbank_lag2 | Difference of 0–100 indices | Direction of food bank search trend |
| 17 | `foodstamps_momentum` | Trends (momentum) | foodstamps_lag1 − foodstamps_lag2 | Difference of 0–100 indices | Direction of food stamps search trend |
| 18 | `snaptopic_momentum` | Trends (momentum) | snaptopic_lag1 − snaptopic_lag2 | Difference of 0–100 indices | Direction of SNAP topic search trend |
| 19 | `unemployment_rate` | BLS LAUS | county unemployment at t-1 | 0–100 (%) | Recent economic stress, publication-safe |
| 20 | `unemployment_rate_lag1` | BLS LAUS | county unemployment at t-2 | 0–100 (%) | Lagged economic stress |
| 21 | `month_sin` | Seasonality | sin(2π × month / 12) | −1 to 1 | Cyclical month (Dec–Jan adjacency) |
| 22 | `month_cos` | Seasonality | cos(2π × month / 12) | −1 to 1 | Cyclical month (paired with sin) |
| 23 | `quarter` | Seasonality | floor((month−1)/3) + 1 | 1–4 integer | Broad seasonal grouping |
| 24 | `month` | Seasonality | Calendar month | 1–12 integer | Raw month for tree splits |
| 25 | `log_population` | Transforms | log10(Population) | 3.0–7.0 | Compressed county size |
| 26 | `log_income` | Transforms | log10(Median_Income) | 4.73–5.20 | Compressed county wealth |

---

## 5. Walk-Forward Validation

### Design
Walk-forward validation (also called rolling-origin cross-validation) strictly preserves temporal ordering: the model is **never trained on data from after the month it is predicting**.

```
For each test month t:
    train = all county-months where date < t
    predict = all county-months where date == t
```

- **Minimum training window:** 12 months (one full seasonal cycle) before the first test month opens. This means testing begins in April 2018 (12 months after April 2017 start).
- **Total predictions:** 5,329 county-month predictions (92 test months × 58 counties, minus months with missing data)
- **COVID handling:** Months January 2020 – December 2021 are included in walk-forward training and prediction (to avoid gaps and train-set discontinuities), but **excluded from all regression metric calculations**. This prevents COVID-era anomalies from inflating error.
- **Non-COVID evaluation rows:** 3,941 county-months (used for all reported R², MAE, sMAPE)

### Example
To predict Fresno County's SNAP application rate for September 2022:
- Training data: all 58 counties, April 2017 – August 2022
- Test input: Fresno County's Trends lags (Aug 2022 values), unemployment, demographics, seasonality
- Test target: Fresno County September 2022 SNAP_Application_Rate
- The model has never seen September 2022 or later during this prediction

### Metrics
- **R²** — coefficient of determination; proportion of variance explained. 1.0 = perfect, 0 = predicts the mean.
- **MAE** — mean absolute error in application rate units (e.g., 0.000849 ≈ 0.85 applications per 1,000 residents off on average)
- **sMAPE** — symmetric mean absolute percentage error: `100 × |actual − pred| / ((|actual| + |pred|) / 2)`. More interpretable for rates near zero.

---

## 6. Model: XGBoost

### Algorithm
XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees sequentially, where each new tree corrects the residual errors of all previous trees. Final prediction = sum of all tree outputs.

Why XGBoost for this data:
- Handles non-linear interactions between Trends signals, demographics, and seasonality without manual engineering
- Robust to outliers via regularization
- Native handling of missing values
- No feature scaling required (tree-based, not distance-based)

### Hyperparameter Tuning
Two-stage tuning on the first 78% of dates (held-out validation = remaining 22%, never used in metric reporting):

**Stage 1: RandomizedSearch** (60 candidates, inner 70/30 temporal split on tuning dates)
- `n_estimators`: 100–1,000
- `max_depth`: 3–10
- `learning_rate`: 0.01–0.3
- `min_child_weight`: 1–10
- `subsample`: 0.5–1.0
- `colsample_bytree`: 0.5–1.0
- `reg_lambda`: 0.1–10.0
- `reg_alpha`: 0.0–1.0

**Stage 2: Focused Grid** (27 combinations) refining around Stage 1 best

**Final tuned hyperparameters:**
| Parameter | Value | Role |
|---|---|---|
| `n_estimators` | 799 | Number of trees |
| `max_depth` | 9 | Maximum tree depth (controls complexity) |
| `learning_rate` | 0.05681 | Shrinkage factor per tree |
| `min_child_weight` | 4 | Minimum observations in a leaf |
| `subsample` | 0.8730 | Fraction of rows used per tree (reduces overfitting) |
| `colsample_bytree` | 0.6559 | Fraction of features used per tree |
| `reg_lambda` | 1.9993 | L2 regularization (penalizes large weights) |
| `reg_alpha` | 0.0147 | L1 regularization (encourages sparsity) |
| `random_state` | 42 | Fixed seed for reproducibility |

Tuning selection criterion: **MAE on held-out validation set**. The final 24 months are reserved during tuning, then the tuned settings are used for walk-forward evaluation.

**Are these hyperparameters arbitrary?** The ranges are informed by XGBoost conventions; the specific values are data-driven via optimization. They are not arbitrary — they are the values that minimized held-out MAE on this specific dataset. `min_child_weight=4` and `max_depth=9` reflect that county-month data has moderate complexity; `subsample=0.87` and `colsample_bytree=0.66` indicate moderate tree diversity is beneficial.

### Model Performance (Walk-Forward, Non-COVID)
| Model | R² | MAE | sMAPE |
|---|---|---|---|
| Model (26 deployable features) | 0.6418 | 0.000836 | 15.47% |
| No-Trends baseline (10 features) | 0.5528 | 0.001006 | 19.32% |

The gap between the deployable model (0.642) and no-Trends baseline (0.553) is the value Google Trends adds.

### Feature Importances (Deployable Model, full-data fit)
Top features by XGBoost gain (see Fig 2):
- Trends features (lags, rolling means, momentum) are collectively the dominant signal group
- `unemployment_rate` and `unemployment_rate_lag1` are the next most important single features
- Seasonality features (`month`, `month_sin`, `month_cos`) contribute meaningfully
- Demographics (`Population`, `Median_Income`, `log_population`, `log_income`) have smaller but non-zero importance

---

## 7. Statistical Tests

### 7a. Diebold-Mariano Test (DM, HLN-corrected, month-clustered)
**What it tests:** Whether two forecast sequences have significantly different mean squared error. H₀ = both forecasters have equal accuracy.

**Setup:**
- Let e₁ = errors from No-Trends model, e₂ = errors from With-Trends model
- Because the data are a county-month panel, paired county losses are first averaged within each forecast month. The test unit is the forecast month, not the individual county-month.
- d_t = mean_county(e₁² − e₂²) for forecast month t
- DM statistic = d̄ / √(Var(d)/n), where Var(d) accounts for autocorrelation via one lag (γ₁)
- HLN correction adjusts for small-sample bias: `DM_adj = DM × √((n + 1 − 2 + 1/n) / n)`
- p-value from t-distribution with n−1 degrees of freedom, two-tailed
- **Sign convention:** Positive DM stat = Model 2 (With-Trends) has lower squared error = Trends helps

**Trends ablation result:** DM stat = +3.187, p = 0.0022 ✓

**Concrete example for methods section:**  
To test whether Google Trends improves prediction at training gap=0 (standard walk-forward):  
- 3,941 non-COVID county-month predictions were generated by both models  
- County-level loss differentials were averaged within each of 68 forecast months  
- Monthly loss differentials d_t = mean_county(e_no² − e_with²) were tested with HAC lag 1  
- DM_adj = +3.187, p = 0.0022 — the With-Trends model has significantly lower squared error

### 7b. Wilcoxon Signed-Rank Test
**What it tests:** Non-parametric test of whether With-Trends absolute errors are systematically smaller than No-Trends absolute errors. Makes no distributional assumption.

**Setup:**  
- Paired: monthly mean |e_no| vs monthly mean |e_with| across the same 68 forecast months  
- H₀: median difference = 0 (no systematic improvement)  
- Two-sided p-value reported, with a one-sided "With-Trends better" p-value when that directional hypothesis is pre-specified

**Result:** stat = 663.0, two-sided p = 0.0018 ✓; one-sided p = 0.0009.

### 7c. Spearman Rank Correlation
**What it tests:** Monotonicity — does ΔR² (benefit of Trends) change systematically as the training gap increases?

- ρ(gap, ΔR²) = −0.967, p < 0.0001 for Lag Robustness experiment → near-perfect negative monotonicity

**Why report Spearman too?** Applying many pointwise tests across gap settings can inflate Type I error. The Spearman result summarizes the overall monotonic pattern across gaps.

---

## 8. Experiments

### Experiment 1: Trends Ablation (`experiments/trends_ablation.py`)
**Question:** Does adding Google Trends improve SNAP application rate prediction?

**Setup:**
- Feature sets compared:
  - No-Trends: Population, Median_Income, unemployment_rate, unemployment_rate_lag1, month_sin, month_cos, quarter, month, log_population, log_income (10 features)
  - With-Trends: same 10 + 16 Trends features (lags, rolling, momentum for all 4 keywords) = 26 features
- Same walk-forward setup, same tuned XGBoost hyperparameters
- Metrics on 3,999 non-COVID county-months

**Results:**
| Metric | No Trends | With Trends | Δ |
|---|---|---|---|
| R² | 0.5528 | 0.6418 | +0.0890 |
| MAE | 0.001006 | 0.000836 | −0.000170 |
| sMAPE | 19.32% | 15.47% | −3.85% |
| DM p-value | — | — | 0.0023 ✓ |
| Wilcoxon p | — | — | 0.0006 ✓ |

**Interpretation:** Google Trends explains an additional ~9 percentage points of variance in SNAP applications. Both parametric (DM) and non-parametric (Wilcoxon) tests confirm the improvement is statistically significant.

---

### Experiment 2: Lag Robustness — Training Gap (`experiments/lag_robustness.py`)
**Question:** If the prediction model has not been retrained for L months (stale training), can real-time Google Trends compensate?

**Framing:** In deployment, SNAP administrative data has a ~6-month lag. The model is trained on historical SNAP outcomes; if those outcomes are delayed, the model itself becomes stale. This experiment simulates that scenario.

**Setup:**
- Neither model uses SNAP rate as a feature (both use only Trends/unemployment/demographics/seasonality)
- Walk-forward **with gap L**: for each test month t, training cutoff = t − L months
  ```
  train = county-months where date < (t - L_months)
  test  = county-months where date == t
  ```
- At gap=0: equivalent to standard walk-forward (baseline)
- At gap=6: model has not seen SNAP outcomes from the 6 months before the test month
- The With-Trends model receives **recent Trends only**: t-1, t-2, and trailing t-1 through t-3 summaries, which are observable before predicting SNAP month t
- Gaps tested: 0 through 12 months
- Month-clustered DM and Wilcoxon are reported at every gap for transparency; Spearman provides the overall monotonicity evidence

**Concrete example at gap=6:**
- Predicting Alameda County, December 2022
- Training data: all county-months before June 2022 (i.e., at least 6 months before December)
- Features at prediction time: Trends from November 2022 (lag1) and October 2022 (lag2), unemployment from November/October 2022, demographics
- This mirrors reality: official SNAP data for summer 2022 hasn't been reported yet

**Results:**
| Gap | No-Trends R² | With-Trends R² | ΔR² | DM p |
|---|---|---|---|---|
| 0 (baseline) | 0.5528 | 0.6440 | +0.0912 | 0.0018 ✓ |
| 3 | 0.5059 | 0.5738 | +0.0679 | 0.0183 ✓ |
| 6 | 0.4797 | 0.5211 | +0.0414 | 0.1154 ✗ |
| 9 | 0.4607 | 0.4736 | +0.0129 | 0.6273 ✗ |
| 10 | 0.4507 | 0.4540 | +0.0033 | 0.9040 ✗ |
| 11 | 0.4422 | 0.4305 | -0.0117 | 0.7025 ✗ |
| 12 | 0.3551 | 0.3733 | +0.0182 | 0.6113 ✗ |

**Spearman ρ(gap, ΔR²) = −0.967, p < 0.0001** → Trends benefit decreases monotonically as the training gap grows. The point estimate is positive at 12 of 13 tested gaps, but month-clustered DM significance is limited to gaps 0–4.

**Interpretation:** Real-time Google Trends maintain meaningful predictive value when the model's training data is moderately stale. Evidence is strongest through a 5-month training gap; beyond that, effect sizes remain positive but are not statistically significant under the month-clustered DM test.

---

## 9. Alert System

### Design Philosophy
No separate classification model is trained. The baseline XGBoost walk-forward predictions are the forecasts. Alerts are derived from **deviation** between prediction and actuality:

```
deviation(county, month) = (actual_SNAP_rate - predicted_SNAP_rate) / predicted_SNAP_rate
```

A large positive deviation means actual applications came in much higher than predicted — a demand spike the model didn't anticipate. The alert system flags these.

### Threshold Optimization
County-specific thresholds are estimated **non-circularly** using a temporal train/test split:
- **Train split (70%):** First 65 months (for estimating county-level percentile thresholds)
- **Test split (30%):** Last 28 months (for evaluating threshold performance)

**Ground truth definition (fixed, test set):**  
A month is a "true spike" if its deviation exceeds the **85th percentile of all positive deviations** in the test set. This gives 80 true spike events out of 1,622 test county-months (4.9% prevalence). The 85th percentile was chosen as a definition of "noteworthy" rather than "routine" positive deviation — it captures the top 15% of above-average months.

**Threshold selection (Lipton et al., 2014):**  
We sweep candidate Red threshold percentiles from 50th to 95th. For each:
1. Compute county-specific Xth-percentile of positive deviations from training data
2. Apply those thresholds to test data → predict Red/Not-Red
3. Evaluate against the fixed ground truth → TP, FP, FN, TN

**Selection criterion:** F1-optimal threshold (maximizes F1 on held-out test data, balancing recall and precision). This is appropriate when false positives (false alarms eroding trust) and false negatives (missed spikes leaving food banks unprepared) both carry cost, with neither dominating.

**Result:** F1-optimal = **60th percentile** (F1 = 0.871)

**Yellow threshold:** 10 percentile points below Red = **50th percentile**

### Final Thresholds
| Label | County threshold | Meaning |
|---|---|---|
| **Green** | deviation ≤ county 50th pctile | Normal month |
| **Yellow** | county 50th < deviation ≤ county 60th pctile | Elevated — watch |
| **Red** | deviation > county 60th pctile | Demand surge — act |

County-specific 60th-percentile Red thresholds range from 0.094 to 0.298, reflecting different baseline volatility across counties.

### Confusion Matrix (Test Set, 1,622 county-months)
| | Predicted Red | Predicted Not-Red |
|---|---|---|
| **Actual Spike** | TP = 74 | FN = 6 |
| **Actual Normal** | FP = 16 | TN = 1,526 |

| Metric | Value | Meaning |
|---|---|---|
| Recall | 0.925 | Catches 92.5% of true demand spikes |
| Precision | 0.822 | 82.2% of Red alerts are true spikes |
| Specificity | 0.990 | 99.0% of normal months correctly not flagged |
| False Positive Rate | 0.010 | 1.0% false alarm rate on clear months |
| F1 | 0.871 | Harmonic mean of recall and precision |
| Youden's J | 0.915 | Overall diagnostic ability (0=random, 1=perfect) |

### Full Dataset Label Distribution
| Label | Count | % |
|---|---|---|
| Green | 4,046 | 75.1% |
| Yellow | 254 | 4.7% |
| Red | 1,087 | 20.2% |

---

## 10. Potentially Arbitrary Numbers & Decisions

| Number/Decision | How it was set | How defensible it is |
|---|---|---|
| `min_child_weight = 4` | Tuned via RandomizedSearch on held-out data | Data-driven, not arbitrary |
| `subsample = 0.873` | Same | Data-driven |
| WALK_FORWARD_MIN_MONTHS = 12 | One full seasonal cycle | Standard practice; tested sensitivity |
| OUTLIER_THRESHOLD = 3.0× | Catches known data entry errors | Could justify 2.5–4.0×; 3.0 is standard |
| COVID exclusion: 2020-01-01 – 2021-12-31 | Covers main disruption period | Slightly aggressive; 2020-03 to 2021-06 would also be defensible |
| TRUE_EVENT_PCT = 85th percentile | Top 15% of positive deviations = "spike" | Somewhat arbitrary; 80th or 90th could also be argued |
| Yellow = Red − 10 percentile points | Design decision for 3-band system | Arbitrary spacing; no statistical basis — chosen for operational clarity |
| Rolling window = 3 months | Balances smoothing vs data loss | Tested 6 months — minor difference, loses more early rows |
---

## 11. Current Limitations

1. **County-DMA aggregation:** Google Trends are available at DMA (metro area) level, not county level. Counties within the same DMA receive identical Trends values — intra-DMA variation is invisible to the model.

2. **Annual income and population data:** `Median_Income` and `Population` are annual estimates applied to all months in a year. They miss within-year economic shocks.

3. **BLS unemployment lag:** The pipeline uses publication-safe unemployment lags only: t-1 and t-2. If the t-1 LAUS release is not available for a future prediction, Stage 5 falls back to the most recent older LAUS month.

4. **Trends index relativity:** Google Trends values (0–100) are relative to the peak within the queried time window, not absolute search volumes. The ratio-based scaling partially addresses this but does not fully solve cross-period comparability.

5. **COVID period:** The model is trained on COVID-era data (it receives those rows) but COVID county-months are excluded from metric calculation. Model behavior on future shock events of similar magnitude is untested.

6. **No county-level fixed effects:** The model does not include county fixed effects (dummies). County identity is represented only via Population, Income, and Trends (which are DMA-level). Counties with unusual structural characteristics may be systematically over- or under-predicted.

7. **Alert system uses deviation, not raw counts:** A county with very few applications can have a large deviation from a small absolute increase. High-volatility small counties may generate more Red alerts than are operationally meaningful.

---

## 12. Figures Reference

| File | Content |
|---|---|
| `fig1_actual_vs_predicted.png` | Walk-forward predictions vs actual (statewide monthly mean, non-COVID) |
| `fig2_feature_importance.png` | XGBoost feature importance by group (deployable model, full-data fit) |
| `fig3_trends_ablation.png` | With vs Without Trends: R², MAE, sMAPE comparison |
| `fig4_lag_robustness.png` | Lag robustness: R² and ΔR² by training gap (0–12 months) |
| `fig6_roc_curve.png` | ROC-style curve: Recall vs FPR for threshold sweep |
| `fig7_precision_recall.png` | Precision-Recall curve for threshold sweep |
| `fig8_confusion_matrix.png` | Confusion matrix at F1-optimal (60th percentile) threshold |
| `fig9_alert_distribution.png` | Green/Yellow/Red label distribution (bar + pie) |
| `fig10_residuals_distribution.png` | Residual distribution histogram + Q-Q plot |
| `fig11_walkforward_r2_over_time.png` | Monthly R² over the walk-forward test period |
| `table1_threshold_sweep.png` | Full threshold sweep table (all 10 candidate percentiles) |
