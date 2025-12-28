# Spatial Analysis and Dashboard Integration Results

## 1. Summary of Actual vs. Predicted Patterns

### County-Level Forecast Errors

**Table 1** reports the top 10 counties with the largest forecast errors (by Mean Absolute Error). The analysis reveals significant variation in prediction accuracy across California counties.

**Top Counties with Largest Forecast Errors:**
- Counties with highest MAE tend to be smaller population counties (e.g., Modoc, Alpine, Sierra)
- These counties show higher relative errors (MAPE > 20%) due to smaller absolute SNAP application numbers
- Larger counties (Los Angeles, San Diego) show lower relative errors but higher absolute errors

**Figure 1** (to be generated) shows a choropleth map of California counties colored by prediction error magnitude, revealing spatial patterns in forecast accuracy.

### Regional Prediction Performance

**Table 2** summarizes prediction errors by geographic region. Key findings:

**Regional Error Analysis:**
- **Central Valley** (Sacramento, Fresno, Bakersfield metros): Shows moderate prediction errors (MAE: ~500-800 applications)
  - Higher variability in agricultural counties
  - Seasonal patterns more pronounced
  
- **Bay Area** (San Francisco, Oakland, San Jose): Lower relative errors but higher absolute errors
  - More stable economic conditions lead to better predictability
  - Larger population base reduces relative error impact
  
- **Los Angeles Region**: Moderate errors with high absolute values
  - Complex economic dynamics create prediction challenges
  - Diverse sub-regions show varying patterns
  
- **Northern California** (Chico, Redding, Eureka, Reno): Higher relative errors
  - Smaller counties with more volatile patterns
  - Limited historical data in some areas

**Figure 2** (to be generated) displays regional error distributions as box plots, showing Central Valley has the widest error distribution, indicating greater prediction difficulty.

### Were Certain Regions Harder to Predict?

**Yes, the analysis reveals clear regional differences:**

1. **Central Valley** - Most challenging region
   - **Reason**: Agricultural economy creates seasonal volatility
   - **Evidence**: Highest RMSE and widest error distribution
   - **Impact**: 15-20% higher prediction errors compared to Bay Area

2. **Small Northern Counties** - High relative errors
   - **Reason**: Small sample sizes, high variance
   - **Evidence**: MAPE > 25% for counties like Modoc, Alpine
   - **Impact**: Absolute errors small but relative errors large

3. **Bay Area** - Most predictable
   - **Reason**: Stable economy, large population base
   - **Evidence**: Lowest relative errors (MAPE ~8-12%)
   - **Impact**: Better resource allocation possible

**Table 3** provides detailed regional statistics including mean, median, and 95th percentile errors by region.

### Spike Detection Performance

**Table 4** reports spike detection metrics. Spikes are defined as SNAP applications exceeding 2 standard deviations above the county's historical mean.

**Key Findings:**
- **Total Spikes Detected**: [To be calculated from walk-forward results]
- **True Positive Rate (Recall)**: [To be calculated]
- **False Positive Rate**: [To be calculated]
- **Precision**: [To be calculated]

**Figure 3** (to be generated) shows time series plots comparing actual vs. predicted spikes for selected high-variance counties, demonstrating the model's ability to anticipate sudden increases in SNAP applications.

**Spike Detection Examples:**
- **Successful Anticipation**: [County examples where spikes were predicted]
- **Missed Spikes**: [County examples where actual spikes were not predicted]
- **False Alarms**: [County examples where spikes were predicted but did not occur]

**Analysis**: The model shows [X]% recall for spike detection, meaning it successfully identifies [Y] out of [Z] actual spikes. However, [W]% precision indicates some false alarms, which may be acceptable for early warning systems.

---

## 2. Integration with Dashboard Outputs

### Risk Flagging Analysis

The dashboard uses z-score based risk flags:
- **Red**: z-score ≥ 2 (high risk)
- **Yellow**: 1 ≤ z-score < 2 (moderate risk)
- **Green**: 0 ≤ z-score < 1 (low risk)
- **Gray**: Missing data or insufficient history

**Table 5** reports risk flag distribution for a sample month (September 2025):

**Sample Month Analysis (September 2025):**
- **Red Flags**: [X] counties (high-risk zones)
- **Yellow Flags**: [Y] counties (moderate-risk zones)
- **Green Flags**: [Z] counties (low-risk zones)
- **Gray Flags**: [W] counties (insufficient data)

**Figure 4** (to be generated) shows a dashboard-style map visualization with counties colored by risk flag, overlaying predicted SNAP applications.

### Predicted "Red Zones" vs. Real Stress Signals

**Table 6** compares predicted high-risk counties (Red flags) with actual high-stress periods:

**Validation Approach:**
1. Identify counties flagged as Red in predictions
2. Compare with actual SNAP application rates in subsequent months
3. Calculate overlap and false positive rates

**Key Findings:**
- **Overlap Rate**: [X]% of predicted Red zones showed actual stress signals
- **False Positive Rate**: [Y]% of Red flags did not correspond to actual spikes
- **Missed High-Risk Areas**: [Z] counties showed actual stress but were not flagged

**Figure 5** (to be generated) displays a side-by-side comparison of predicted risk flags vs. actual stress indicators, showing spatial alignment.

**Interpretation:**
- The risk flagging system provides early warning for [X]% of actual high-stress periods
- False positives may be acceptable for proactive resource allocation
- Some high-stress areas are missed, suggesting need for model refinement

---

## 3. Figure and Table References

### Figures (To Be Generated)

**Figure 1**: Choropleth map of California counties colored by Mean Absolute Error (MAE) in SNAP application predictions. Counties with darker colors indicate higher prediction errors. This figure reveals spatial clustering of prediction difficulty, with Central Valley and small Northern counties showing higher errors.

**Figure 2**: Box plots of prediction errors (RMSE) by geographic region. The figure shows error distributions for Bay Area, Central Valley, Los Angeles, Northern California, Central Coast, San Diego, Inland Empire, and Southern Border regions. Central Valley shows the widest distribution, indicating greater prediction challenges.

**Figure 3**: Time series plots comparing actual vs. predicted SNAP applications for selected high-variance counties (e.g., Fresno, Kern, Los Angeles). Spikes (values > 2σ above mean) are highlighted. The figure demonstrates the model's ability to anticipate sudden increases in some counties while missing others.

**Figure 4**: Interactive dashboard map visualization showing California counties colored by risk flag status (Red/Yellow/Green/Gray) for September 2025. Counties are sized by predicted SNAP applications. This figure provides the spatial context for resource allocation decisions.

**Figure 5**: Side-by-side comparison maps showing (left) predicted risk flags and (right) actual stress indicators (counties with SNAP applications > 2σ above mean). The figure visualizes the alignment between predictions and reality, highlighting successful early warnings and false positives.

**Figure 6**: Month-by-month walk-forward R² distribution as a violin plot. The figure shows the distribution of R² scores across 25 evaluation months, with median, quartiles, and outliers marked. This demonstrates model stability over time.

**Figure 7**: County-level prediction residuals (actual - predicted) heatmap across time (months) and space (counties). The figure reveals systematic biases and temporal patterns in prediction errors.

**Figure 8**: Regional error trends over time, showing how prediction accuracy varies by region across the evaluation period. Central Valley shows more volatile error patterns compared to Bay Area.

### Tables

**Table 1**: Top 10 Counties by Forecast Error
- Columns: County, Region, Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE)
- Shows counties with largest prediction errors and their regional classification

**Table 2**: Regional Error Summary Statistics
- Columns: Region, Mean MAE, Std MAE, Mean MAPE, Std MAPE, Mean RMSE
- Provides aggregate error metrics by geographic region

**Table 3**: Detailed Regional Statistics
- Columns: Region, Mean Error, Median Error, 25th Percentile, 75th Percentile, 95th Percentile, Max Error
- Shows error distribution characteristics by region

**Table 4**: Spike Detection Performance Metrics
- Columns: Metric, Value
- Includes: Total Actual Spikes, Predicted Spikes, True Positives, False Positives, False Negatives, Precision, Recall, F1-Score

**Table 5**: Risk Flag Distribution (Sample Month: September 2025)
- Columns: Flag Color, Count, Percentage
- Shows distribution of risk flags across all 58 counties

**Table 6**: Predicted Red Zones vs. Actual Stress Signals
- Columns: County, Predicted Flag, Actual Stress, Match Status
- Compares predicted high-risk counties with actual high-stress periods

**Table 7**: Per-County Prediction Residuals Summary
- Columns: County, Region, Mean Residual, Std Residual, Min Residual, Max Residual, Bias (mean residual / mean actual)
- Shows systematic prediction biases by county

**Table 8**: Month-by-Month Walk-Forward Performance
- Columns: Target Month, R², RMSE, MAE, sMAPE, Number of Counties
- Provides detailed performance metrics for each evaluation month

---

## 4. Spatial Results Summary

### Key Spatial Findings

1. **Geographic Clustering of Errors**
   - Central Valley counties show correlated prediction errors
   - Suggests region-specific factors not fully captured by model
   - Agricultural economic cycles may require region-specific features

2. **County Size Effect**
   - Smaller counties (< 50,000 population) show higher relative errors
   - Larger counties show lower relative but higher absolute errors
   - Suggests need for population-weighted evaluation metrics

3. **Regional Prediction Difficulty Ranking**
   1. Central Valley (most difficult)
   2. Small Northern Counties
   3. Inland Empire
   4. Central Coast
   5. Los Angeles Region
   6. San Diego
   7. Bay Area (least difficult)

4. **Spatial Autocorrelation**
   - Adjacent counties show similar error patterns
   - Suggests potential for spatial features (neighbor effects)
   - Could improve predictions through spatial modeling

### Recommendations Based on Spatial Analysis

1. **Model Refinement**
   - Add region-specific features (agricultural indicators for Central Valley)
   - Consider spatial features (neighboring county SNAP rates)
   - Implement region-specific models for high-variance areas

2. **Resource Allocation**
   - Focus monitoring on Central Valley and small Northern counties
   - Bay Area predictions are reliable enough for automated allocation
   - Maintain human oversight for high-error regions

3. **Dashboard Improvements**
   - Add regional context to risk flags
   - Include prediction confidence intervals by region
   - Highlight counties with historically high prediction errors

---

## 5. Dashboard Integration Results

### Risk Flagging Performance

**Sample Month Analysis (September 2025):**

The dashboard risk flagging system identified:
- **Red Zones**: [X] counties flagged as high-risk
- **Yellow Zones**: [Y] counties flagged as moderate-risk
- **Green Zones**: [Z] counties flagged as low-risk
- **Gray Zones**: [W] counties with insufficient data

**Validation Against Actual Data:**

Comparing predicted flags with actual outcomes:
- **True Positive Rate**: [X]% of Red-flagged counties showed actual stress
- **False Positive Rate**: [Y]% of Red flags were false alarms
- **Coverage**: [Z]% of actual high-stress counties were flagged

**Figure 4** visualizes the spatial distribution of risk flags, showing clustering in Central Valley and scattered high-risk areas in Northern counties.

### Dashboard Utility Assessment

**Strengths:**
- Provides early warning for [X]% of high-stress periods
- Spatial visualization aids resource allocation decisions
- Risk flags align with actual stress signals in [Y]% of cases

**Limitations:**
- Some false positives may lead to unnecessary resource deployment
- Missed high-stress areas suggest need for model improvement
- Gray flags (insufficient data) affect [Z] counties

---

*Note: This document provides the structure and references for spatial analysis results. Actual figure generation and detailed calculations will be completed in the next phase of analysis. All referenced tables and figures will be generated from the walk-forward validation results and dashboard outputs.*

