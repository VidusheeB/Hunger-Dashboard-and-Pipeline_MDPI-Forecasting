#!/bin/bash

# Runner script for the trend lags experiment
# Tests whether lagged Google Trends + rolling means improve next-month SNAP predictions

echo "=========================================="
echo "Running Trend Lags Experiment"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "src/data/aggregateTrends_scaled.csv" ]; then
    echo "Error: aggregateTrends_scaled.csv not found. Please run from project root."
    exit 1
fi

# Create artifacts directory if it doesn't exist
mkdir -p artifacts/experiments

# Run the experiment
echo "Starting experiment..."
python experiments/test_trend_lags_predict_spikes.py

# Check if results were created
if [ -f "artifacts/experiments/trend_lags_cv_metrics.csv" ] && [ -f "artifacts/experiments/trend_lags_cv_summary.json" ]; then
    echo ""
    echo "=========================================="
    echo "Experiment completed successfully!"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - artifacts/experiments/trend_lags_cv_metrics.csv"
    echo "  - artifacts/experiments/trend_lags_cv_summary.json"
    echo ""
    echo "Quick summary:"
    python -c "
import pandas as pd
import json

# Load results
results = pd.read_csv('artifacts/experiments/trend_lags_cv_metrics.csv')
with open('artifacts/experiments/trend_lags_cv_summary.json', 'r') as f:
    summary = json.load(f)

print(f'Folds: {len(results)}')
print(f'Avg R²: {summary[\"r2_mean\"]:.4f} ± {summary[\"r2_std\"]:.4f}')
print(f'Avg RMSE: {summary[\"rmse_mean\"]:.6f}')
print(f'Avg MAE: {summary[\"mae_mean\"]:.6f}')
if 'r2_spike_mean' in summary and summary['r2_spike_mean'] != 'nan':
    print(f'Avg R² (spikes): {summary[\"r2_spike_mean\"]:.4f}')
    print(f'Total spikes: {summary[\"total_spikes\"]}')
"
else
    echo "Error: Experiment failed to produce expected output files."
    exit 1
fi
