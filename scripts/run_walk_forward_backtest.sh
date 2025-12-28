#!/bin/bash

# Runner script for the walk-forward backtest experiment
# Tests the most rigorous leak-free evaluation methodology

echo "=========================================="
echo "Running Walk-Forward Backtest Experiment"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "src/data/aggregateTrends_scaled.csv" ]; then
    echo "Error: aggregateTrends_scaled.csv not found. Please run from project root."
    exit 1
fi

# Create artifacts directory if it doesn't exist
mkdir -p artifacts/experiments

# Run the walk-forward backtest
echo "Starting walk-forward backtest..."
python experiments/walk_forward_backtest.py

# Check if results were created
if [ -f "artifacts/experiments/walk_forward_backtest_metrics.csv" ] && [ -f "artifacts/experiments/walk_forward_backtest_summary.json" ]; then
    echo ""
    echo "=========================================="
    echo "Walk-Forward Backtest completed successfully!"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - artifacts/experiments/walk_forward_backtest_metrics.csv"
    echo "  - artifacts/experiments/walk_forward_backtest_summary.json"
    echo ""
    echo "Quick summary:"
    python -c "
import pandas as pd
import json

# Load results
results = pd.read_csv('artifacts/experiments/walk_forward_backtest_metrics.csv')
with open('artifacts/experiments/walk_forward_backtest_summary.json', 'r') as f:
    summary = json.load(f)

print(f'Prediction months: {len(results)}')
print(f'Overall R²: {summary[\"overall_r2\"]:.4f}')
print(f'Overall RMSE: {summary[\"overall_rmse\"]:.6f}')
print(f'Overall MAE: {summary[\"overall_mae\"]:.6f}')
print(f'Overall sMAPE: {summary[\"overall_smape\"]:.2f}%')
print(f'Total predictions: {summary[\"total_predictions\"]}')
"
else
    echo "Error: Walk-forward backtest failed to produce expected output files."
    exit 1
fi
