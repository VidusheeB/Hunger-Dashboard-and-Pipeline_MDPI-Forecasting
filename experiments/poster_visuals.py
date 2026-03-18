"""
Generate research poster visuals and results tables.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = 'artifacts/poster'
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ['Population', 'Median_Income', 'monthly_average_CalFresh', 'monthly_average_FoodBank', 'month']


def load_data():
    df = pd.read_csv('src/data/aggregateTrends_scaled.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['county', 'date']).reset_index(drop=True)
    df['SNAP_target'] = df.groupby('county')['SNAP_Application_Rate'].shift(-1)
    df['target_month'] = df['date'] + pd.DateOffset(months=1)
    df = df.dropna(subset=['SNAP_target', 'target_month'] + FEATURES)
    return df


def calc_metrics(yt, yp):
    yp = np.clip(yp, 0, None)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    return {
        'R²': r2_score(yt, yp),
        'RMSE': np.sqrt(mean_squared_error(yt, yp)),
        'MAE': mean_absolute_error(yt, yp),
        'sMAPE (%)': np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100,
    }


def make_models():
    return {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
        'XGBoost (default)': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        'XGBoost (tuned)': xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.01,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.9,
            reg_lambda=4, reg_alpha=0, random_state=42, n_jobs=-1,
        ),
    }


def walk_forward(df, model_factory, gap=0):
    """Returns (metrics_dict, actuals_array, predictions_array, months_array)."""
    unique_months = sorted(df['target_month'].unique())
    n = len(unique_months)
    start = max(10, 5 + gap)

    all_pred, all_actual, all_months = [], [], []

    for i in range(start, n):
        T = unique_months[i]
        cutoff = unique_months[i - gap] if gap > 0 else T
        train_df = df[df['target_month'] < cutoff]
        test_df = df[df['target_month'] == T]

        X_tr, y_tr = train_df[FEATURES], train_df['SNAP_target']
        X_te, y_te = test_df[FEATURES], test_df['SNAP_target']

        ok_tr = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
        ok_te = np.isfinite(X_te).all(axis=1)
        X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]
        X_te, y_te = X_te[ok_te], y_te[ok_te]

        if len(X_tr) < 50 or len(X_te) < 3:
            continue

        model = model_factory()
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)

        all_pred.extend(yp)
        all_actual.extend(y_te.values)
        all_months.extend([pd.Timestamp(T)] * len(y_te))

    return (np.array(all_actual), np.array(all_pred), np.array(all_months))


def plot_actual_vs_predicted(actual, predicted, title, filename):
    """Scatter plot of actual vs predicted SNAP rates."""
    fig, ax = plt.subplots(figsize=(7, 6))

    predicted = np.clip(predicted, 0, None)

    ax.scatter(actual * 100, predicted * 100, alpha=0.15, s=12, c='#2563eb', edgecolors='none')

    mn = min(actual.min(), predicted.min()) * 100
    mx = max(actual.max(), predicted.max()) * 100
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1, alpha=0.6, label='Perfect prediction')

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    ax.text(0.05, 0.92, f'R² = {r2:.3f}\nMAE = {mae:.6f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.set_xlabel('Actual SNAP Application Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted SNAP Application Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_time_series(actual, predicted, months, title, filename):
    """Aggregate actual vs predicted over time (monthly mean across all counties)."""
    df_ts = pd.DataFrame({'actual': actual, 'predicted': predicted, 'month': months})
    monthly = df_ts.groupby('month').mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly['actual'] * 100, 'o-', color='#1e40af', markersize=4, label='Actual (mean across counties)')
    ax.plot(monthly.index, monthly['predicted'] * 100, 's--', color='#dc2626', markersize=4, label='Predicted (mean across counties)')

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('SNAP Application Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_feature_importance(model, features, filename):
    """Horizontal bar chart of feature importance."""
    imp = model.feature_importances_
    idx = np.argsort(imp)

    labels = {
        'Population': 'Population',
        'Median_Income': 'Median Income',
        'monthly_average_CalFresh': 'CalFresh Trend',
        'monthly_average_FoodBank': 'FoodBank Trend',
        'month': 'Month (Seasonality)',
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(features))
    colors = ['#2563eb' if 'trend' not in features[i].lower() and 'average' not in features[i].lower()
              else '#16a34a' for i in idx]

    ax.barh(y_pos, imp[idx], color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels.get(features[i], features[i]) for i in idx], fontsize=11)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Feature Importance — XGBoost (Tuned)', fontsize=13, fontweight='bold')

    # Add value labels
    for i, v in enumerate(imp[idx]):
        ax.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_gap_analysis(df, filename):
    """Bar chart showing model performance across different data gaps."""
    gaps = [0, 1, 2, 3, 4, 5]
    maes = []
    smapes = []

    model_factory = lambda: xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.01,
        min_child_weight=6, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=4, reg_alpha=0, random_state=42, n_jobs=-1)

    for gap in gaps:
        actual, predicted, _ = walk_forward(df, model_factory, gap=gap)
        m = calc_metrics(actual, predicted)
        maes.append(m['MAE'])
        smapes.append(m['sMAPE (%)'])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(gaps))
    bars = ax1.bar(x, [s for s in smapes], color='#2563eb', width=0.5, alpha=0.85)
    ax1.set_xlabel('SNAP Data Gap (months)', fontsize=12)
    ax1.set_ylabel('sMAPE (%)', fontsize=12, color='#2563eb')
    ax1.set_xticks(x)
    labels = [str(g) for g in gaps]
    labels[0] = '0\n(no gap)'
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#2563eb')

    # Add value labels on bars
    for bar, val in zip(bars, smapes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color='#2563eb')

    ax1.set_title('Prediction Robustness to SNAP Data Lag', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(smapes) * 1.25)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def plot_model_comparison(results_df, filename):
    """Grouped bar chart comparing models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    models = results_df['Model'].values
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_df['sMAPE (%)'], width, label='sMAPE (%)', color='#2563eb', alpha=0.85)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, results_df['MAE'] * 1000, width, label='MAE (×10³)', color='#dc2626', alpha=0.85)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('sMAPE (%)', fontsize=12, color='#2563eb')
    ax2.set_ylabel('MAE (×10³)', fontsize=12, color='#dc2626')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.tick_params(axis='y', labelcolor='#2563eb')
    ax2.tick_params(axis='y', labelcolor='#dc2626')

    ax.set_title('Model Comparison — Walk-Forward Chronological Validation', fontsize=13, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved {filename}')


def main():
    df = load_data()
    print(f'Data: {len(df)} rows, {df["county"].nunique()} counties\n')

    models = make_models()
    gap = 0  # standard walk-forward (no assumed lag)

    # ── Run all models with gap=0 ──
    print('Running walk-forward validation (gap=0) for all models...')
    all_results = []

    for name, model_fn in models.items():
        factory = (lambda m=model_fn: m.__class__(**m.get_params()))
        actual, predicted, months = walk_forward(df, factory, gap=gap)
        metrics = calc_metrics(actual, predicted)
        metrics['Model'] = name
        metrics['Predictions'] = len(actual)
        all_results.append(metrics)
        print(f'  {name}: sMAPE={metrics["sMAPE (%)"]:.2f}%, MAE={metrics["MAE"]:.6f}')

        # Save actuals/predictions for the tuned XGBoost (for scatter + time series)
        if 'tuned' in name.lower():
            best_actual, best_predicted, best_months = actual, predicted, months

    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Model', 'R²', 'RMSE', 'MAE', 'sMAPE (%)', 'Predictions']]
    results_df = results_df.sort_values('MAE')

    # ── Print results table ──
    print('\n' + '=' * 90)
    print('MODEL COMPARISON — Walk-Forward Chronological Validation')
    print('=' * 90)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    results_df.to_csv(os.path.join(OUT_DIR, 'model_comparison_table.csv'), index=False)
    print(f'\n  Table saved to {OUT_DIR}/model_comparison_table.csv')

    # ── Generate plots ──
    print('\nGenerating plots...')

    # 1. Actual vs Predicted scatter
    plot_actual_vs_predicted(best_actual, best_predicted,
        'Actual vs Predicted SNAP Application Rates\n(Walk-Forward Chronological Validation)',
        'actual_vs_predicted.png')

    # 2. Time series
    plot_time_series(best_actual, best_predicted, best_months,
        'SNAP Application Rate — Actual vs Predicted Over Time\n(Monthly Mean Across Counties)',
        'time_series.png')

    # 3. Feature importance (train on full data for this)
    full_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.01,
        min_child_weight=6, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=4, reg_alpha=0, random_state=42, n_jobs=-1)
    X_full = df[FEATURES]
    y_full = df['SNAP_target']
    ok = np.isfinite(y_full) & np.isfinite(X_full).all(axis=1)
    full_model.fit(X_full[ok], y_full[ok])
    plot_feature_importance(full_model, FEATURES, 'feature_importance.png')

    # 4. Gap analysis
    print('\nRunning gap analysis (0-5 months)...')
    plot_gap_analysis(df, 'gap_analysis.png')

    # 5. Model comparison bar chart
    plot_model_comparison(results_df, 'model_comparison.png')

    print(f'\nAll outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
