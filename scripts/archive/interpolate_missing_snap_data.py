import pandas as pd
import numpy as np
import os
from datetime import datetime

def interpolate_missing_snap_data():
    """
    Interpolate missing SNAP data in aggregateTrends.csv using linear interpolation:
    - Linear interpolation for missing values
    - Forward fill and backward fill for any remaining edge cases
    """
    # File paths
    input_file = "src/data/aggregateTrends.csv"
    output_file = "src/data/aggregateTrends_linear_interpolated.csv"
    print("Loading aggregateTrends.csv...")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Original data shape: {df.shape}")
    print(f"Original missing SNAP_Applications: {df['SNAP_Applications'].isna().sum()}")
    df_original = df.copy()
    counties = df['county'].unique()
    print(f"\nProcessing {len(counties)} counties...")
    interpolation_stats = {
        'county': [],
        'original_missing': [],
        'after_linear': [],
        'final_missing': []
    }
    for county in counties:
        county_data = df[df['county'] == county].copy()
        original_missing = county_data['SNAP_Applications'].isna().sum()
        if original_missing == 0:
            continue
        print(f"\nProcessing {county}: {original_missing} missing values")
        county_data = county_data.sort_values('date')
        # Linear interpolation
        county_data['SNAP_Applications_linear'] = county_data['SNAP_Applications'].interpolate(method='linear')
        after_linear = int(county_data['SNAP_Applications_linear'].isna().sum())
        # Fallback: ffill then bfill
        county_data['SNAP_Applications_interpolated'] = (
            county_data['SNAP_Applications_linear']
            .fillna(method='ffill')
            .fillna(method='bfill')
        )
        final_missing = county_data['SNAP_Applications_interpolated'].isna().sum()
        df.loc[df['county'] == county, 'SNAP_Applications'] = county_data['SNAP_Applications_interpolated']
        interpolation_stats['county'].append(county)
        interpolation_stats['original_missing'].append(original_missing)
        interpolation_stats['after_linear'].append(after_linear)
        interpolation_stats['final_missing'].append(final_missing)
        print(f"  After linear: {after_linear} missing")
        print(f"  Final: {final_missing} missing")
    stats_df = pd.DataFrame(interpolation_stats)
    print(f"\n=== INTERPOLATION SUMMARY ===")
    print(f"Total missing values before: {df_original['SNAP_Applications'].isna().sum()}")
    print(f"Total missing values after: {df['SNAP_Applications'].isna().sum()}")
    print(f"Values filled: {df_original['SNAP_Applications'].isna().sum() - df['SNAP_Applications'].isna().sum()}")
    df.to_csv(output_file, index=False)
    print(f"\nSaved interpolated data to: {output_file}")
    stats_file = "src/data/linear_interpolation_analysis.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved linear interpolation analysis to: {stats_file}")
    print(f"\n=== DETAILED STATISTICS ===")
    print(stats_df.to_string(index=False))
    print(f"\n=== QUALITY ANALYSIS ===")
    counties_with_data = df_original[df_original['SNAP_Applications'].notna()]['county'].unique()
    quality_metrics = []
    for county in counties_with_data:
        original_data = df_original[df_original['county'] == county]
        interpolated_data = df[df['county'] == county]
        common_dates = pd.Series(list(set(original_data['date']) & set(interpolated_data['date'])))
        if len(common_dates) > 0:
            orig_values = original_data[original_data['date'].isin(common_dates)]['SNAP_Applications']
            interp_values = interpolated_data[interpolated_data['date'].isin(common_dates)]['SNAP_Applications']
            correlation = orig_values.corr(interp_values)
            mape = np.mean(np.abs((orig_values - interp_values) / orig_values)) * 100 if orig_values.sum() > 0 else 0
            quality_metrics.append({
                'county': county,
                'correlation': correlation,
                'mape': mape,
                'data_points': len(common_dates)
            })
    quality_df = pd.DataFrame(quality_metrics)
    if not quality_df.empty:
        print(f"Average correlation with original data: {quality_df['correlation'].mean():.3f}")
        print(f"Average MAPE: {quality_df['mape'].mean():.2f}%")
        quality_file = "src/data/linear_interpolation_quality.csv"
        quality_df.to_csv(quality_file, index=False)
        print(f"Saved linear quality metrics to: {quality_file}")
    return df, stats_df

if __name__ == "__main__":
    print("Starting SNAP data linear interpolation...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        interpolated_df, stats_df = interpolate_missing_snap_data()
        print("\nLinear interpolation completed successfully!")
    except Exception as e:
        print(f"Error during linear interpolation: {str(e)}")
        import traceback
        traceback.print_exc() 