import os
import pandas as pd
import numpy as np
from pathlib import Path


def _read_prediction_csv(csv_path):
    """Read a prediction CSV regardless of whether it has 1 or 2 header rows.
    Returns (df with columns ['date','value'], col_header_line) where
    col_header_line is the original second-column header string (e.g. 'CalFresh')."""
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Detect format: new files start with "Time","..." as row 0
    # Old files have "Category: All categories" on row 0
    if lines and lines[0].strip().startswith('"Time"'):
        # New format: single header row
        col_header = lines[0].strip().split(',', 1)[1].strip().strip('"')
        skiprows = 1
    else:
        # Old format: two header rows (category + column names)
        col_header = lines[1].strip().split(',', 1)[1].strip() if len(lines) > 1 else 'value'
        skiprows = 2

    data_lines = lines[skiprows:]
    rows = []
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            rows.append([parts[0].strip().strip('"'), parts[1].strip()])
    df = pd.DataFrame(rows, columns=['date', 'value'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df, col_header


def _write_prediction_csv(csv_path, df, col_header):
    """Write prediction data back using the new single-header format."""
    with open(csv_path, 'w') as f:
        f.write(f'"Time","{col_header}"\n')
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            f.write(f'"{date_str}",{row["value"]:.1f}\n')


def fill_zero_values_in_metro_keywords(
    prediction_dir="src/data/prediction",
    training_file="src/data/aggregateTrends.csv"
):
    """
    For prediction files that contain zero values:
    - If the file has >= 4 nonzero rows: fill zeros with the mean of its nonzero values
    - Otherwise (almost no signal): fill zeros with the national training average
      (last 6 months from aggregateTrends.csv)

    Args:
        prediction_dir: Directory with CalFresh/, FoodBank/ subdirs
        training_file: Path to aggregateTrends.csv with training data
    """
    print(f"Loading training data from {training_file}...")
    train_df = pd.read_csv(training_file, parse_dates=['date'])
    train_df = train_df.sort_values('date')

    results = {'filled': [], 'skipped': []}
    keywords = ['CalFresh', 'FoodBank']

    for keyword in keywords:
        print(f"\n{'='*80}")
        print(f"Filling zeros for keyword: {keyword}")
        print(f"{'='*80}")

        keyword_dir = os.path.join(prediction_dir, keyword)
        if not os.path.exists(keyword_dir):
            print(f"Skipping {keyword_dir} (not found)")
            continue

        # National training average as fallback for nearly-blank DMAs
        col = f'monthly_average_{keyword}'
        if col in train_df.columns:
            recent_data = train_df.sort_values('date').tail(6)
            national_avg = recent_data[col].mean()
        else:
            national_avg = None
        print(f"National training avg (last 6 mo): {national_avg:.2f}" if national_avg else "National avg not available")

        csv_files = sorted([f for f in os.listdir(keyword_dir) if f.endswith('.csv')])
        for csv_file in csv_files:
            metro = csv_file.replace('.csv', '')
            csv_path = os.path.join(keyword_dir, csv_file)
            try:
                df, col_header = _read_prediction_csv(csv_path)
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

                zero_mask = df['value'] == 0
                n_zeros = zero_mask.sum()
                if n_zeros == 0:
                    results['skipped'].append(f"{metro} ({keyword}) - no zeros")
                    continue

                nonzero_vals = df.loc[~zero_mask, 'value'].dropna()
                n_nonzero = len(nonzero_vals)

                if n_nonzero >= 4:
                    fill_val = nonzero_vals.mean()
                    source = f"local mean of {n_nonzero} nonzero rows"
                elif national_avg is not None and national_avg > 0:
                    fill_val = national_avg
                    source = "national training avg (too few nonzero rows locally)"
                else:
                    print(f"  {metro}: {n_zeros} zeros, {n_nonzero} nonzero - no fill value available, skipping")
                    results['skipped'].append(f"{metro} ({keyword}) - no fill value")
                    continue

                df['value'] = df['value'].astype(float)
                df.loc[zero_mask, 'value'] = round(fill_val, 1)
                _write_prediction_csv(csv_path, df, col_header)
                print(f"  {metro}: filled {n_zeros} zeros with {fill_val:.1f} ({source})")
                results['filled'].append(f"{metro} ({keyword}): {n_zeros} zeros → {fill_val:.1f}")

            except Exception as e:
                print(f"  ERROR {metro}: {str(e)}")

    return results


def fill_missing_metro_keywords(
    prediction_dir="src/data/prediction",
    training_file="src/data/aggregateTrends.csv"
):
    """
    For metros missing data for a keyword:
    1. Get the standard date range from metros that HAVE data for that keyword
    2. Calculate average of last 6 months from training data for that keyword
    3. Fill missing metro/keyword files with those dates + average value

    Args:
        prediction_dir: Directory with CalFresh/, FoodBank/ subdirs
        training_file: Path to aggregateTrends.csv with training data
    """

    print(f"Loading training data from {training_file}...")
    train_df = pd.read_csv(training_file, parse_dates=['date'])
    train_df = train_df.sort_values('date')

    results = {'filled': [], 'skipped': []}
    keywords = ['CalFresh', 'FoodBank']

    for keyword in keywords:
        print(f"\n{'='*80}")
        print(f"Processing keyword: {keyword}")
        print(f"{'='*80}")

        keyword_dir = os.path.join(prediction_dir, keyword)
        if not os.path.exists(keyword_dir):
            print(f"Skipping {keyword_dir} (not found)")
            continue

        csv_files = sorted([f for f in os.listdir(keyword_dir) if f.endswith('.csv')])
        print(f"Found {len(csv_files)} metro files")

        # Step 1: Find standard date range from metros that HAVE data
        all_dates = []
        for csv_file in csv_files:
            csv_path = os.path.join(keyword_dir, csv_file)
            try:
                df, _ = _read_prediction_csv(csv_path)
                if len(df) > 0:
                    all_dates.extend(df['date'].tolist())
            except:
                pass

        if all_dates:
            standard_dates = sorted(set(all_dates))
            print(f"Standard date range: {standard_dates[0].strftime('%Y-%m-%d')} to {standard_dates[-1].strftime('%Y-%m-%d')}")
            print(f"Total dates: {len(standard_dates)}")
        else:
            print(f"No dates found for {keyword}")
            continue

        # Step 2: Calculate average of last 6 months for this keyword
        col = f'monthly_average_{keyword}'
        if col in train_df.columns:
            # Get last 6 months
            recent_data = train_df.sort_values('date').tail(6)
            avg_value = recent_data[col].mean()
            print(f"Average of last 6 months (training): {avg_value:.2f}")
        else:
            print(f"Column {col} not found in training data")
            continue

        # Step 3: Check each metro and fill if missing
        for csv_file in csv_files:
            metro = csv_file.replace('.csv', '')
            csv_path = os.path.join(keyword_dir, csv_file)

            try:
                df, _ = _read_prediction_csv(csv_path)

                # Check if metro has data
                if len(df) == 0:
                    print(f"  {metro}: EMPTY - Creating with {len(standard_dates)} dates, avg={avg_value:.2f}")

                    new_df = pd.DataFrame({
                        'date': standard_dates,
                        'value': [avg_value] * len(standard_dates)
                    })
                    _write_prediction_csv(csv_path, new_df, keyword)
                    results['filled'].append(f"{metro} ({keyword})")
                else:
                    results['skipped'].append(f"{metro} ({keyword}) - has {len(df)} rows")

            except Exception as e:
                print(f"  ERROR {metro}: {str(e)}")

    return results

def main():
    print("="*80)
    print("PREPROCESSING: Step 1 — Fill completely missing metro files")
    print("="*80)
    results = fill_missing_metro_keywords()
    print(f"\nFilled (empty files): {len(results['filled'])}")
    for item in results['filled']:
        print(f"  ✓ {item}")

    print("\n" + "="*80)
    print("PREPROCESSING: Step 2 — Fill zero values within existing files")
    print("="*80)
    zero_results = fill_zero_values_in_metro_keywords()
    print(f"\nFilled (zero rows): {len(zero_results['filled'])}")
    for item in zero_results['filled']:
        print(f"  ✓ {item}")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
