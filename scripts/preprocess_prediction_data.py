"""
Preprocess Prediction Data

Run this BEFORE retraining when new prediction CSVs are uploaded.
Only task: create placeholder files for any DMA that is missing a prediction file.
Zeros in existing files are kept as-is — they are real low-activity signal.
"""

import os
import json
import pandas as pd

PREDICTION_DIR   = "src/data/prediction"
SCALING_PARAMS   = "src/data/trend_scaling_params.json"
KEYWORDS         = ["CalFresh", "FoodBank"]


def _read_prediction_csv(csv_path):
    """Read a prediction CSV. Returns (df with columns date/value, keyword_name)."""
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('"Time"') or s.startswith('Time,') or s.startswith('Day,'):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(columns=['date', 'value']), 'value'

    col_header = lines[header_idx].strip().split(',', 1)[1].strip().strip('"')
    data_lines = lines[header_idx + 1:]
    rows = []
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            rows.append([parts[0].strip().strip('"'), parts[1].strip()])

    df = pd.DataFrame(rows, columns=['date', 'value'])
    df['date']  = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df, col_header


def _write_prediction_csv(csv_path, df, col_header):
    """Write prediction data in the standard single-header format."""
    with open(csv_path, 'w') as f:
        f.write(f'"Time","{col_header}"\n')
        for _, row in df.iterrows():
            f.write(f'"{row["date"].strftime("%Y-%m-%d")}",{row["value"]:.1f}\n')


def fill_missing_metro_files():
    """
    For each keyword, if a DMA file is missing or empty, create a placeholder
    using the standard date range from existing files and the training average
    for that DMA (from trend_scaling_params.json).

    Zeros in existing files are NOT modified — they represent real low search volume.
    """
    # Load training averages for fallback fill values
    scaling_params = {}
    if os.path.exists(SCALING_PARAMS):
        with open(SCALING_PARAMS) as f:
            scaling_params = json.load(f)
    else:
        print(f"WARNING: {SCALING_PARAMS} not found. Run build_training_data.py first.")

    for keyword in KEYWORDS:
        kw_dir = os.path.join(PREDICTION_DIR, keyword)
        if not os.path.exists(kw_dir):
            print(f"Skipping {kw_dir} (not found)")
            continue

        csv_files = sorted([f for f in os.listdir(kw_dir) if f.endswith('.csv')])
        print(f"\n{keyword}: {len(csv_files)} files found")

        # Find the standard date range from whichever files have data
        all_dates = []
        for csv_file in csv_files:
            df, _ = _read_prediction_csv(os.path.join(kw_dir, csv_file))
            all_dates.extend(df['date'].tolist())

        if not all_dates:
            print(f"  No dates found for {keyword}, skipping")
            continue

        standard_dates = sorted(set(all_dates))
        print(f"  Date range: {standard_dates[0].date()} → {standard_dates[-1].date()} ({len(standard_dates)} dates)")

        # Global fallback: mean training avg across all DMAs for this keyword
        kw_params = scaling_params.get(keyword, {})
        global_avg = sum(kw_params.values()) / len(kw_params) if kw_params else 50.0

        for csv_file in csv_files:
            metro = csv_file.replace('.csv', '')
            csv_path = os.path.join(kw_dir, csv_file)
            df, _ = _read_prediction_csv(csv_path)

            if len(df) == 0:
                fill_val = kw_params.get(metro, global_avg)
                new_df = pd.DataFrame({
                    'date':  standard_dates,
                    'value': [fill_val] * len(standard_dates)
                })
                _write_prediction_csv(csv_path, new_df, keyword)
                print(f"  CREATED {metro}: {len(standard_dates)} rows, value={fill_val:.1f}")


def main():
    print("=" * 60)
    print("PREDICTION DATA PREPROCESSING")
    print("Filling missing DMA files only. Zeros are kept as-is.")
    print("=" * 60)
    fill_missing_metro_files()
    print("\nDone.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
