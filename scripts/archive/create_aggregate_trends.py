import os
import pandas as pd
import glob

def create_aggregate_trends():
    """
    Create aggregateTrends.csv by combining keyword trend data with metro areas and SNAP applications.
    """
    
    # File paths
    trends_base_dir = "src/data/trends"
    county_metro_file = "src/data/county_to_metro.csv"
    snap_data_file = "src/data/SNAPApps/SNAPData.csv"
    pop_data_file = "src/data/popData.csv"
    output_file = "src/data/aggregateTrends.csv"
    
    # Automatically detect keyword directories
    if not os.path.exists(trends_base_dir):
        print(f"Error: Trends directory not found: {trends_base_dir}")
        return None
    
    keyword_dirs = []
    for item in os.listdir(trends_base_dir):
        item_path = os.path.join(trends_base_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            keyword_dirs.append(item_path)
    
    if not keyword_dirs:
        print(f"No keyword directories found in {trends_base_dir}")
        return None
    
    print(f"Found keyword directories: {[os.path.basename(d) for d in keyword_dirs]}")
    
    # Read county to metro mapping
    county_metro_df = pd.read_csv(county_metro_file)
    county_metro_df.columns = county_metro_df.columns.str.strip()
    
    # Read SNAP data
    snap_df = pd.read_csv(snap_data_file, header=None, names=["county", "month_year", "SNAP_Applications"])
    snap_df["date"] = pd.to_datetime(snap_df["month_year"], format="%b %Y", errors="coerce")
    if bool(snap_df["date"].isna().any()):
        snap_df.loc[snap_df["date"].isna(), "date"] = pd.to_datetime(
            snap_df.loc[snap_df["date"].isna(), "month_year"], format="%B %Y", errors="coerce"
        )
    snap_df["SNAP_Applications"] = pd.to_numeric(snap_df["SNAP_Applications"], errors="coerce")
    
    # Shift SNAP data forward by one month to create prediction relationship
    # Month N trends should predict Month N+1 SNAP applications
    snap_df["trend_date"] = snap_df["date"] - pd.DateOffset(months=1)
    print(f"Shifted SNAP data: Month N trends will predict Month N+1 SNAP applications")
    print(f"Example: May 2022 trends predict June 2022 SNAP applications")
    
    # Read population data
    pop_df = pd.read_csv(pop_data_file)
    pop_df.columns = pop_df.columns.str.strip()
    
    # Initialize result DataFrame
    result_data = []
    
    # Get all unique counties from SNAP data
    counties = snap_df['county'].unique()
    
    # Get all unique dates from SNAP data
    dates = snap_df['date'].unique()
    
    # Create base DataFrame with all county-date combinations
    base_df = pd.DataFrame([
        {'county': county, 'date': date} 
        for county in counties 
        for date in dates
    ])
    
    # Add metro area mapping
    base_df = base_df.merge(county_metro_df, left_on='county', right_on='county', how='left')
    
    # Add SNAP applications
    base_df = base_df.merge(
        snap_df[['county', 'trend_date', 'SNAP_Applications']], 
        left_on=['county', 'date'], 
        right_on=['county', 'trend_date'], 
        how='left'
    )
    base_df = base_df.drop(columns=['trend_date'])
    
    # Merge population into base_df
    base_df = base_df.merge(pop_df[['County', 'Population']], left_on='county', right_on='County', how='left')
    base_df = base_df.drop(columns=['County'])
    
    # Process each keyword directory
    keyword_names = []
    for keyword_dir in keyword_dirs:
        keyword_name = os.path.basename(keyword_dir)
        keyword_names.append(keyword_name)
        
        # Get all CSV files in the keyword directory
        csv_files = glob.glob(os.path.join(keyword_dir, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {keyword_dir}")
            continue
            
        print(f"Processing {keyword_name} with {len(csv_files)} CSV files")
            
        # Read and combine all CSV files for this keyword
        keyword_data = []
        for csv_file in csv_files:
            metro_area_name = os.path.splitext(os.path.basename(csv_file))[0]
            # Read CSV without headers - first column is date, second is value
            df = pd.read_csv(csv_file, header=None, names=['date', 'weekly_value'])
            df['metro_area'] = metro_area_name
            df['date'] = pd.to_datetime(df['date'])
            # Add year and month columns
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            # Group by year, month, metro_area and compute monthly average
            monthly_avg = df.groupby(['metro_area', 'year', 'month'])['weekly_value'].mean().reset_index()
            monthly_avg = monthly_avg.rename(columns={'weekly_value': 'monthly_average'})
            keyword_data.append(monthly_avg)
        
        if keyword_data:
            keyword_df = pd.concat(keyword_data, ignore_index=True)
            # Create a monthly date for merging (first of month)
            keyword_df['date'] = pd.to_datetime(keyword_df[['year', 'month']].assign(day=1))
            # Merge with base DataFrame
            base_df = base_df.merge(
                keyword_df[['metro_area', 'date', 'monthly_average']], 
                on=['metro_area', 'date'], 
                how='left',
                suffixes=('', f'_{keyword_name}')
            )
            # Rename the monthly_average column to include keyword name
            base_df = base_df.rename(columns={'monthly_average': f'monthly_average_{keyword_name}'})
    
    # Clean up the DataFrame
    # Remove rows where metro_area is NaN (counties not mapped to metros)
    base_df = base_df.dropna(subset=['metro_area'])
    
    # Sort by county and date
    base_df = base_df.sort_values(['county', 'date'])
    
    # Save to CSV
    base_df.to_csv(output_file, index=False)
    
    print(f"Created {output_file}")
    print(f"Shape: {base_df.shape}")
    print(f"Columns: {list(base_df.columns)}")
    print(f"Keywords included: {keyword_names}")
    
    # Show sample data
    print("\nSample data:")
    print(base_df.head())
    
    return base_df

if __name__ == "__main__":
    create_aggregate_trends() 