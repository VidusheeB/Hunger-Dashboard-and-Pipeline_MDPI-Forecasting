import numpy as np
import pandas as pd

def scale_trends(df: pd.DataFrame, previous_column, current_column):
    # Init: Default to 1
    scaling_factor = 1
    print(previous_column, current_column)
    max_value_previous_year = df[previous_column].max()
    date_of_max_value_previous_year = df.loc[df[previous_column].idxmax(), 'date']
    print(f"* Previous Year High: {date_of_max_value_previous_year}")
    
    # then in the following year Y, find the previous year's high and see if there's a scale factor
    max_value_current_year = df[current_column].max()
    date_of_max_value_current_year = df.loc[df[current_column].idxmax(), 'date']
    print(f"* Current Year High: {date_of_max_value_current_year}")
    
    # if the date of the current max value is later than the previous 
    if max_value_previous_year != 0 and date_of_max_value_previous_year < date_of_max_value_current_year:
        scaled_value_of_previous_max_value = (
            df.loc[df['date'] == date_of_max_value_previous_year, previous_column].values[0]
        )
        # Find the previous values in the current year
        current_value_of_previous_max_value = (
            df.loc[df['date'] == date_of_max_value_previous_year, current_column].values[0]
        )
        
        # if the current trend does not have the max value in it - adjust with first value available.
        if np.isnan(current_value_of_previous_max_value):
            print(f"* Previous high not in current trend, using earliest date to scale")
            # Get the first non nan/non 0 date and value
            # first_available_value_date_current_year = df.loc[df[current_column].notna(), 'date'].iloc[0]
            first_available_value_date_current_year = df.loc[(df[current_column] != 0) & (df[current_column].notna()), 'date'].iloc[0]
            first_available_value_current_year = df.loc[df['date'] == first_available_value_date_current_year, current_column].values[0]
            
            # Get the corresponding value using the date in the previous column
            corresponding_value_previous_year = df.loc[df['date'] == first_available_value_date_current_year, previous_column].values[0]
            
            print(first_available_value_date_current_year, first_available_value_current_year, corresponding_value_previous_year)
            
            # If we can't find a corresponding value (no overlapping dates), use a reasonable scaling factor
            if np.isnan(corresponding_value_previous_year):
                print(f"* No overlapping dates found, using reasonable scaling factor")
                # Use the average of the training data as a reasonable scale
                avg_training_value = df[previous_column].mean()
                avg_prediction_value = df[current_column].mean()
                if avg_prediction_value > 0:
                    scaling_factor = avg_training_value / avg_prediction_value
                    print(f"* Scale change: {avg_training_value}/{avg_prediction_value} = {scaling_factor}")
                else:
                    scaling_factor = 1  # Default to no scaling
                    print(f"* No valid prediction values, using default scaling factor of 1")
            else:
                scaling_factor = corresponding_value_previous_year / first_available_value_current_year
                print(f"* Scale change: {corresponding_value_previous_year}/{first_available_value_current_year} = {scaling_factor}")
        else:
            scaling_factor = scaled_value_of_previous_max_value / current_value_of_previous_max_value
            print(f"* Scale change: {scaled_value_of_previous_max_value}/{current_value_of_previous_max_value} = {scaling_factor}")
    
    df[current_column] = df[current_column] * scaling_factor
    # Remove the line that overwrites new data with old training data
    # df[current_column] = df[previous_column].combine_first(df[current_column])
    return df


def normalize_trends_by_population(df, county_col='county', trend_cols=None, popdata_path="src/data/popData.csv"):
    """
    Normalize trend columns by county population.
    Args:
        df: DataFrame with a 'county' column and trend columns.
        county_col: Name of the county column.
        trend_cols: List of trend columns to normalize. If None, all columns except county_col are used.
        popdata_path: Path to the population data CSV.
    Returns:
        DataFrame with trend columns divided by population.
    """
    pop_df = pd.read_csv(popdata_path)
    pop_map = pop_df.set_index('County')['Population'].to_dict()
    if trend_cols is None:
        trend_cols = [col for col in df.columns if col != county_col]
    df = df.copy()
    df['__pop'] = df[county_col].map(pop_map)
    for col in trend_cols:
        df[col] = df[col] / df['__pop']
    df = df.drop(columns='__pop')
    return df 