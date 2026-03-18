import pandas as pd
import pickle
import sys
import os
import json
import logging
from datetime import datetime, timedelta
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "county_models"
COUNTY_METRO_MAP = "src/data/county_to_metro.csv"
POP_DATA_FILE = "src/data/popData.csv"
PREDICTION_BASE_DIR = "src/data/prediction"
SCALING_PARAMS_FILE = "src/data/trend_scaling_params.json"

# Always load the global model

def load_global_model():
    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "rb") as f:
        return pickle.load(f)

def predict_next_month(trends_dict, population, median_income, prediction_month=8):
    """Predict SNAP applications for next month.

    Returns:
        tuple: (predicted_applications, lower_bound, upper_bound)
    """
    if not trends_dict:
        raise ValueError("No trend data provided for prediction")

    model_info = load_global_model()
    features = model_info["features"]
    model = model_info["model"]

    # trends_dict: {"trend_keyword1": value, "trend_keyword2": value, ...}
    # Add population and median income to the features
    input_dict = {**trends_dict, "Population": population, "Median_Income": median_income}

    # Add minimal month feature (avoid overfitting)
    input_dict["month"] = prediction_month

    # Ensure all features are present
    X_pred = pd.DataFrame([input_dict], columns=features)

    # Check for missing features
    missing_features = []
    for feature in features:
        if pd.isna(X_pred[feature].iloc[0]):
            missing_features.append(feature)

    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")

    # Concept drift detection: warn if inputs are outside training range
    feature_ranges = model_info.get("feature_ranges")
    if feature_ranges:
        for feat in features:
            val = X_pred[feat].iloc[0]
            r = feature_ranges.get(feat)
            if r and (val < r['min'] or val > r['max']):
                logger.warning(
                    f"DRIFT: '{feat}' = {val:.6f} is outside training range "
                    f"[{r['min']:.6f}, {r['max']:.6f}]"
                )

    # Model now predicts rates (applications per population)
    predicted_rate = model.predict(X_pred)[0]

    # Ensure rate is never negative
    predicted_rate = max(0, predicted_rate)

    # Confidence interval based on walk-forward MAE saved during training
    BACKTEST_MAE = model_info.get("walkforward_mae", 0.000877)
    lower_rate = max(0, predicted_rate - BACKTEST_MAE)
    upper_rate = predicted_rate + BACKTEST_MAE

    # Convert rates back to absolute number of applications
    predicted_applications = predicted_rate * population
    lower_bound = lower_rate * population
    upper_bound = upper_rate * population

    return predicted_applications, lower_bound, upper_bound

def get_metro_for_county(county):
    county_metro_map = pd.read_csv(COUNTY_METRO_MAP)
    row = county_metro_map[county_metro_map["county"] == county]
    if row.empty:
        raise ValueError(f"No metro found for county {county}")
    return row.iloc[0]["metro_area"]

def get_population_for_county(county):
    pop_df = pd.read_csv(POP_DATA_FILE)
    pop_df.columns = pop_df.columns.str.strip()
    row = pop_df[pop_df["County"] == county]
    if row.empty:
        raise ValueError(f"No population found for county {county}")
    return float(row.iloc[0]["Population"])

def get_median_income_for_county(county):
    try:
        income_data = pd.read_csv("src/data/MedianIncome.csv")
        # Clean the median income column (remove commas and convert to numeric)
        income_data['Median_Income'] = income_data['Median Income'].str.replace(',', '').astype(float)
        
        # First try exact match
        row = income_data[income_data["County"] == county]
        
        # If not found, try with spaces removed (for cases like "San Benito" vs "SanBenito")
        if row.empty:
            county_no_spaces = county.replace(" ", "")
            row = income_data[income_data["County"] == county_no_spaces]
        
        # If still not found, try with spaces added (for cases like "SanBenito" vs "San Benito")
        if row.empty:
            # Add spaces before capital letters (except the first one)
            import re
            county_with_spaces = re.sub(r'(?<!^)(?=[A-Z])', ' ', county)
            row = income_data[income_data["County"] == county_with_spaces]
        
        if row.empty:
            # Use dataset median instead of arbitrary hardcoded value
            dataset_median = income_data['Median_Income'].median()
            print(f"Warning: County '{county}' not found in median income data, using dataset median ${dataset_median:,.0f}")
            return dataset_median
        return row["Median_Income"].iloc[0]
    except Exception as e:
        print(f"Warning: Could not load median income data: {str(e)}, using fallback $60,000")
        return 60000

def _load_scaling_params():
    """Load per-DMA training averages saved by build_training_data.py."""
    if not os.path.exists(SCALING_PARAMS_FILE):
        logger.warning(f"Scaling params not found at {SCALING_PARAMS_FILE}. Run build_training_data.py first.")
        return {}
    with open(SCALING_PARAMS_FILE, "r") as f:
        return json.load(f)


def _read_prediction_csv(pred_file):
    """
    Read a prediction CSV (new format: quoted header "Time","CalFresh").
    Returns DataFrame with columns: date (datetime), value (float).
    """
    with open(pred_file, 'r') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('Day,') or s.startswith('"Time"') or s.startswith('Time,'):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.read_csv(pred_file, skiprows=header_idx)
    if len(df.columns) < 2:
        return pd.DataFrame(columns=["date", "value"])

    df.columns = ["date", "value"] + list(df.columns[2:])
    df = df[["date", "value"]]

    # Filter out non-date rows
    known_non_data = {
        'Category: All categories', 'Region:', 'Week', 'Day', 'Month', 'Year',
        'United States', 'State', 'City', 'Metro', 'Subregion', 'Search term',
        'Note:', 'Notes:', 'Interest over time', 'Time', 'Geo', 'isPartial',
        'date', 'value', 'Average', 'Total', 'N/A', 'nan', '', None
    }
    df = df[~df['date'].astype(str).str.strip().isin(known_non_data)]

    df['date']  = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['date', 'value'])
    return df


def get_latest_trends_for_metro(metro_area, county=None):
    """
    Load prediction trend data for a metro area, scale it to the training data's
    absolute 0-100 reference frame, and return feature values ready for the model.

    Scaling formula (per DMA, per keyword):
        scaled_value = latest_month_avg * (train_avg / pred_window_avg)

    This preserves the spike signal: if the current month is elevated relative
    to the prediction window average, the scaled value is proportionally above
    the training average that the model learned from.

    No population division — trends stay on the 0-100 Google Trends scale.
    Population is already a separate model feature.
    """
    if not os.path.exists(PREDICTION_BASE_DIR):
        logger.error(f"Prediction directory {PREDICTION_BASE_DIR} does not exist")
        return {}

    scaling_params = _load_scaling_params()
    keywords = ['CalFresh', 'FoodBank']
    trends = {}

    # Detect prediction month from Bakersfield file (used for logging only)
    prediction_month = None
    for kw in keywords:
        sample = os.path.join(PREDICTION_BASE_DIR, kw, "Bakersfield.csv")
        if os.path.exists(sample):
            df_sample = _read_prediction_csv(sample)
            if not df_sample.empty:
                last_date = df_sample['date'].max()
                prediction_month = last_date.replace(day=1)
                logger.info(f"Detected prediction month: {prediction_month.strftime('%B %Y')}")
                break

    if not prediction_month:
        prediction_month = datetime.now().replace(day=1)
        logger.warning("Could not detect prediction month, using current month")

    for keyword in keywords:
        pred_file = os.path.join(PREDICTION_BASE_DIR, keyword, f"{metro_area}.csv")
        if not os.path.exists(pred_file):
            logger.warning(f"No prediction file for {metro_area}/{keyword}")
            continue

        df_pred = _read_prediction_csv(pred_file)
        if df_pred.empty:
            logger.warning(f"Empty prediction data for {metro_area}/{keyword}")
            continue

        # Monthly aggregate (zeros kept — they are real low-activity signal)
        df_pred['ym'] = df_pred['date'].dt.to_period('M')
        monthly = df_pred.groupby('ym')['value'].mean().reset_index()
        monthly['date'] = monthly['ym'].dt.to_timestamp()

        if monthly.empty:
            continue

        # Latest month's value
        latest_month_avg = monthly['value'].iloc[-1]

        # Prediction window average (all months in file)
        pred_window_avg = monthly['value'].mean()

        # Training average for this DMA/keyword
        train_avg = scaling_params.get(keyword, {}).get(metro_area)
        if train_avg is None:
            logger.warning(f"No training avg for {metro_area}/{keyword} in scaling params")
            continue

        # Scale: align prediction window to training reference frame
        if pred_window_avg == 0:
            # All-zero prediction window — no search signal, use training avg directly
            scaled_value = train_avg
            logger.info(f"{metro_area}/{keyword}: all-zero prediction window, using train_avg={train_avg:.2f}")
        else:
            scaled_value = latest_month_avg * (train_avg / pred_window_avg)

        logger.info(
            f"{metro_area}/{keyword}: latest={latest_month_avg:.1f}, "
            f"pred_avg={pred_window_avg:.1f}, train_avg={train_avg:.1f}, "
            f"scaled={scaled_value:.2f}"
        )

        trends[f"monthly_average_{keyword}"] = scaled_value

    return trends

def _detect_target_month():
    """Detect the target prediction month from the prediction data files."""
    keywords = ['CalFresh', 'FoodBank']
    for keyword in keywords:
        sample_file = os.path.join(PREDICTION_BASE_DIR, keyword, "Bakersfield.csv")
        if os.path.exists(sample_file):
            try:
                last_date = None
                with open(sample_file, 'r') as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped or ',' not in stripped:
                            continue
                        date_part = stripped.split(',')[0].strip().strip('"')
                        if date_part and date_part[0].isdigit() and '-' in date_part:
                            try:
                                last_date = pd.to_datetime(date_part)
                            except (ValueError, pd.errors.ParserError):
                                continue
                if last_date is not None:
                    return last_date.replace(day=1)
            except Exception:
                continue
    # Fallback to current month
    return pd.Timestamp(datetime.now().replace(day=1))

def list_available_counties():
    """List all available counties that have population and metro mapping"""
    try:
        county_metro_map = pd.read_csv(COUNTY_METRO_MAP)
        pop_df = pd.read_csv(POP_DATA_FILE)
        counties = set(county_metro_map["county"]).intersection(set(pop_df["County"]))
        return sorted(counties)
    except Exception as e:
        logger.error(f"Error listing counties: {e}")
        return []

def zscore_to_flag(z):
    """Convert z-score to flag color."""
    if pd.isna(z):
        return 'Gray'
    if z < 0:
        return 'Green'
    if z >= 2:
        return 'Red'
    elif z >= 1:
        return 'Yellow'
    else:
        return 'Green'

def save_predictions_to_csv(predictions, output_file="src/data/finalPrediction.csv"):
    """Save predictions to a CSV file with risk flags."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create DataFrame — predictions can be (value, lower, upper) tuples or plain values
        rows = []
        for (county, date), pred in predictions.items():
            if isinstance(pred, tuple):
                val, lower, upper = pred
            else:
                val, lower, upper = pred, None, None
            rows.append({
                'date': date,
                'county': county,
                'predicted_applications': val,
                'lower_bound': lower,
                'upper_bound': upper,
            })
        df = pd.DataFrame(rows)
        
        # Add population data for z-score calculation
        try:
            pop_data = pd.read_csv("src/data/popData.csv")
            # Handle the column name case difference
            if 'County' in pop_data.columns:
                pop_data = pop_data.rename(columns={'County': 'county'})
            pop_data['county'] = pop_data['county'].astype(str)
            df['county'] = df['county'].astype(str)
            df = df.merge(pop_data[['county', 'Population']], on='county', how='left')
        except Exception as e:
            print(f"Warning: Could not load population data: {str(e)}")
            # Set a default population if we can't load the data
            df['Population'] = 100000  # Default population for z-score calculation
        
        # Load historical SNAP data to calculate county-specific statistics
        try:
            snap_data = pd.read_csv("src/data/SNAPApps/SNAPData.csv", header=None, 
                                   names=["county", "date_str", "SNAP_Applications"], thousands=",")
            snap_data["date"] = pd.to_datetime(snap_data["date_str"].str.strip(), format="%b %Y", errors="coerce")
            snap_data.loc[snap_data["date"].isna(), "date"] = pd.to_datetime(
                snap_data.loc[snap_data["date"].isna(), "date_str"].str.strip(), format="%B %Y", errors="coerce"
            )
            snap_data["SNAP_Applications"] = pd.to_numeric(snap_data["SNAP_Applications"].replace("*", pd.NA), errors="coerce")
            
            # Use the population data already loaded above
            snap_data['county'] = snap_data['county'].astype(str)
            
            # Merge with population data from the predictions dataframe
            snap_data = snap_data.merge(df[['county', 'Population']].drop_duplicates(), on='county', how='left')
            
            # Calculate historical SNAP application rates
            snap_data['SNAP_Application_Rate'] = snap_data['SNAP_Applications'] / snap_data['Population']
            
            # Calculate county-specific statistics from historical rate data
            county_stats = snap_data.groupby('county')["SNAP_Application_Rate"].agg(['mean', 'std']).reset_index()
            
            # Merge with predictions
            df = df.merge(county_stats, on='county', how='left')
            
            # Calculate predicted rates for z-score calculation
            df['predicted_rate'] = df['predicted_applications'] / df['Population']
            
            # Calculate z-scores using historical county-specific rate statistics
            df['z_score'] = (df['predicted_rate'] - df['mean']) / df['std']
            # Leave NaN z-scores as NaN — zscore_to_flag() will map them to 'Gray'

        except Exception as e:
            print(f"Warning: Could not load historical data for county-specific z-scores: {str(e)}")
            print("Falling back to prediction-based z-scores...")

            # Fallback: only runs when historical data fails to load
            df['predicted_rate'] = df['predicted_applications'] / df['Population']
            mean_apps = df['predicted_applications'].mean()
            std_apps = df['predicted_applications'].std()

            if std_apps == 0:
                df['z_score'] = np.nan
            else:
                df['z_score'] = (df['predicted_applications'] - mean_apps) / std_apps
        
        # Add flag based on z-score
        df['flag'] = df['z_score'].apply(zscore_to_flag)
        
        # Drop intermediate columns as they're not needed in the final output
        columns_to_drop = ['z_score']
        if 'mean' in df.columns:
            columns_to_drop.append('mean')
        if 'std' in df.columns:
            columns_to_drop.append('std')
        df = df.drop(columns=columns_to_drop)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Predictions with risk flags saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        return False

def generate_predictions(counties=None):
    """Generate predictions for all counties or specified counties."""
    if counties is None:
        counties = list_available_counties()
    
    predictions = {}
    
    # Detect the target month from prediction data
    target_month = None
    sample_file = os.path.join(PREDICTION_BASE_DIR, "CalFresh", "Bakersfield.csv")
    if os.path.exists(sample_file):
        try:
            with open(sample_file, 'r') as f:
                lines = f.readlines()
                last_date = None
                for line in lines:
                    stripped = line.strip()
                    if not stripped or ',' not in stripped:
                        continue
                    date_part = stripped.split(',')[0].strip().strip('"')
                    if date_part and date_part[0].isdigit() and '-' in date_part:
                        try:
                            last_date = pd.to_datetime(date_part)
                        except (ValueError, pd.errors.ParserError):
                            continue
                if last_date is not None:
                    prediction_month = last_date
                    target_month = prediction_month.replace(day=1)
                    print(f"Detected prediction data through: {prediction_month.strftime('%B %Y')}")
                    print(f"Predicting for: {target_month.strftime('%B %Y')}")
        except Exception as e:
            print(f"Could not detect prediction month: {e}")
    
    if not target_month:
        print("Could not detect target month, using current month")
        target_month = datetime.now().replace(day=1)
    
    # Generate prediction date (first day of target month)
    prediction_date_str = target_month.strftime("%Y-%m-01")
    
    print(f"Generating predictions for {prediction_date_str}...")
    
    for county in counties:
        try:
            metro = get_metro_for_county(county)
            if not metro:
                print(f"Skipping {county}: No metro area found")
                continue
            
            trends = get_latest_trends_for_metro(metro, county=county)
            if not trends:
                print(f"Skipping {county}: No trend data available")
                continue

            population = get_population_for_county(county)
            median_income = get_median_income_for_county(county)
            # Use prediction data trends to predict target month
            prediction, lower, upper = predict_next_month(trends, population, median_income, prediction_month=target_month.month)
            predictions[(county, prediction_date_str)] = (round(prediction, 2), round(lower, 2), round(upper, 2))
            print(f"{county}: {prediction:.2f} [{lower:.2f} - {upper:.2f}]")
            
        except Exception as e:
            print(f"Error predicting for {county}: {str(e)}")
    
    return predictions

def main():
    if len(sys.argv) > 1:
        # Predict for specific county if provided
        county = sys.argv[1]
        try:
            metro = get_metro_for_county(county)
            if not metro:
                print(f"No metro area found for county: {county}")
                sys.exit(1)

            # Get latest trends for the metro area, normalized by this county's population
            trends = get_latest_trends_for_metro(metro, county=county)
            if not trends:
                print(f"No trend data available for {metro}")
                sys.exit(1)

            population = get_population_for_county(county)
            median_income = get_median_income_for_county(county)

            # Detect target month from prediction data
            target_month = _detect_target_month()
            prediction, lower, upper = predict_next_month(trends, population, median_income, prediction_month=target_month.month)
            print(f"Predicted SNAP applications for {county} ({target_month.strftime('%B %Y')}): {prediction:.2f} [{lower:.2f} - {upper:.2f}]")

            # Save to CSV
            save_predictions_to_csv({
                (county, target_month.strftime("%Y-%m-01")): (round(prediction, 2), round(lower, 2), round(upper, 2))
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    else:
        # Generate predictions for all counties
        predictions = generate_predictions()
        if predictions:
            save_predictions_to_csv(predictions)

if __name__ == "__main__":
    main()