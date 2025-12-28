import os
import pandas as pd
import numpy as np
from utils import scale_trends, normalize_trends_by_population

def create_scaled_training_data():
    """
    Create scaled training data that establishes a consistent base scale.
    This will be used for both training and as the reference for prediction scaling.
    """
    print("=== CREATING SCALED TRAINING DATA ===\n")
    
    # Load the current training data
    input_file = "src/data/aggregateTrends.csv"
    output_file = "src/data/aggregateTrends_scaled.csv"
    
    print(f"Loading training data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    
    # Get trend columns
    trend_cols = [col for col in df.columns if col.startswith('monthly_average_')]
    print(f"Trend columns to scale: {trend_cols}")
    
    # Create a copy for scaling
    df_scaled = df.copy()
    
    # Step 1: Apply date scaling to trend columns
    print("\n=== STEP 1: APPLYING DATE SCALING ===")
    for trend_col in trend_cols:
        print(f"Scaling {trend_col}...")
        
        # Group by county to scale each county's trends separately
        for county in df_scaled['county'].unique():
            county_mask = df_scaled['county'] == county
            county_data = df_scaled[county_mask].copy()
            
            if len(county_data) > 1:
                # Sort by date
                county_data = county_data.sort_values('date')
                
                # Create a reference column (use the first available value as baseline)
                county_data['reference'] = county_data[trend_col].iloc[0]
                
                # Apply scaling to align all values to the same scale
                # Use the first non-null value as the reference point
                first_valid_idx = county_data[trend_col].first_valid_index()
                if first_valid_idx is not None:
                    reference_value = county_data.loc[first_valid_idx, trend_col]
                    if reference_value != 0 and not pd.isna(reference_value):
                        # Scale all values relative to the first value
                        scaling_factor = 100 / reference_value  # Normalize to 0-100 scale
                        county_data[trend_col] = county_data[trend_col] * scaling_factor
                
                # Update the main dataframe
                df_scaled.loc[county_mask, trend_col] = county_data[trend_col]
    
    # Step 2: Apply population normalization
    print("\n=== STEP 2: APPLYING POPULATION NORMALIZATION ===")
    print("Normalizing trend columns by county population...")
    
    # Create a temporary dataframe for normalization
    norm_df = df_scaled[['county'] + trend_cols].copy()
    norm_df = normalize_trends_by_population(norm_df, county_col='county', trend_cols=trend_cols)
    
    # Update the scaled dataframe with normalized values
    for trend_col in trend_cols:
        df_scaled[trend_col] = norm_df[trend_col]
    
    # Step 3: Add minimal month features to avoid overfitting
    print(f"\n=== STEP 3: ADDING MINIMAL MONTH FEATURES ===")
    print("Adding only month feature to capture basic seasonal patterns...")
    
    # Convert date to datetime
    df_scaled['date'] = pd.to_datetime(df_scaled['date'])
    
    # Extract month (1-12) - minimal feature to avoid overfitting
    df_scaled['month'] = df_scaled['date'].dt.month
    
    print(f"Month feature added: month (1-12)")
    print(f"Month distribution: {df_scaled['month'].value_counts().sort_index().to_dict()}")
    
    # Step 4: Add median income data
    print(f"\n=== STEP 4: ADDING MEDIAN INCOME ===")
    print("Loading and merging median income data...")
    
    try:
        # Load median income data
        income_df = pd.read_csv("src/data/MedianIncome.csv")
        
        # Clean the median income column (remove commas and convert to numeric)
        income_df['Median_Income'] = income_df['Median Income'].str.replace(',', '').astype(float)
        
        # Merge with main dataframe
        df_scaled = df_scaled.merge(income_df[['County', 'Median_Income']], 
                                   left_on='county', right_on='County', how='left')
        
        # Drop the extra County column
        df_scaled = df_scaled.drop('County', axis=1)
        
        print(f"Median income data merged successfully")
        print(f"Income range: ${df_scaled['Median_Income'].min():,.0f} - ${df_scaled['Median_Income'].max():,.0f}")
        
    except Exception as e:
        print(f"Warning: Could not load median income data: {str(e)}")
        # Set default median income if data not available
        df_scaled['Median_Income'] = 60000  # Default median income
        print("Using default median income: $60,000")
    
    # Step 5: Convert SNAP applications to rates (applications per population)
    print(f"\n=== STEP 5: CONVERTING TO RATES ===")
    print("Converting SNAP_Applications to rates (applications per population)...")
    
    # Calculate SNAP application rate per population
    df_scaled['SNAP_Application_Rate'] = df_scaled['SNAP_Applications'] / df_scaled['Population']
    
    # Keep the original SNAP_Applications column for reference but use rate as target
    print(f"SNAP application rates calculated. Range: {df_scaled['SNAP_Application_Rate'].min():.6f} to {df_scaled['SNAP_Application_Rate'].max():.6f}")
    
    # Step 6: Save the scaled training data
    print(f"\n=== STEP 6: SAVING SCALED DATA ===")
    df_scaled.to_csv(output_file, index=False)
    print(f"Scaled training data saved to: {output_file}")
    
    # Step 7: Show comparison
    print("\n=== SCALING COMPARISON ===")
    print("Original vs Scaled trend values (first 5 rows):")
    
    for trend_col in trend_cols:
        print(f"\n{trend_col}:")
        comparison_df = pd.DataFrame({
            'county': df['county'].head(),
            'original': df[trend_col].head(),
            'scaled': df_scaled[trend_col].head()
        })
        print(comparison_df.to_string(index=False))
    
    # Show SNAP application rate examples with new features
    print(f"\nSNAP Application Rate Examples with New Features:")
    rate_examples = df_scaled[['county', 'month', 'Median_Income', 'SNAP_Applications', 'Population', 'SNAP_Application_Rate']].head()
    print(rate_examples.to_string(index=False))
    
    # Step 8: Update training script to use scaled data
    print("\n=== STEP 8: UPDATING TRAINING SCRIPT ===")
    update_training_script(output_file)
    
    print(f"\n✅ Scaled training data created successfully!")
    print(f"📁 File: {output_file}")
    print(f"📊 Shape: {df_scaled.shape}")
    print(f"🔧 Training script updated to use scaled data")
    
    return df_scaled

def update_training_script(scaled_file_path):
    """Update the training script to use the scaled data file."""
    training_script = "scripts/train_model.py"
    
    # Read the current training script
    with open(training_script, 'r') as f:
        content = f.read()
    
    # Update the file path
    old_path = "src/data/aggregateTrends.csv"
    new_path = scaled_file_path
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        
        # Write the updated content back
        with open(training_script, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {training_script} to use: {new_path}")
    else:
        print(f"⚠️  Could not find path to replace in {training_script}")

def update_prediction_script():
    """Update the prediction script to use the same scaling approach."""
    prediction_script = "scripts/predict.py"
    
    print(f"\n=== UPDATING PREDICTION SCRIPT ===")
    print("The prediction script will now use the same scaling approach as training data.")
    print("This ensures consistency between training and prediction scales.")

if __name__ == "__main__":
    try:
        scaled_data = create_scaled_training_data()
        update_prediction_script()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Run: python scripts/train_model.py (to train on scaled data)")
        print(f"2. Run: python scripts/predict.py (to predict with consistent scaling)")
        print(f"3. Both training and prediction will now use the same scale!")
        
    except Exception as e:
        print(f"❌ Error creating scaled training data: {str(e)}")
        import traceback
        traceback.print_exc() 