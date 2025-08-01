import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import glob, os
from datetime import datetime

# --- CONFIGURATION ---
TITLE = "California Food Assistance Dashboard"
SUBTITLE = "SNAP food assistance application collected by Google Trends."
GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson"

# --- LOAD SNAP DATA ---
@st.cache_data
def load_snap_data():
    df = pd.read_csv(
        "src/data/SNAPApps/SNAPData.csv",
        header=None,
        names=["county", "date_str", "SNAP_Applications"],
        thousands=","  # in case of thousands separator
    )
    df["date"] = pd.to_datetime(df["date_str"].str.strip(), format="%b %Y", errors="coerce")
    df.loc[df["date"].isna(), "date"] = pd.to_datetime(
        df.loc[df["date"].isna(), "date_str"].str.strip(), format="%B %Y", errors="coerce"
    )
    df["SNAP_Applications"] = pd.to_numeric(df["SNAP_Applications"].replace("*", pd.NA), errors="coerce")
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    return df

# --- LOAD POPULATION DATA ---
@st.cache_data
def load_pop_data():
    """Load population data with metro area information"""
    try:
        pop_df = pd.read_csv("src/data/popData.csv")
        return pop_df
    except Exception as e:
        print(f"Warning: Could not load population data: {e}")
        return None

# --- LOAD GEOJSON ---
@st.cache_data
def load_geojson():
    response = requests.get(GEOJSON_URL)
    response.raise_for_status()
    geojson = response.json()
    for feature in geojson['features']:
        name = feature['properties'].get('name', '')
        feature['properties']['name'] = name.replace(' County', '').strip()
    return geojson

# --- UTILITIES ---
def format_snap(val):
    try:
        if pd.isnull(val):
            return 'No data available'
        v = float(val)
        return f"{int(v):,}" if v.is_integer() else f"{v:,.0f}"
    except Exception:
        return str(val)

def zscore_to_flag(z):
    if pd.isnull(z): return 'Gray'
    if z < 0: return 'Green'
    if z >= 2: return 'Red'
    elif z >= 1: return 'Yellow'
    return 'Green'

def compute_z_scores(df):
    county_stats = df.groupby('county')["SNAP_Applications"].agg(['mean', 'std'])
    df = df.merge(county_stats, on='county', how='left')
    z_scores = (df["SNAP_Applications"] - df["mean"]) / df["std"]
    return z_scores.fillna(0)

# --- LAYOUT ---
st.set_page_config(page_title=TITLE, layout="wide")
tabs = st.tabs(["Current Map", "Predictions Map"])

# --- TAB 1: CURRENT MAP ---
with tabs[0]:
    st.title(TITLE)
    st.markdown(f"<h4 style='margin-top:-12px;color:gray'>{SUBTITLE}</h4>", unsafe_allow_html=True)

    snap_df = load_snap_data()
    pop_df = load_pop_data()
    counties_geojson = load_geojson()

    unique_dates = snap_df['date'].sort_values().unique()
    date_options = [d.strftime('%b %Y') for d in unique_dates]
    selected_date = st.selectbox("Select Month", options=date_options, index=len(date_options)-1)

    filtered_df = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].copy()
    filtered_df['SNAP_Applications_Display'] = filtered_df['SNAP_Applications'].apply(format_snap)
    
    # Merge with population data for hover information
    if pop_df is not None:
        # Clean up column names in case of whitespace
        pop_df.columns = pop_df.columns.str.strip()
        # Clean county names for matching
        filtered_df['county_clean'] = filtered_df['county'].str.replace(' County', '', regex=False)
        pop_df['county_clean'] = pop_df['County'].str.replace(' County', '', regex=False)
        # Check for fips column
        if 'fips' not in pop_df.columns:
            print('DEBUG: popData columns:', pop_df.columns.tolist())
            raise KeyError("'fips' column not found in popData.csv. Columns are: " + str(pop_df.columns.tolist()))
        filtered_df = filtered_df.merge(
            pop_df[['county_clean', 'metro_area', 'Population', 'Population Density', 'fips']], 
            on='county_clean', 
            how='left'
        )
        
        # Create hover text with the specified order, including FIPS before SNAP Applications
        filtered_df['hover_text'] = filtered_df.apply(
            lambda row: f"<b>{row['county']}</b><br>Metro: {row.get('metro_area', 'N/A')}<br>Population: {row.get('Population', 'N/A'):,}<br>Population Density: {row.get('Population Density', 'N/A'):,}<br>FIPS: {row.get('fips', 'N/A')}<br>SNAP Applications: {row['SNAP_Applications_Display']}", 
            axis=1
        )
    else:
        # Fallback if population data is not available
        filtered_df['hover_text'] = filtered_df.apply(
            lambda row: f"<b>{row['county']}</b><br>SNAP Applications: {row['SNAP_Applications_Display']}", 
            axis=1
        )

    snap_df['z_score'] = compute_z_scores(snap_df)
    zscore_map = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].set_index('county')['z_score'].to_dict()
    filtered_df['z_score'] = filtered_df['county'].map(zscore_map)
    filtered_df['Flag'] = filtered_df['z_score'].apply(zscore_to_flag)

    flag_color_map = {
        "Red": "#e74c3c",
        "Yellow": "#f7ca18",
        "Green": "#27ae60",
        "Gray": "#888888",
        None: "#888888"
    }
    filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))

    fig = px.choropleth(
        filtered_df,
        geojson=counties_geojson,
        locations='county',
        color='Flag',
        color_discrete_map=flag_color_map,
        custom_data=['hover_text'],
        featureidkey="properties.name",
        scope="usa",
        height=700
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PREDICTIONS MAP ---
# Define flag color map at the top level
flag_color_map = {
    "Red": "#e74c3c",
    "Yellow": "#f7ca18",
    "Green": "#27ae60",
    "Gray": "#888888",
    None: "#888888"
}

with tabs[1]:
    st.title("Predictions")
    snap_df = load_snap_data()
    pop_df = load_pop_data()
    counties_geojson = load_geojson()
    snap_df['county'] = snap_df['county'].astype(str)
    unique_dates = snap_df['date'].sort_values().unique()
    date_options = [d.strftime('%b %Y') for d in unique_dates]

    pred_path = os.path.join('src', 'data', 'finalPrediction.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        pred_df['county'] = pred_df['county'].astype(str)
        pred_df['county_normalized'] = pred_df['county'].str.strip().str.lower()

        # Set predicted month to current month and year
        current_date = datetime.now()
        pred_month_label = current_date.strftime("%Y-%m-01")  # Match the CSV format
        pred_dt = pd.to_datetime(pred_month_label, format='%Y-%m-%d')
        pred_month_english = pred_dt.strftime('%b %Y')
        pred_option = f"Predicted SNAP Applications for {pred_month_english}"
        date_options_with_pred = date_options + [pred_option]
    else:
        pred_df = None
        date_options_with_pred = date_options

    selected_date = st.selectbox("Select Month", options=date_options_with_pred, index=len(date_options_with_pred)-1, key="pred_month")
    use_predicted = pred_df is not None and selected_date == f"Predicted SNAP Applications for {pred_month_english}"

    if use_predicted:
        pred_rows = pred_df[pred_df['date'] == pred_month_label].copy()
        pred_map = pred_rows.set_index('county_normalized')['predicted_applications'].to_dict()
        
        # Use the same z-score calculation as the current map for consistency
        base_df = snap_df[snap_df['date'] == unique_dates[-1]].copy()
        base_df['county_normalized'] = base_df['county'].str.strip().str.lower()
        base_df['prediction'] = base_df['county_normalized'].map(pred_map)
        
        # Calculate z-scores using the same method as current map
        snap_df['z_score'] = compute_z_scores(snap_df)
        # Create a mapping of predictions to z-scores using the same historical baseline
        pred_z_scores = {}
        for county in pred_map.keys():
            county_original = base_df[base_df['county_normalized'] == county]['county'].iloc[0] if not base_df[base_df['county_normalized'] == county].empty else None
            if county_original:
                # Get historical stats for this county
                county_data = snap_df[snap_df['county'] == county_original]
                if not county_data.empty:
                    county_mean = county_data['SNAP_Applications'].mean()
                    county_std = county_data['SNAP_Applications'].std()
                    if county_std > 0:
                        pred_value = pred_map[county]
                        z_score = (pred_value - county_mean) / county_std
                        pred_z_scores[county] = z_score
        
        # Apply z-score to flag conversion using the same function
        base_df['pred_z_score'] = base_df['county_normalized'].map(pred_z_scores)
        base_df['Flag'] = base_df['pred_z_score'].apply(zscore_to_flag)
        base_df['color'] = base_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))

        # Merge with popData for hover
        if pop_df is not None:
            pop_df.columns = pop_df.columns.str.strip()
            base_df['county_clean'] = base_df['county'].str.replace(' County', '', regex=False)
            pop_df['county_clean'] = pop_df['County'].str.replace(' County', '', regex=False)
            if 'fips' not in pop_df.columns:
                print('DEBUG: popData columns:', pop_df.columns.tolist())
                raise KeyError("'fips' column not found in popData.csv. Columns are: " + str(pop_df.columns.tolist()))
            base_df = base_df.merge(
                pop_df[['county_clean', 'metro_area', 'Population', 'Population Density', 'fips']],
                on='county_clean',
                how='left'
            )
            base_df['hover_text'] = base_df.apply(
                lambda row: f"<b>{row['county']}</b><br>Metro: {row.get('metro_area', 'N/A')}<br>Population: {row.get('Population', 'N/A'):,}<br>Population Density: {row.get('Population Density', 'N/A'):,}<br>FIPS: {row.get('fips', 'N/A')}<br>Prediction for {pred_month_english}: {format_snap(row['prediction'])}",
                axis=1
            )
        else:
            base_df['hover_text'] = base_df.apply(
                lambda row: f"<b>{row['county']}</b><br>Prediction for {pred_month_english}: {format_snap(row['prediction'])}",
                axis=1
            )
        filtered_df = base_df.copy()
    else:
        selected_date_dt = pd.to_datetime(selected_date)
        filtered_df = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].copy()
        
        # FIX: Use the same z-score calculation as the current tab (full dataset)
        snap_df['z_score'] = compute_z_scores(snap_df)
        zscore_map = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].set_index('county')['z_score'].to_dict()
        filtered_df['z_score'] = filtered_df['county'].map(zscore_map)
        filtered_df['Flag'] = filtered_df['z_score'].apply(zscore_to_flag)
        filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))
        filtered_df['actual'] = filtered_df['SNAP_Applications'].apply(format_snap)
        
        # Get predictions for July 2025 if available
        if pred_df is not None:
            july_2025_pred = pred_df[pred_df['date'] == '2025-07-01'].copy()
            if not july_2025_pred.empty:
                july_2025_pred['county_normalized'] = july_2025_pred['county'].str.strip().str.lower()
                pred_map_july = july_2025_pred.set_index('county_normalized')['predicted_applications'].to_dict()
                filtered_df['county_normalized'] = filtered_df['county'].str.strip().str.lower()
                filtered_df['july_prediction'] = filtered_df['county_normalized'].map(pred_map_july)
                # Merge with popData for hover
                if pop_df is not None:
                    pop_df.columns = pop_df.columns.str.strip()
                    filtered_df['county_clean'] = filtered_df['county'].str.replace(' County', '', regex=False)
                    pop_df['county_clean'] = pop_df['County'].str.replace(' County', '', regex=False)
                    if 'fips' not in pop_df.columns:
                        print('DEBUG: popData columns:', pop_df.columns.tolist())
                        raise KeyError("'fips' column not found in popData.csv. Columns are: " + str(pop_df.columns.tolist()))
                    filtered_df = filtered_df.merge(
                        pop_df[['county_clean', 'metro_area', 'Population', 'Population Density', 'fips']],
                        on='county_clean',
                        how='left'
                    )
                    filtered_df['hover_text'] = filtered_df.apply(
                        lambda row: f"<b>{row['county']}</b><br>Metro: {row.get('metro_area', 'N/A')}<br>Population: {row.get('Population', 'N/A'):,}<br>Population Density: {row.get('Population Density', 'N/A'):,}<br>FIPS: {row.get('fips', 'N/A')}<br>Actual: {row['actual']}<br>Prediction for Jul 2025: {format_snap(row['july_prediction'])}",
                        axis=1
                    )
                else:
                    filtered_df['hover_text'] = filtered_df.apply(
                        lambda row: f"Actual: {row['actual']}<br>Prediction for Jul 2025: {format_snap(row['july_prediction'])}",
                        axis=1
                    )
            else:
                filtered_df['hover_text'] = filtered_df.apply(
                    lambda row: f"Actual: {row['actual']}<br>No prediction available", axis=1)
        else:
            filtered_df['hover_text'] = filtered_df.apply(
            lambda row: f"Actual: {row['actual']}<br>No prediction available", axis=1)

    # Debug: Print the columns in filtered_df
    print("Columns in filtered_df:", filtered_df.columns.tolist())
    print("Sample Flag values:", filtered_df['Flag'].head().tolist())
    print("Sample color values:", filtered_df['color'].head().tolist())
    
    fig = px.choropleth(
        filtered_df,
        geojson=counties_geojson,
        locations='county',
        color='Flag',
        color_discrete_map=flag_color_map,
        category_orders={"Flag": ["Red", "Yellow", "Green", "Gray"]},
        custom_data=['hover_text'],
        featureidkey="properties.name",
        scope="usa",
        height=700
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
