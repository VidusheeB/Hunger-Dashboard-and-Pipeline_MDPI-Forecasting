import streamlit as st
import pandas as pd
from pandas import DateOffset
import plotly.express as px
import requests
import os
import json

TITLE    = "California Food Assistance Dashboard"
SUBTITLE = "CalFresh (SNAP) application rates by county, predicted using Google Trends and BLS unemployment data."
GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson"

ALERT_LABELS_CSV  = os.path.join("outputs", "metrics", "threshold_alert_labels.csv")
ALERT_SUMMARY_JSON = os.path.join("outputs", "metrics", "threshold_alert_summary.json")

FLAG_COLOR_MAP = {
    "High Risk":    "#e74c3c",
    "Medium Risk":  "#f7ca18",
    "Low Risk":     "#27ae60",
    "No Data":      "#888888",
}

DISPLAY_LABEL = {
    "Red":    "High Risk",
    "Yellow": "Medium Risk",
    "Green":  "Low Risk",
    "Gray":   "No Data",
    None:     "No Data",
}

ALL_FLAGS = ["High Risk", "Medium Risk", "Low Risk", "No Data"]


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_snap_data():
    df = pd.read_csv(
        "src/data/SNAPApps/SNAPData.csv",
        header=None,
        names=["county", "date_str", "SNAP_Applications"],
        thousands=","
    )
    df["date"] = pd.to_datetime(df["date_str"].str.strip(), format="%b %Y", errors="coerce")
    df.loc[df["date"].isna(), "date"] = pd.to_datetime(
        df.loc[df["date"].isna(), "date_str"].str.strip(), format="%B %Y", errors="coerce"
    )
    df["SNAP_Applications"] = pd.to_numeric(
        df["SNAP_Applications"].replace("*", pd.NA), errors="coerce"
    )
    return df


@st.cache_data
def load_pop_data():
    try:
        pop_df = pd.read_csv("src/data/popData.csv")
        pop_df.columns = pop_df.columns.str.strip()
        pop_df["county_clean"] = pop_df["County"].str.replace(" County", "", regex=False)
        return pop_df
    except Exception as e:
        st.warning(f"Could not load population data: {e}")
        return None


@st.cache_data
def load_geojson():
    response = requests.get(GEOJSON_URL)
    response.raise_for_status()
    geojson = response.json()
    for feature in geojson["features"]:
        name = feature["properties"].get("name", "")
        feature["properties"]["name"] = name.replace(" County", "").strip()
    return geojson


@st.cache_data
def load_alert_labels():
    """Returns dict: (county, 'YYYY-MM-DD') -> label (Green/Yellow/Red)."""
    if not os.path.exists(ALERT_LABELS_CSV):
        return {}
    df = pd.read_csv(ALERT_LABELS_CSV, parse_dates=["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    return {(row["county"], row["date_str"]): row["label"]
            for _, row in df.iterrows()}


@st.cache_data
def load_county_thresholds():
    """Returns (red_thresholds, yellow_thresholds) dicts: county -> float."""
    if not os.path.exists(ALERT_SUMMARY_JSON):
        return {}, {}
    with open(ALERT_SUMMARY_JSON) as f:
        data = json.load(f)
    return data.get("county_red_thresholds", {}), data.get("county_yellow_thresholds", {})


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_snap(val):
    try:
        if pd.isnull(val):
            return "No data"
        v = float(val)
        return f"{int(v):,}"
    except Exception:
        return str(val)


def label_from_deviation(deviation, county, red_thresholds, yellow_thresholds):
    """Apply county-specific deviation thresholds → Green/Yellow/Red/Gray."""
    if pd.isnull(deviation):
        return "Gray"
    red_thr    = red_thresholds.get(county, float("inf"))
    yellow_thr = yellow_thresholds.get(county, float("inf"))
    if deviation > red_thr:
        return "Red"
    elif deviation > yellow_thr:
        return "Yellow"
    else:
        return "Green"


def get_historical_label(county, date, alert_labels):
    """Look up label for a historical county-month. Gray if not yet in model."""
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    return alert_labels.get((county, date_str), "Gray")


def add_pop_columns(df, pop_df):
    """Merge population info into df on county_clean."""
    if pop_df is None:
        return df
    df["county_clean"] = df["county"].str.replace(" County", "", regex=False)
    return df.merge(
        pop_df[["county_clean", "metro_area", "Population", "Population Density", "fips"]],
        on="county_clean", how="left"
    )


def build_hover(row, snap_label, snap_val, pred_val=None, pred_month=None):
    pop   = f"{int(row['Population']):,}"   if "Population" in row and pd.notna(row.get("Population")) else "N/A"
    dens  = f"{int(row['Population Density']):,}" if "Population Density" in row and pd.notna(row.get("Population Density")) else "N/A"
    metro = row.get("metro_area", "N/A")
    fips  = row.get("fips", "N/A")
    text  = (f"<b>{row['county']}</b><br>"
             f"Metro: {metro}<br>"
             f"Population: {pop}<br>"
             f"Population Density: {dens}<br>"
             f"FIPS: {fips}<br>"
             f"{snap_label}: {snap_val}")
    if pred_val is not None and pred_month is not None:
        text += f"<br>Predicted for {pred_month}: {pred_val}"
    return text


def draw_map(filtered_df, title_note=""):
    counties_geojson = load_geojson()
    plot_df = filtered_df.copy()

    # Remap internal flag names to display labels
    plot_df["Flag"] = plot_df["Flag"].map(lambda f: DISPLAY_LABEL.get(f, "No Data"))

    # Add one dummy row per missing category so all legend entries always appear
    present = set(plot_df["Flag"])
    dummy_rows = []
    for flag in ALL_FLAGS:
        if flag not in present:
            dummy_rows.append({"county": "", "Flag": flag, "hover_text": ""})
    if dummy_rows:
        plot_df = pd.concat([plot_df, pd.DataFrame(dummy_rows)], ignore_index=True)

    fig = px.choropleth(
        plot_df,
        geojson=counties_geojson,
        locations="county",
        color="Flag",
        color_discrete_map=FLAG_COLOR_MAP,
        category_orders={"Flag": ALL_FLAGS},
        custom_data=["hover_text"],
        featureidkey="properties.name",
        scope="usa",
        height=680,
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(
        margin={"r": 0, "t": 70, "l": 0, "b": 0},
        showlegend=True,
        legend=dict(
            title=None,
            orientation="h",
            x=0.0, y=1.08,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
        ),
        title=dict(text=title_note, x=0.5, y=1.0, yanchor="top", font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title=TITLE, layout="wide")

# Load shared resources once
snap_df        = load_snap_data()
pop_df         = load_pop_data()
alert_labels   = load_alert_labels()
red_thresholds, yellow_thresholds = load_county_thresholds()

tabs = st.tabs(["Historical Map", "Predictions Map"])


# ── Tab 1: Historical Map ─────────────────────────────────────────────────────

with tabs[0]:
    st.title(TITLE)
    st.markdown(f"<h4 style='margin-top:-12px;color:gray'>{SUBTITLE}</h4>",
                unsafe_allow_html=True)

    unique_dates  = snap_df["date"].sort_values().unique()
    date_options  = [pd.Timestamp(d).strftime("%b %Y") for d in unique_dates]
    selected_date = st.selectbox("Select Month", options=date_options,
                                 index=len(date_options) - 1)

    sel_dt       = pd.to_datetime(selected_date, format="%b %Y")
    filtered_df  = snap_df[snap_df["date"].dt.strftime("%b %Y") == selected_date].copy()
    filtered_df  = add_pop_columns(filtered_df, pop_df)

    # Flag from alert labels (Gray if pre-model or missing)
    filtered_df["Flag"] = filtered_df.apply(
        lambda row: get_historical_label(row["county"], row["date"], alert_labels), axis=1
    )

    filtered_df["hover_text"] = filtered_df.apply(
        lambda row: build_hover(row,
                                snap_label="SNAP Applications",
                                snap_val=format_snap(row["SNAP_Applications"])),
        axis=1,
    )

    note = ("High/Medium/Low Risk: county flagged when actual SNAP applications exceeded model prediction "
            "by more than a county-specific threshold (60th / 50th percentile of historical deviations). "
            "No Data = month predates the model window (pre-Apr 2018) or missing source data (e.g. BLS gap).")
    draw_map(filtered_df, title_note=selected_date)
    st.caption(note)


# ── Tab 2: Predictions Map ────────────────────────────────────────────────────

with tabs[1]:
    st.title("Predictions")

    pred_path = os.path.join("outputs", "predictions", "finalPrediction.csv")
    legacy_pred_path = os.path.join("src", "data", "finalPrediction.csv")
    if not os.path.exists(pred_path) and os.path.exists(legacy_pred_path):
        pred_path = legacy_pred_path
    pred_df   = None
    pred_month_english = None

    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        pred_df["county"] = pred_df["county"].astype(str)
        pred_df["county_clean"] = pred_df["county"].str.replace(" County", "", regex=False)

        if not pred_df.empty:
            # Date in CSV is the target SNAP month.
            pred_date          = pred_df["date"].iloc[0]
            pred_dt            = pd.to_datetime(pred_date)
            target_month       = pred_dt
            pred_month_english = target_month.strftime("%b %Y")

    # ── Prominent prediction target banner ────────────────────────────────────
    if pred_month_english:
        st.info(
            f"**Predicting: {pred_month_english}** — Select "
            f"\"Predicted — {pred_month_english}\" at the bottom of the dropdown to view the forecast."
        )
    else:
        st.warning("No prediction file found. Run the pipeline to generate forecasts.")

    unique_dates = snap_df["date"].sort_values().unique()
    date_options = [pd.Timestamp(d).strftime("%b %Y") for d in unique_dates]

    if pred_month_english:
        pred_option           = f"Predicted — {pred_month_english}"
        date_options_extended = date_options + [pred_option]
    else:
        date_options_extended = date_options

    selected_date = st.selectbox("Select Month", options=date_options_extended,
                                 index=len(date_options_extended) - 1, key="pred_month")

    use_predicted = (pred_df is not None and pred_month_english is not None
                     and selected_date == f"Predicted — {pred_month_english}")

    if use_predicted:
        # ── Predicted month: score against each county's 12-month baseline ───
        base_df = pred_df.copy()
        base_df = add_pop_columns(base_df, pop_df)

        # Build county baseline rates from last 12 months of historical SNAP data
        county_baselines = {}
        if pop_df is not None:
            snap_r = snap_df.copy()
            snap_r["county_clean"] = snap_r["county"].str.replace(" County", "", regex=False)
            snap_r = snap_r.merge(pop_df[["county_clean", "Population"]], on="county_clean", how="left")
            snap_r = snap_r.dropna(subset=["Population", "SNAP_Applications"])
            snap_r["rate"] = snap_r["SNAP_Applications"] / snap_r["Population"]
            county_baselines = (
                snap_r.sort_values("date")
                      .groupby("county")
                      .tail(12)
                      .groupby("county")["rate"]
                      .mean()
                      .to_dict()
            )

        def score_predicted(row):
            cty      = row.get("county_clean", row.get("county", ""))
            baseline = county_baselines.get(cty)
            if baseline is None or baseline <= 0:
                return "Gray"
            pr = row.get("predicted_rate")
            if pr is None or pd.isna(pr):
                return "Gray"
            deviation = (pr - baseline) / baseline
            return label_from_deviation(deviation, cty, red_thresholds, yellow_thresholds)

        base_df["Flag"] = base_df.apply(score_predicted, axis=1)

        base_df["hover_text"] = base_df.apply(
            lambda row: build_hover(
                row,
                snap_label=f"Predicted applications ({pred_month_english})",
                snap_val=format_snap(row.get("predicted_applications")),
            ),
            axis=1,
        )
        note = (f"Predicted alert level for {pred_month_english}: "
                f"High/Medium/Low Risk = predicted rate vs each county's 12-month historical baseline, "
                f"using the same county-specific deviation thresholds as the Historical Map. "
                f"No Data = insufficient historical data for that county.")
        draw_map(base_df, title_note=f"Predicted — {pred_month_english}")
        st.caption(note)

    else:
        # ── Historical month: same alert labels as Tab 1, plus prediction in hover ──
        filtered_df = snap_df[snap_df["date"].dt.strftime("%b %Y") == selected_date].copy()
        filtered_df = add_pop_columns(filtered_df, pop_df)

        filtered_df["Flag"] = filtered_df.apply(
            lambda row: get_historical_label(row["county"], row["date"], alert_labels), axis=1
        )

        # Add predicted applications in hover if prediction file exists
        if pred_df is not None and pred_month_english:
            filtered_df["county_clean"] = filtered_df["county"].str.replace(" County", "", regex=False)
            pred_map = pred_df.set_index("county_clean")["predicted_applications"].to_dict()
            filtered_df["hover_text"] = filtered_df.apply(
                lambda row: build_hover(
                    row,
                    snap_label="SNAP Applications",
                    snap_val=format_snap(row["SNAP_Applications"]),
                    pred_val=format_snap(pred_map.get(
                        row["county"].replace(" County", "").strip()
                    )),
                    pred_month=pred_month_english,
                ),
                axis=1,
            )
        else:
            filtered_df["hover_text"] = filtered_df.apply(
                lambda row: build_hover(row,
                                        snap_label="SNAP Applications",
                                        snap_val=format_snap(row["SNAP_Applications"])),
                axis=1,
            )

        note = ("High/Medium/Low Risk: county flagged when actual SNAP applications exceeded model prediction "
                "by more than a county-specific threshold (60th / 50th percentile of historical deviations). "
                "No Data = month predates the model window (pre-Apr 2018) or missing source data.")
        draw_map(filtered_df, title_note=selected_date)
        st.caption(note)
