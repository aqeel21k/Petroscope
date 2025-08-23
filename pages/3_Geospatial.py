import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from utils import validate_and_read_data
def generate_geospatial_data(num_points, lon_range, lat_range):
    """
    Generates realistic synthetic geospatial data.
    
    Parameters:
    - num_points (int): Number of data points to generate.
    - lon_range (tuple): A tuple (min_lon, max_lon) for longitude.
    - lat_range (tuple): A tuple (min_lat, max_lat) for latitude.
    
    Returns:
    - A pandas DataFrame with generated geospatial data.
    """
    data = {
        'Well_Name': [f'Well_{i+1}' for i in range(num_points)],
        'POROSITY': np.random.uniform(0.15, 0.25, num_points),
        'PERMEABILITY': np.random.uniform(50, 500, num_points),
        'Well_Type': np.random.choice(['Oil', 'Gas', 'Water'], num_points),
        'Longitude': np.random.uniform(lon_range[0], lon_range[1], num_points),
        'Latitude': np.random.uniform(lat_range[0], lat_range[1], num_points),
    }
    return pd.DataFrame(data)

# --- Set page configuration ---
st.set_page_config(page_title="Geospatial Analysis", layout="wide")
st.title("Geospatial Analysis for Well Data")
st.markdown("---")

# --- Sidebar for user controls ---
st.sidebar.header("Options")
analysis_property = st.sidebar.selectbox(
    'Select a property to analyze:',
    ('POROSITY', 'PERMEABILITY')
)

# --- Main Geospatial Analysis ---
st.header("Geospatial Analysis")
st.markdown("Please upload a CSV file with well locations and properties.")
uploaded_file_geo = st.file_uploader("Choose a file for Geospatial Data", type=["csv"])

df_geo = None
if uploaded_file_geo is not None:
    df_geo = pd.read_csv(uploaded_file_geo)
    df_geo.columns = [c.strip().replace(' ', '_').replace(':', '_').upper() for c in df_geo.columns]
else:
    st.info("No file uploaded. Generating synthetic geospatial data for demonstration.")
    longitude_range = (40, 48)  # Example range for Iraq
    latitude_range = (29, 37)   # Example range for Iraq
    df_geo = generate_geospatial_data(num_points=50, lon_range=longitude_range, lat_range=latitude_range)
    
if df_geo is not None and 'LATITUDE' in df_geo.columns and 'LONGITUDE' in df_geo.columns:
    # --- Interactive Map ---
    st.subheader("Interactive Map")
    fig_map = px.scatter_mapbox(df_geo, 
                                lat="LATITUDE", 
                                lon="LONGITUDE", 
                                hover_name="WELL_NAME", 
                                color=analysis_property, # Color based on user selection
                                size=analysis_property, # Size based on user selection
                                color_continuous_scale=px.colors.sequential.Viridis, 
                                zoom=4, 
                                height=600,
                                title=f"Well Locations by {analysis_property.replace('_', ' ')}")
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

    # --- Well Type Distribution Bar Chart ---
    st.subheader("Well Type Distribution")
    fig_bar = px.bar(df_geo, x='WELL_TYPE', title='Count of Wells by Type')
    fig_bar.update_yaxes(title_text="Number of Wells")
    fig_bar.update_xaxes(title_text="Well Type")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Scatter Plot for Property Correlation ---
    st.subheader("Porosity vs Permeability Correlation")
    fig_scatter = px.scatter(df_geo, 
                             x='POROSITY', 
                             y='PERMEABILITY', 
                             color='WELL_TYPE', 
                             hover_name='WELL_NAME',
                             title="Porosity vs Permeability")
    fig_scatter.update_xaxes(title_text="Porosity")
    fig_scatter.update_yaxes(title_text="Permeability (mD)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Box Plot for Property Distribution ---
    st.subheader(f"Distribution of {analysis_property.replace('_', ' ')} by Well Type")
    fig_box = px.box(df_geo, x='WELL_TYPE', y=analysis_property, color='WELL_TYPE',
                     title=f"Box Plot of {analysis_property.replace('_', ' ')} by Well Type")
    fig_box.update_xaxes(title_text="Well Type")
    fig_box.update_yaxes(title_text=f"{analysis_property.replace('_', ' ')}")
    st.plotly_chart(fig_box, use_container_width=True)

else:
    st.warning("Please ensure your file contains the required columns: LATITUDE, LONGITUDE, WELL_NAME, WELL_TYPE, POROSITY, PERMEABILITY.")
