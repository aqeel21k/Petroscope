# Updated Streamlit Petrophysical Analysis App
# This application is designed to read .las and .csv files,
# perform petrophysical calculations, and visualize the results.

# 1. Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lasio  # Import lasio for robust .las file reading
from utils import validate_and_read_data

# 2. Set Streamlit page configuration
st.set_page_config(page_title="Advanced Petrophysical Analysis", layout="wide", page_icon="📈")

# --- Define a dictionary for flexible column names ---
# This allows the app to recognize common variations of log names.
LOG_NAME_MAPPING = {
    'DEPTH': ['DEPTH', 'DEPT', 'DEPTH_M', 'DEPTH_FT', 'MD', 'TVD'],
    'GR': ['GR', 'GAMMA_RAY', 'GAMMARAY', 'GR_API', 'GR_LOG'],
    'RES': ['RES', 'RT', 'RESISTIVITY', 'LLD', 'LLD_OHMM'],
    'RHOB': ['RHOB', 'RHO', 'DENS', 'DEN', 'BULK_DENSITY'],
    'NPHI': ['NPHI', 'NPH', 'NEUT', 'NEUTRON_POROSITY'],
    'DT': ['DT', 'DELT', 'SONIC', 'SONIC_MICROSEC'],
    'CALI': ['CALI', 'CAL'],
    'SP': ['SP', 'SPONTANEOUS_POTENTIAL'],
    'PEF': ['PEF', 'PE'],
    'PHIE': ['PHIE', 'EFFECTIVE_POROSITY', 'PHIE_VSHALE_CORR'],
    'SW': ['SW', 'WATER_SATURATION', 'SW_ARCHIE'],
    'SH': ['SH', 'HYDROCARBON_SATURATION']
}

# --- Function to standardize column names ---
def standardize_columns(df, mapping):
    """
    Standardizes column names in a DataFrame based on a mapping dictionary.
    This function is now a helper for the new manual selection method.
    """
    reverse_mapping = {val.upper().strip().replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', ''): key for key, values in mapping.items() for val in values}
    
    # Standardize the DataFrame columns first
    df.columns = [col.upper().strip().replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', '') for col in df.columns]

    standardized_names = {}
    for col in df.columns:
        if col in reverse_mapping:
            standardized_names[col] = reverse_mapping[col]
            
    df.rename(columns=standardized_names, inplace=True)
    return df

# --- Function to read LAS files ---
@st.cache_data
def read_las_file(file):
    """
    Reads a .las file using the more reliable lasio library and returns a DataFrame.
    """
    try:
        # Use lasio.read with the file's content directly
        las_file = lasio.read(file.getvalue().decode('utf-8', errors='ignore'))
        df = las_file.df()

        # Check for a depth column
        depth_col_options = [col.upper().strip().replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', '') for col in LOG_NAME_MAPPING['DEPTH']]
        found_depth_col = next((col for col in df.columns if col in depth_col_options), None)
        
        # If no explicit depth column is found, use the DataFrame index
        if found_depth_col is None:
            st.warning("⚠️ Warning: No explicit depth column found (e.g., DEPTH, DEPT, MD). Using the data index as the depth column.")
            df['DEPTH'] = df.index
            found_depth_col = 'DEPTH'

        # Make sure the depth column is the index
        if 'DEPTH' in df.columns:
            df.set_index('DEPTH', inplace=True)
        else:
            df.set_index(found_depth_col, inplace=True)
            df.index.name = 'DEPTH'

        # Convert numerical columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.success("✅ LAS file uploaded successfully!")
        return df

    except Exception as e:
        st.error(f"❌ Error reading .las file: {e}")
        st.error("Please ensure the file is in a valid LAS format.")
        return None

# --- Function to read CSV files ---
@st.cache_data
def read_csv_file(file):
    """
    Reads a .csv file and returns a DataFrame.
    """
    try:
        df = pd.read_csv(file)
        
        # Standardize columns
        df.columns = [col.upper().strip().replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', '') for col in df.columns]
        
        if not any(col in df.columns for col in LOG_NAME_MAPPING['DEPTH']):
            st.error("❌ The CSV file does not have a 'DEPTH' column.")
            return None
        
        # Set the depth column as the index
        depth_col = next(col for col in LOG_NAME_MAPPING['DEPTH'] if col in df.columns)
        df.set_index(depth_col, inplace=True)
        df.index.name = 'DEPTH'

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        st.success("✅ CSV file uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Error reading .csv file: {e}")
        return None

# --- Main application ---
def main():
    st.title("Wireline Log Petrophysical Analysis 🛢️")
    st.markdown("---")
    
    st.header("Upload Log File")
    uploaded_file = st.file_uploader("Upload a .las or .csv file", type=['las', 'csv'])

    if uploaded_file is None:
        st.info("Upload a log file to start the analysis.")
        return

    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'las':
        data = read_las_file(uploaded_file)
    else:
        data = read_csv_file(uploaded_file)
    
    if data is None:
        return
    
    # --- Manual Log Selection ---
    st.header("Select Required Logs")
    available_logs = data.columns.tolist()
    
    # Default to an empty list to avoid errors if the log is not present
    default_gr_log = next((log for log in available_logs if log.upper() in LOG_NAME_MAPPING['GR']), None)
    default_res_log = next((log for log in available_logs if log.upper() in LOG_NAME_MAPPING['RES']), None)

    st.markdown("Please select the logs for Gamma Ray, Resistivity, and Density from the available logs in your file.")

    gr_log = st.selectbox("Gamma Ray Log", available_logs, index=available_logs.index(default_gr_log) if default_gr_log in available_logs else 0)
    res_log = st.selectbox("Resistivity Log", available_logs, index=available_logs.index(default_res_log) if default_res_log in available_logs else 0)

    # Check for required logs
    if not gr_log or not res_log:
        st.error("❌ Please select both a Gamma Ray and a Resistivity log to continue.")
        return

    # Assign selected logs to standard names
    data['GR'] = data[gr_log]
    data['RES'] = data[res_log]

    # --- Analysis Settings ---
    st.header("Analysis Settings")

    # --- Shale Volume (Vshale) Calculation ---
    st.subheader("Shale Volume (Vshale) Calculation")
    col1, col2 = st.columns(2)
    with col1:
        gr_min = st.number_input("Minimum GR Value (Clean Sandstone)", value=data['GR'].quantile(0.05), help="The lowest GR value, representing clean sandstone.")
    with col2:
        gr_max = st.number_input("Maximum GR Value (Shale Point)", value=data['GR'].quantile(0.95), help="The highest GR value, representing shale.")
    
    data['IGR'] = (data['GR'] - gr_min) / (gr_max - gr_min)
    data['IGR'] = data['IGR'].clip(0, 1)
    
    data['VSHALE'] = data['IGR'] / (3 - 2 * data['IGR'])
    data['VSHALE'] = data['VSHALE'].clip(0, 1)
    st.info("✅ Vshale has been calculated based on the Gamma Ray log.")

    # --- Porosity Calculation ---
    st.subheader("Porosity Calculation")
    
    # Select Porosity Log
    porosity_options = ['None'] + [log for log in available_logs if log.upper() in LOG_NAME_MAPPING['RHOB'] or log.upper() in LOG_NAME_MAPPING['NPHI'] or log.upper() in LOG_NAME_MAPPING['DT']]
    selected_porosity_log = st.selectbox("Select Porosity Log (Optional)", porosity_options)
    
    if selected_porosity_log == 'None':
        st.warning("❌ No porosity log selected. Porosity and Saturation calculations will not be performed.")
        data['PHIE'] = np.nan
    else:
        log_type = next((key for key, values in LOG_NAME_MAPPING.items() if selected_porosity_log.upper() in values), None)
        
        if log_type == 'RHOB':
            col3, col4 = st.columns(2)
            with col3:
                phi_d_matrix = st.number_input("Matrix Density (RHOB_matrix)", value=2.71, help="Matrix density of the rock, typically 2.71 for limestone or 2.65 for sandstone.")
            with col4:
                phi_d_fluid = st.number_input("Fluid Density (RHOB_fluid)", value=1.0, help="Fluid density, typically 1.0 for fresh water.")
            data['PHIE'] = (phi_d_matrix - data[selected_porosity_log]) / (phi_d_matrix - phi_d_fluid)
            st.info("✅ Density Porosity (PHIE) has been calculated.")
        elif log_type == 'NPHI':
            data['PHIE'] = data[selected_porosity_log]
            st.info("✅ Effective Porosity (PHIE) has been set to Neutron Porosity (NPHI).")
        elif log_type == 'DT':
            st.error("❌ Sonic Porosity calculation is not yet implemented.")
            data['PHIE'] = np.nan

        if 'PHIE' in data.columns and not data['PHIE'].isnull().all():
            data['PHIE'] = data['PHIE'] * (1 - data['VSHALE'])
            data['PHIE'] = data['PHIE'].clip(lower=0)
            st.info("✅ Porosity has been corrected for shale volume.")
        else:
            data['PHIE'] = np.nan
            st.warning("❌ No porosity logs were found or selected.")

    # --- Water Saturation (Sw) Calculation ---
    st.subheader("Water Saturation (Sw) Calculation")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        a = st.number_input("Tortuosity Factor (a)", value=1.0)
    with col6:
        m = st.number_input("Cementation Exponent (m)", value=2.0)
    with col7:
        n = st.number_input("Saturation Exponent (n)", value=2.0)
    with col8:
        rw = st.number_input("Water Resistivity (Rw)", value=0.03, help="Water resistivity in ohm.m")

    if 'PHIE' in data.columns and 'RES' in data.columns and not data['PHIE'].isnull().all():
        data['SW'] = (a * rw / (data['PHIE']**m * data['RES']))**(1/n)
        data['SW'] = data['SW'].clip(upper=1)
        data['SW'] = data['SW'].fillna(1.0)
        st.info("✅ Water Saturation (Sw) has been calculated using Archie's Equation.")
    else:
        st.warning("❌ Cannot calculate Water Saturation (Sw). Please check if 'PHIE' and 'RES' logs are present and have valid data.")
        data['SW'] = np.nan

    if 'SW' in data.columns:
        data['SH'] = 1 - data['SW']
        data['SH'] = data['SH'].clip(lower=0)
        st.info("✅ Hydrocarbon Saturation (Sh) has been calculated.")
    else:
        data['SH'] = np.nan

    st.subheader("Petrophysical Analysis Results")
    st.dataframe(data.head())
    
    # --- Plotting the results ---
    fig = make_subplots(
        rows=1, cols=3, 
        shared_yaxes=True,
        horizontal_spacing=0.02,
        column_widths=[0.33, 0.33, 0.34],
        subplot_titles=("Gamma Ray & Shale", "Porosity", "Saturation")
    )
    
    # Track 1: GR and Vshale
    if 'GR' in data.columns and 'VSHALE' in data.columns:
        fig.add_trace(go.Scatter(x=data['GR'], y=data.index, mode='lines', name='GR', line=dict(color='green', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['VSHALE'] * 100, y=data.index, mode='lines', name='Vshale', line=dict(color='black', width=2)), row=1, col=1)

    # Track 2: Porosity
    if 'PHIE' in data.columns:
        fig.add_trace(go.Scatter(x=data['PHIE'] * 100, y=data.index, mode='lines', name='Effective Porosity', fill='tozerox', fillcolor='rgba(135, 206, 250, 0.5)', line=dict(color='blue', width=2)), row=1, col=2)
    
    # Track 3: Saturation
    if 'SW' in data.columns and 'SH' in data.columns:
        fig.add_trace(go.Scatter(x=data['SW'] * 100, y=data.index, mode='lines', name='Water Saturation (Sw)', fill='tozerox', fillcolor='rgba(0, 0, 255, 0.5)', line=dict(color='blue', width=2)), row=1, col=3)
        fig.add_trace(go.Scatter(x=data['SH'] * 100, y=data.index, mode='lines', name='Hydrocarbon Saturation (Sh)', fill='tonextx', fillcolor='rgba(255, 165, 0, 0.5)', line=dict(color='orange', width=2)), row=1, col=3)

    fig.update_xaxes(title_text="GR (API)", row=1, col=1, range=[0, 150])
    fig.update_xaxes(title_text="Porosity (%)", range=[0, 30], row=1, col=2)
    fig.update_xaxes(title_text="Saturation (%)", range=[0, 100], row=1, col=3)
    fig.update_yaxes(title_text="DEPTH", autorange="reversed", row=1, col=1)

    fig.update_layout(
        title_text="Petrophysical Analysis",
        height=800,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
