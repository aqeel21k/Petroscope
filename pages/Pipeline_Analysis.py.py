import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
from plotly.subplots import make_subplots

# --- Define helper functions ---
@st.cache_data
def read_pipeline_data(uploaded_file):
    """
    Reads a pipeline data file (CSV or Excel) and returns a DataFrame.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None
        
        # Standardize column names for consistency
        df.columns = [c.strip().replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '').upper() for c in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error processing the file. Please check its format. Error: {e}")
        return None

# --- Beggs and Brill function ---
def beggs_and_brill(rho_g, rho_l, mu_g, mu_l, sigma, V_sg, V_sl, D, L, angle, P_avg, T_avg):
    """
    Calculates pressure drop and determines flow regime using Beggs and Brill correlation.
    All units must be in SI.
    """
    g = 9.81  # m/s^2

    # Beggs and Brill constants
    C_c = (1 - (0.011 * math.log(V_sl**2)))
    L_1 = 3.16 * V_sg**0.353
    L_2 = 0.009 * V_sl**(-0.54)
    L_3 = 0.62 * V_sg**0.415
    L_4 = 0.0011 * V_sl**(-1.75)
    
    # No-slip holdup
    E_l_ns = V_sl / (V_sg + V_sl)
    
    # No-slip Froude number
    Fr_ns = (V_sg + V_sl)**2 / (g * D)
    
    # Calculate holdup correction factor 'a', 'b', and 'c'
    if E_l_ns < 0.01:
        a = 0.98
        b = 0.4846
        c = 0.0868
    elif E_l_ns < 0.4:
        a = (0.845 - 0.28 * E_l_ns)
        b = 0.5309 - 0.1708 * E_l_ns
        c = 0.0173 - 0.0203 * E_l_ns
    else:
        a = (0.52 - 0.24 * E_l_ns)
        b = 0.5824 - 0.354 * E_l_ns
        c = 0.0468 - 0.063 * E_l_ns

    # Holdup for horizontal flow
    E_l_horz = a * E_l_ns**b / Fr_ns**c
    
    # Determine flow regime
    if L_1 < Fr_ns < L_3:
        flow_regime = "Transition"
        # In transition flow, use no-slip holdup for stability
        E_l = E_l_ns
    elif Fr_ns <= L_1 and E_l_ns < L_2:
        flow_regime = "Segregated"
        E_l = E_l_horz
    elif Fr_ns <= L_1 and E_l_ns >= L_2:
        flow_regime = "Intermittent"
        E_l = E_l_horz
    elif Fr_ns >= L_3 and E_l_ns < L_4:
        flow_regime = "Distributed"
        E_l = E_l_horz
    elif Fr_ns >= L_3 and E_l_ns >= L_4:
        flow_regime = "Distributed"
        E_l = E_l_horz
    else:
        flow_regime = "Undefined"
        E_l = E_l_horz
        
    E_g = 1 - E_l

    # Inclination correction
    B = (1 - E_l_horz) * (1.2 - E_l_horz)**-1.4
    
    psi = 1 + B * (math.sin(math.radians(angle)) - 0.33 * math.sin(math.radians(angle))**3)
    
    E_l_inc = E_l_horz * psi
    E_l = E_l_inc if abs(angle) > 0.1 else E_l_horz
    E_g = 1 - E_l

    # Frictional pressure drop
    Re_tp = (rho_g * V_sg * D) / mu_g
    f_ns = 0.0055 * (1 + (20000 * E_l_ns)) if Re_tp < 2000 else 0.0055 * (1 + (20000 * E_l_ns))
    
    # Holdup for frictional pressure drop calculation
    frictional_holdup = E_l_horz if abs(angle) > 0.1 else E_l_inc

    y = frictional_holdup / E_l_ns**2
    S = math.log(y) / (math.log(1 / E_l_ns**2)) if y > 1 else 0
    
    f_tp = f_ns * math.exp(S)

    friction_drop = (f_tp * L * (V_sg + V_sl)**2) / (2 * D * g)
    
    # Gravitational pressure drop
    gravitational_drop = (rho_l * E_l + rho_g * E_g) * g * L * math.sin(math.radians(angle))
    
    # Acceleration pressure drop (negligible for most cases)
    acceleration_drop = 0
    
    total_drop = friction_drop + gravitational_drop + acceleration_drop
    
    return total_drop, flow_regime, E_l, Fr_ns

# --- Set page configuration ---
st.set_page_config(page_title="Pipeline Analysis", layout="wide", page_icon="🛢️")

st.title("Pipeline Analysis Dashboard")
st.markdown("---")

# Create tabs to organize content
tab1, tab2 = st.tabs(["Pipeline Data Analysis", "Beggs and Brill Calculator"])

# --- Tab 1: Pipeline Data Analysis ---
with tab1:
    st.header("1. Upload Your Pipeline Data")
    st.markdown("Please ensure your file contains columns for analysis. This single upload will be used for all analyses in this tab.")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = read_pipeline_data(uploaded_file)
        
        if df is not None:
            st.success("File uploaded and processed successfully!")
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Check for required columns for each analysis section
            required_corrosion_cols = ['DISTANCE', 'CORROSION_RATE']
            required_flow_cols = ['DISTANCE', 'PRESSURE', 'FLOW_RATE']
            required_integrity_cols = ['X', 'Y', 'Z', 'DEFECT_TYPE', 'DEFECT_SIZE']
            
            # --- 2. Corrosion Analysis ---
            st.markdown("---")
            st.header("2. Corrosion Analysis")
            if all(col in df.columns for col in required_corrosion_cols):
                fig_corrosion = px.line(df, x='DISTANCE', y='CORROSION_RATE', title='Corrosion Rate vs. Pipeline Distance')
                fig_corrosion.update_yaxes(title_text='Corrosion Rate (mm/year)')
                st.plotly_chart(fig_corrosion, use_container_width=True)
            else:
                st.warning(f"Required columns for Corrosion Analysis are missing. Please ensure your file contains: {', '.join(required_corrosion_cols)}")
                
            # --- 3. Hydraulic Flow Analysis ---
            st.markdown("---")
            st.header("3. Hydraulic Flow Analysis")
            if all(col in df.columns for col in required_flow_cols):
                fig_flow = go.Figure()
                fig_flow.add_trace(go.Scatter(x=df['DISTANCE'], y=df['PRESSURE'], mode='lines', name='Pressure', yaxis='y1'))
                fig_flow.add_trace(go.Scatter(x=df['DISTANCE'], y=df['FLOW_RATE'], mode='lines', name='Flow Rate', yaxis='y2'))
                fig_flow.update_layout(
                    title_text='Pressure and Flow Rate vs. Pipeline Distance',
                    xaxis_title='Distance (km)',
                    yaxis=dict(title='Pressure (bar)'),
                    yaxis2=dict(title='Flow Rate (m³/h)', overlaying='y', side='right')
                )
                st.plotly_chart(fig_flow, use_container_width=True)
            else:
                st.warning(f"Required columns for Hydraulic Flow Analysis are missing. Please ensure your file contains: {', '.join(required_flow_cols)}")

            # --- 4. Pipeline Integrity Analysis (3D Plot) ---
            st.markdown("---")
            st.header("4. Pipeline Integrity Analysis")
            st.markdown(f"**This section requires columns: {', '.join(required_integrity_cols)}**")

            if all(col in df.columns for col in required_integrity_cols):
                st.subheader("Interactive Pipeline Defect Visualization")
                
                # --- Add Interactive Filters ---
                st.markdown("#### Filter Data")
                col1, col2 = st.columns(2)
                with col1:
                    defect_types = df['DEFECT_TYPE'].unique()
                    selected_defects = st.multiselect("Select Defect Type(s)", options=defect_types, default=list(defect_types))
                
                with col2:
                    min_size = df['DEFECT_SIZE'].min()
                    max_size = df['DEFECT_SIZE'].max()
                    size_range = st.slider("Select Defect Size Range", float(min_size), float(max_size), (float(min_size), float(max_size)))

                filtered_df = df[
                    (df['DEFECT_TYPE'].isin(selected_defects)) &
                    (df['DEFECT_SIZE'] >= size_range[0]) &
                    (df['DEFECT_SIZE'] <= size_range[1])
                ]
                
                if not filtered_df.empty:
                    fig_integrity = px.scatter_3d(filtered_df, 
                                                x='X', 
                                                y='Y', 
                                                z='Z', 
                                                color='DEFECT_TYPE',
                                                size='DEFECT_SIZE',
                                                title='Pipeline Defect Visualization',
                                                hover_data={'X': True, 'Y': True, 'Z': True, 'DEFECT_TYPE': True, 'DEFECT_SIZE': True})
                    
                    fig_integrity.update_layout(scene = dict(
                                        xaxis_title='X Coordinate',
                                        yaxis_title='Y Coordinate',
                                        zaxis_title='Z Coordinate'))
                    
                    st.plotly_chart(fig_integrity, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Key Statistics")
                    stats_df = filtered_df.groupby('DEFECT_TYPE').agg(
                        Count=('X', 'count'),
                        Average_Size=('DEFECT_SIZE', 'mean'),
                        Max_Size=('DEFECT_SIZE', 'max')
                    ).reset_index()
                    st.dataframe(stats_df.rename(columns={'DEFECT_TYPE': 'Defect Type', 'Count': 'Number of Defects', 'Average_Size': 'Avg. Size', 'Max_Size': 'Max Size'}))
                    
                else:
                    st.warning("No data points match the selected filters. Please adjust your selections.")
            else:
                st.warning("Could not find the necessary columns for Pipeline Integrity Analysis. Please check your file.")
        else:
            st.info("Please upload a valid data file to view the analysis.")
    else:
        st.info("Please upload a data file to start the analysis.")

# --- Tab 2: Beggs and Brill Calculator ---
with tab2:
    st.header("Beggs and Brill Pressure Drop Calculator")
    st.markdown("This section calculates the pressure drop and determines the flow regime using the Beggs and Brill correlation.")

    # --- New Section for Diagrams and Explanations ---
    st.subheader("Flow Regime Diagrams and Explanations")
    st.markdown("""
        The Beggs and Brill correlation helps to determine the **flow regime** within the pipeline. The flow regime is the way the liquid and gas move together inside the pipe. Understanding it is crucial for accurate pressure drop calculations.

        **The main flow regimes are:**

        * **Segregated Flow:** This occurs when the liquid and gas separate into layers. In horizontal pipes, the gas flows at the top and the liquid at the bottom.
        * **Intermittent Flow:** This is characterized by liquid "slugs" that move along the pipe, separated by gas pockets.
        * **Distributed Flow:** This happens when one phase is dispersed within the other.
        * **Transition Flow:** This occurs when the flow is between different regimes, which makes calculations less accurate.
        
        $
        \text{Flow Regimes Diagram}
        $
            """)
    st.markdown("---")
    
    # --- Input fields for the calculator ---
    st.subheader("Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        rho_l_input = st.number_input("Liquid Density (kg/m³)", value=800.0)
        mu_l_input = st.number_input("Liquid Viscosity (Pa·s)", value=0.001)
        V_sl_input = st.number_input("Superficial Liquid Velocity (m/s)", value=0.5)
        D_input = st.number_input("Pipe Diameter (m)", value=0.1)

    with col2:
        rho_g_input = st.number_input("Gas Density (kg/m³)", value=50.0)
        mu_g_input = st.number_input("Gas Viscosity (Pa·s)", value=2e-5, format="%e")
        V_sg_input = st.number_input("Superficial Gas Velocity (m/s)", value=2.0)
        L_input = st.number_input("Pipe Length (m)", value=1000.0)

    with col3:
        sigma_input = st.number_input("Surface Tension (N/m)", value=0.03)
        angle_input = st.number_input("Pipe Angle from Horizontal (degrees)", value=5.0)
        P_avg_input = st.number_input("Average Pressure (Pa)", value=1000000.0, format="%e")
        T_avg_input = st.number_input("Average Temperature (K)", value=298.0)

    # Calculate button
    st.subheader("Calculate")
    if st.button("Calculate Pressure Drop"):
        try:
            total_drop, flow_regime, E_l, Fr_ns = beggs_and_brill(
                rho_g=rho_g_input,
                rho_l=rho_l_input,
                mu_g=mu_g_input,
                mu_l=mu_l_input,
                sigma=sigma_input,
                V_sg=V_sg_input,
                V_sl=V_sl_input,
                D=D_input,
                L=L_input,
                angle=angle_input,
                P_avg=P_avg_input,
                T_avg=T_avg_input
            )
            
            st.markdown("### Results")
            st.success(f"**Calculated Pressure Drop:** {total_drop:.2f} Pa")
            st.info(f"**Identified Flow Regime:** {flow_regime}")
            st.write(f"**Liquid Holdup ($$E_l$$):** {E_l:.4f}")
            st.write(f"**No-slip Froude Number ($$Fr_{{ns}}$$):** {Fr_ns:.4f}")
            
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")
