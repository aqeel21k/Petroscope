# Updated Streamlit App for Reservoir Analysis with separate file uploaders per tab.

# --- 1. Import necessary libraries ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import lasio
import io
from plotly.subplots import make_subplots

# --- 2. Set Streamlit page configuration ---
st.set_page_config(page_title="Reservoir Analysis", layout="wide", page_icon="🛢️")

# --- 3. Header and Title ---
st.title("Reservoir Simulation & Data Analysis")
st.markdown("---")

# --- 4. Define helper functions ---
@st.cache_data
def read_reservoir_data(uploaded_file):
    """
    Reads a reservoir data file (CSV, Excel, or LAS) and returns a DataFrame.
    """
    df = None
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'las':
            las = lasio.read(uploaded_file)
            df = las.df().reset_index()
            
        if df is not None:
            # Standardize column names
            df.columns = [c.strip().replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '').upper() for c in df.columns]
            return df
        
    except Exception as e:
        st.error(f"Error processing the file. Please check its format. Error: {e}")
        return pd.DataFrame()

@st.cache_data
def read_pvt_data(uploaded_file):
    """
    Reads a PVT data file (CSV or Excel) and returns a DataFrame.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None
        
        df.columns = [c.strip().replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '').upper() for c in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error processing PVT file. Please check its format. Error: {e}")
        return None
    
@st.cache_data
def read_core_data(uploaded_file):
    """
    Reads a core data file (CSV or Excel) and returns a DataFrame.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None
            
        df.columns = [c.strip().replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '').upper() for c in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error processing Core file. Please check its format. Error: {e}")
        return None

# --- 5. Main application layout with tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Simulation Results", "3D Reservoir Model", "PVT Data Analysis", "Core Data Analysis"])

# --- Tab 1: Simulation Results ---
with tab1:
    st.header("Simulation Results: Time-Based Plots")
    st.markdown("Please upload a CSV, Excel (.xlsx/.xls), or LAS file.")
    uploaded_file_sim = st.file_uploader("Choose a file for Simulation Data", key="sim_uploader", type=["csv", "xlsx", "las"])

    if uploaded_file_sim is not None:
        st.info("Processing simulation data...")
        df_time = read_reservoir_data(uploaded_file_sim)
        
        if not df_time.empty:
            st.success("Simulation data processed successfully!")
            
            with st.expander("Basic Data Statistics"):
                st.write(df_time.describe())

            if 'OIL_PRODUCTION' in df_time.columns and 'TIME' in df_time.columns:
                st.subheader("Oil Production vs Time")
                fig_oil = px.line(df_time, x='TIME', y='OIL_PRODUCTION', title='Oil Production Over Time', markers=True)
                fig_oil.update_yaxes(title_text="Oil Production (bbl)")
                st.plotly_chart(fig_oil, use_container_width=True)
            
            if 'WATER_CUT' in df_time.columns and 'TIME' in df_time.columns:
                st.subheader("Water Cut vs Time")
                fig_wc = px.line(df_time, x='TIME', y='WATER_CUT', title='Water Cut Over Time', markers=True)
                fig_wc.update_yaxes(title_text="Water Cut (%)")
                st.plotly_chart(fig_wc, use_container_width=True)
        else:
            st.warning("No time-based data found or file could not be processed.")

# Ensure necessary libraries are imported at the top of your script
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata

# This is a dummy function, assuming you have a proper `read_reservoir_data` function
# If your function is different, keep your original one.
def read_reservoir_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.las'):
        # Placeholder for LAS file reading, assuming it returns a DataFrame
        st.error("LAS file support is not fully implemented in this example.")
        return pd.DataFrame()
    return pd.DataFrame()

# --- Tab 2: 3D Reservoir Model ---
with tab2:
    st.header("3D Reservoir Model")
    st.markdown("This tab visualizes the 3D reservoir based on a property from your data.")
    
    uploaded_file_3d = st.file_uploader("Choose a file for 3D Reservoir Data", key="3d_uploader", type=["csv", "xlsx", "las"])
    
    if uploaded_file_3d is not None:
        st.info("Processing 3D model data...")
        df_3d = read_reservoir_data(uploaded_file_3d)

        if not df_3d.empty:
            st.success("3D model data processed successfully!")
            required_reservoir_cols = ['I', 'J', 'K', 'POROSITY']
            
            # Check for required columns
            if all(col in df_3d.columns for col in required_reservoir_cols):
                grid_cols = ['I', 'J', 'K']
                property_columns = [col for col in df_3d.columns if col not in grid_cols]
                
                selected_property = st.selectbox(
                    "Select a property to visualize:",
                    options=property_columns,
                    index=property_columns.index('POROSITY') if 'POROSITY' in property_columns else 0
                )
                
                # --- Select plot type ---
                plot_type = st.radio(
                    "Select 3D plot type:",
                    ('3D Scatter Plot', 'Isosurface Plot', '3D Volume Plot')
                )
                
                # Drop rows with NaN values in the selected property columns
                df_3d_clean = df_3d.dropna(subset=[selected_property] + grid_cols)
                
                if not df_3d_clean.empty:
                    st.subheader(f"3D Reservoir {selected_property} Distribution")
                    
                    if plot_type == '3D Scatter Plot':
                        fig_3d = px.scatter_3d(
                            df_3d_clean,
                            x='I',
                            y='J',
                            z='K',
                            color=selected_property,
                            color_continuous_scale='Viridis',
                            title=f"3D Scatter Plot of {selected_property}"
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

                    elif plot_type == 'Isosurface Plot' or plot_type == '3D Volume Plot':
                        # This is the corrected section for interpolation
                        # Prepare data for interpolation
                        points = df_3d_clean[grid_cols].values
                        values = df_3d_clean[selected_property].values
                        
                        # Create a regular grid for interpolation
                        i_vals = np.linspace(df_3d_clean['I'].min(), df_3d_clean['I'].max(), 50)
                        j_vals = np.linspace(df_3d_clean['J'].min(), df_3d_clean['J'].max(), 50)
                        k_vals = np.linspace(df_3d_clean['K'].min(), df_3d_clean['K'].max(), 50)
                        I, J, K = np.meshgrid(i_vals, j_vals, k_vals)
                        
                        # Interpolate the data onto the new grid
                        gridded_data = griddata(points, values, (I, J, K), method='linear')
                        
                        # Replace NaNs with a safe value (e.g., 0 or the mean) to prevent plotting errors
                        gridded_data = np.nan_to_num(gridded_data, nan=0)

                        if plot_type == 'Isosurface Plot':
                            if st.checkbox('Control Isosurface Value', value=False, key="isosurface_value_control"):
                                min_val = np.nanmin(gridded_data)
                                max_val = np.nanmax(gridded_data)
                                isovalue = st.slider(
                                    f"Select {selected_property} Isosurface Value",
                                    float(min_val), float(max_val), float((min_val + max_val) / 2),
                                    key="isosurface_slider"
                                )
                                fig_3d = go.Figure(data=go.Isosurface(
                                    x=I.flatten(),
                                    y=J.flatten(),
                                    z=K.flatten(),
                                    value=gridded_data.flatten(),
                                    isomin=isovalue,
                                    isomax=isovalue,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar_title=selected_property
                                ))
                            else:
                                fig_3d = go.Figure(data=go.Isosurface(
                                    x=I.flatten(),
                                    y=J.flatten(),
                                    z=K.flatten(),
                                    value=gridded_data.flatten(),
                                    surface_count=5,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar_title=selected_property
                                ))

                        elif plot_type == '3D Volume Plot':
                            fig_3d = go.Figure(data=go.Volume(
                                x=I.flatten(),
                                y=J.flatten(),
                                z=K.flatten(),
                                value=gridded_data.flatten(),
                                opacity=0.1,
                                surface_count=10,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar_title=selected_property
                            ))
                        
                        # Update layout for all 3D plots
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis_title='I Index',
                                yaxis_title='J Index',
                                zaxis_title='K Index',
                                aspectmode='cube'
                            ),
                            height=800
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # --- Additional Plots (Your old code, integrated here) ---
                    st.markdown("---")
                    st.subheader("Additional Reservoir Plots")

                    if 'POROSITY' in df_3d.columns:
                        st.subheader("Porosity Distribution Histogram")
                        fig_hist = go.Figure(data=[go.Histogram(x=df_3d['POROSITY'])])
                        fig_hist.update_layout(
                            title_text='Porosity Distribution',
                            xaxis_title_text='Porosity',
                            yaxis_title_text='Count',
                            bargap=0.2
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    if 'PERMEABILITY' in df_3d.columns:
                        st.subheader("Porosity vs. Permeability")
                        fig_scatter = go.Figure(data=go.Scatter(
                            x=df_3d['POROSITY'],
                            y=df_3d['PERMEABILITY'],
                            mode='markers'
                        ))
                        fig_scatter.update_layout(
                            title_text='Porosity vs. Permeability',
                            xaxis_title_text='Porosity',
                            yaxis_title_text='Permeability (mD)',
                            yaxis_type='log'
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    if 'FACIES' in df_3d.columns:
                        st.subheader("Facies Distribution")
                        facies_counts = df_3d['FACIES'].value_counts()
                        fig_pie = go.Figure(data=[go.Pie(labels=facies_counts.index, values=facies_counts.values, hole=.3)])
                        fig_pie.update_traces(textinfo='percent+label')
                        fig_pie.update_layout(title_text='Reservoir Facies Breakdown')
                        st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("Selected property column contains no valid data for plotting. Please check your data.")
            else:
                st.warning("Could not find the necessary grid columns (I, J, K, and a property like POROSITY) for 3D plotting. Please check your file format.")
        else:
            st.warning("Please upload a valid 3D reservoir data file.")
    else:
        st.info("Please upload a 3D reservoir data file to view the analysis.")


# --- Tab 3: PVT Data Analysis ---
with tab3:
    st.header("PVT Data Analysis")
    st.markdown("Please upload a CSV or Excel file containing your PVT data.")
    
    uploaded_file_pvt = st.file_uploader("Choose a file for PVT Data", key="pvt_uploader", type=["csv", "xlsx"])
    
    if uploaded_file_pvt is not None:
        st.info("Processing PVT data...")
        df_pvt = read_pvt_data(uploaded_file_pvt)
        
        if df_pvt is not None and not df_pvt.empty:
            st.success("PVT data processed successfully!")
            
            if 'PRESSURE' in df_pvt.columns and 'BO' in df_pvt.columns:
                st.subheader("Bo (Oil FVF) vs Pressure")
                fig_bo = px.line(df_pvt, x='PRESSURE', y='BO', title='Bo vs Pressure')
                fig_bo.update_yaxes(title_text="Oil FVF (rb/stb)")
                st.plotly_chart(fig_bo, use_container_width=True)

            if 'PRESSURE' in df_pvt.columns and 'RS' in df_pvt.columns:
                st.subheader("Rs (Solution GOR) vs Pressure")
                fig_rs = px.line(df_pvt, x='PRESSURE', y='RS', title='Rs vs Pressure')
                fig_rs.update_yaxes(title_text="Solution GOR (scf/stb)")
                st.plotly_chart(fig_rs, use_container_width=True)
            
            if 'PRESSURE' in df_pvt.columns and 'VISCOSITY' in df_pvt.columns:
                st.subheader("Viscosity vs Pressure")
                fig_visc = px.line(df_pvt, x='PRESSURE', y='VISCOSITY', title='Viscosity vs Pressure')
                fig_visc.update_yaxes(title_text="Viscosity (cp)")
                st.plotly_chart(fig_visc, use_container_width=True)
        else:
            st.warning("Please upload a valid PVT data file.")
    else:
        st.info("Please upload a PVT data file to view the analysis.")
        
# --- Tab 4: Core Data Analysis ---
with tab4:
    st.header("Core Data Analysis")
    st.markdown("Please upload a CSV or Excel file containing your Core data.")
    
    uploaded_file_core = st.file_uploader("Choose a file for Core Data", key="core_uploader", type=["csv", "xlsx"])
    
    if uploaded_file_core is not None:
        st.info("Processing Core data...")
        df_core = read_core_data(uploaded_file_core)
        
        if df_core is not None and not df_core.empty:
            st.success("Core data processed successfully!")
            
            # Check for required columns for plotting
            required_cols = ['DEPTH', 'POROSITY', 'PERMEABILITY']
            
            if all(col in df_core.columns for col in required_cols):
                st.subheader("Core Data Logs")
                
                # Create a multi-trace vertical log plot for Core data
                fig_core = make_subplots(
                    rows=1, cols=2, 
                    shared_yaxes=True,
                    horizontal_spacing=0.02,
                    subplot_titles=("Porosity", "Permeability")
                )
                
                # Add Porosity trace
                fig_core.add_trace(
                    go.Scatter(
                        x=df_core['POROSITY'], 
                        y=df_core['DEPTH'], 
                        mode='lines', 
                        name='Core Porosity', 
                        line=dict(color='blue')
                    ), 
                    row=1, col=1
                )
                
                # Add Permeability trace
                fig_core.add_trace(
                    go.Scatter(
                        x=df_core['PERMEABILITY'], 
                        y=df_core['DEPTH'], 
                        mode='lines', 
                        name='Core Permeability', 
                        line=dict(color='orange')
                    ), 
                    row=1, col=2
                )
                
                # Update layout and axes for a vertical log plot
                fig_core.update_yaxes(
                    title_text="Depth (ft)", 
                    autorange="reversed", 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgray'
                )
                fig_core.update_xaxes(
                    title_text="Porosity (%)", 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    row=1, col=1
                )
                fig_core.update_xaxes(
                    title_text="Permeability (mD)", 
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    row=1, col=2
                )
                
                fig_core.update_layout(
                    title_text="Core Data Analysis - Porosity & Permeability Logs",
                    height=800,
                    showlegend=True
                )
                
                st.plotly_chart(fig_core, use_container_width=True)

            else:
                st.warning("The uploaded file must contain 'DEPTH', 'POROSITY', and 'PERMEABILITY' columns to display the logs.")
            
            st.markdown("---")
            st.subheader("Key Concepts in Petrophysical Analysis")
            st.markdown("""
            * **Porosity:** The ratio of pore volume to the total rock volume, indicating the capacity of the rock to hold fluids.
            * **Permeability:** A measure of the ability of a porous material to allow fluids to pass through it, affecting the flow rate of oil, gas, or water.
            * **Water Saturation:** The fraction of the pore volume occupied by water.
            * **Bulk Volume:** The total volume of the rock, including both the solid rock matrix and the pores.
            """)
        else:
            st.warning("Please upload a valid Core data file.")
    else:
        st.info("Please upload a Core data file to view the analysis.")
