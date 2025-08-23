import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
import lasio
import math
from scipy.stats import linregress

# --- Main App Configuration ---
st.set_page_config(page_title="Petroleum Engineering Analysis", layout="wide", page_icon="🛢️")
st.title("Petroleum Engineering Dashboard")

# --- Helper function to read data ---
@st.cache_data
def read_data(uploaded_file):
    """
    Reads a data file (CSV, Excel, or LAS) and returns a DataFrame.
    Includes robust error handling for common file format issues.
    """
    df = None
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            try:
                # First attempt: standard CSV read
                df = pd.read_csv(uploaded_file)
            except pd.errors.ParserError:
                # Second attempt: try with a different separator or error handling
                try:
                    df = pd.read_csv(uploaded_file, sep=';') # Try with semicolon as separator
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(uploaded_file, on_bad_lines='skip') # Skip problematic lines
                    except Exception as e:
                        st.error(f"Error processing CSV file. Please check its format. Original error: {e}")
                        return None
        elif file_extension in ['xlsx', 'xls']:
            try:
                # Specify engine to handle different Excel formats
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except ImportError:
                st.error("Please install openpyxl to read .xlsx files: pip install openpyxl")
                return None
            except Exception as e:
                try:
                    df = pd.read_excel(uploaded_file, engine='xlrd')
                except ImportError:
                    st.error("Please install xlrd to read .xls files: pip install xlrd")
                    return None
                except Exception as e:
                    st.error(f"Error processing Excel file. Please check its format. Error: {e}")
                    return None
        elif file_extension == 'las':
            try:
                las = lasio.read(uploaded_file)
                df = las.df().reset_index()
                # Clean up column names for consistency
                df.columns = [c.strip().replace(':', '_').upper() for c in las.keys()]
            except Exception as e:
                st.error(f"Error processing LAS file. Please check its format. Error: {e}")
                return None
        
        # Clean up column names in general to handle case and spacing
        if df is not None:
            df.columns = [c.strip().replace(' ', '_').replace('.', '_').upper() for c in df.columns]
            
        return df
            
    except Exception as e:
        st.error(f"Error processing the file. Please check its format. A general error occurred: {e}")
        return None

# --- Main Content Tabs ---
main_tab1, main_tab2, main_tab3 = st.tabs(["Production Analysis", "Water Cut & GOR Analysis", "Well Test Analysis"])

# --- Tab 1: Production Summary & Analysis ---
with main_tab1:
    st.header("Production Analysis & Forecasting")
    st.markdown("Please upload a CSV, Excel, or LAS file containing production data.")

    # Single file uploader for Production Analysis tab
    uploaded_file = st.file_uploader("Upload your CSV, Excel, or LAS file here", type=["csv", "xlsx", "las"], key="tab1_uploader")

    if uploaded_file is not None:
        df = read_data(uploaded_file)
        if df is not None:
            st.success("File uploaded successfully!")
            
            # --- Data Management Section (Sidebar) ---
            st.sidebar.header("Data Management")
            all_columns = df.columns.tolist()
            
            # Dynamic selection for well name and date columns
            date_column = st.sidebar.selectbox('Select Date Column:', all_columns, key="date_col_1")
            well_name_column = st.sidebar.selectbox('Select Well Name Column:', ['None'] + all_columns, key="well_name_col_1")

            if well_name_column != 'None':
                selected_wells = st.sidebar.multiselect(
                    'Select Wells to Compare:',
                    df[well_name_column].unique(),
                    default=df[well_name_column].unique().tolist()
                )
            else:
                selected_wells = []
            
            production_columns = st.sidebar.multiselect(
                'Select Production Logs:',
                [col for col in all_columns if col not in [date_column, well_name_column]],
                default=[col for col in all_columns if 'OIL' in col.upper() or 'GAS' in col.upper()]
            )

            # --- Start of Tab 1 Content ---
            st.subheader('Filtered Production Data Over Time')
            if date_column in df.columns and production_columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                df = df.dropna(subset=[date_column])
                df = df.sort_values(by=date_column)
                
                if well_name_column != 'None' and selected_wells:
                    df_filtered = df[df[well_name_column].isin(selected_wells)].copy()
                else:
                    df_filtered = df.copy()

                if not df_filtered.empty:
                    # Melt the dataframe for easy plotting with Plotly Express
                    id_vars_list = [date_column]
                    if well_name_column != 'None':
                        id_vars_list.append(well_name_column)

                    df_long = df_filtered.melt(
                        id_vars=id_vars_list,
                        value_vars=[col for col in production_columns if col in df_filtered.columns],
                        var_name='Log Type',
                        value_name='Rate'
                    )
                    
                    if not df_long.empty:
                        fig_prod = px.line(
                            df_long,
                            x=date_column,
                            y='Rate',
                            color='Log Type',
                            facet_col=well_name_column if well_name_column != 'None' else None,
                            facet_col_wrap=2,
                            title='Production Over Time'
                        )
                        fig_prod.update_layout(height=600)
                        st.plotly_chart(fig_prod)
                    else:
                        st.info("No data to display. Please check your production log selections.")
                else:
                    st.info("No data found for the selected wells. Please check your well selections.")
            else:
                st.info("Please select a date column and at least one production log from the sidebar to view the plot.")

            st.divider()

            # --- Productivity Index (PI) Analysis ---
            with st.expander("Productivity Index (PI) Analysis", expanded=False):
                st.subheader('Calculate and Compare Productivity Index (PI)')
                st.markdown("Use this section to calculate the Productivity Index for multiple wells and compare them.")
                
                pi_df_analysis = df.copy()

                q_col_pi = st.selectbox('Select Flow Rate (q) Column:', all_columns, key="pi_q_col")
                pwf_col_pi = st.selectbox('Select Bottom-Hole Flowing Pressure (Pwf) Column:', all_columns, key="pi_pwf_col")
                
                # New approach: Manual input for P_res and PI calculation
                st.info("Enter a constant value for Reservoir Pressure ($P_{res}$) as it is a single value per well/reservoir.")
                p_res_input = st.number_input('Enter Reservoir Pressure ($P_{res}$):', min_value=1.0, value=3000.0, step=100.0)

                if q_col_pi in df.columns and pwf_col_pi in df.columns and well_name_column != 'None' and selected_wells:
                    pi_df_filtered = pi_df_analysis[pi_df_analysis[well_name_column].isin(selected_wells)].copy()
                    pi_df_filtered = pi_df_filtered.dropna(subset=[q_col_pi, pwf_col_pi])

                    if not pi_df_filtered.empty:
                        pi_df_filtered['PI'] = pi_df_filtered[q_col_pi] / (p_res_input - pi_df_filtered[pwf_col_pi])
                        
                        avg_pi_df = pi_df_filtered.groupby(well_name_column)['PI'].mean().reset_index()
                        
                        st.subheader("Productivity Index Results")
                        st.write("Average PI per well:")
                        st.dataframe(avg_pi_df.style.format({'PI': "{:.2f}"}), use_container_width=True)
                        
                        fig_pi = px.bar(
                            avg_pi_df.sort_values(by='PI', ascending=False),
                            x=well_name_column,
                            y='PI',
                            title='Average Productivity Index Comparison'
                        )
                        st.plotly_chart(fig_pi)
                        
                        best_well = avg_pi_df.loc[avg_pi_df['PI'].idxmax()]
                        st.success(f"**The well with the highest average Productivity Index is:** {best_well[well_name_column]} (PI: {best_well['PI']:.2f})")
                    else:
                        st.warning("Insufficient data to perform PI analysis. Please check your column selections and filter.")
                else:
                    st.info("Please select the required columns ('Flow Rate', 'Pwf') and a well name column to perform PI analysis.")

            st.divider()

            # --- Decline Curve Analysis (DCA) ---
            with st.expander("Decline Curve Analysis (DCA)", expanded=False):
                st.subheader("Decline Curve Analysis (DCA)")
                if production_columns and well_name_column != 'None':
                    decline_log = st.selectbox('Select Log for DCA:', production_columns, key="dca_log")
                    
                    if decline_log:
                        st.subheader(f"Arps Hyperbolic Decline Model for {decline_log} (per well)")
                        
                        fig_dca = go.Figure()
                        for well in selected_wells:
                            df_dca = df[df[well_name_column] == well].dropna(subset=[decline_log, date_column])
                            if not df_dca.empty and len(df_dca) > 5:
                                df_dca['Time'] = (df_dca[date_column] - df_dca[date_column].min()).dt.days
                                
                                def hyperbolic_decline(t, qi, Di, b):
                                    return qi / ((1 + b * Di * t)**(1/b))
                                
                                try:
                                    popt, pcov = curve_fit(hyperbolic_decline, df_dca['Time'], df_dca[decline_log], maxfev=10000, bounds=(0, np.inf))
                                    qi_fit, di_fit, b_fit = popt
                                    
                                    t_forecast = np.arange(df_dca['Time'].min(), df_dca['Time'].max() + 365)
                                    q_forecast = hyperbolic_decline(t_forecast, qi_fit, di_fit, b_fit)
                                    
                                    fig_dca.add_trace(go.Scatter(x=df_dca[date_column], y=df_dca[decline_log], mode='markers', name=f'Actual Data ({well})'))
                                    date_forecast = df_dca[date_column].min() + pd.to_timedelta(t_forecast, unit='D')
                                    fig_dca.add_trace(go.Scatter(x=date_forecast, y=q_forecast, mode='lines', name=f'Forecast ({well})', line=dict(dash='dash')))
                                except (RuntimeError, ValueError) as e:
                                    st.warning(f"Could not fit the hyperbolic model for well: {well}. Error: {e}")
                            else:
                                st.info(f"Insufficient data for DCA on well: {well}")

                        if fig_dca.data:
                            fig_dca.update_layout(title=f'DCA for {decline_log}', xaxis_title='Date', yaxis_title=f'{decline_log} Rate')
                            st.plotly_chart(fig_dca)
                        else:
                            st.info("No valid data found to plot DCA.")
                    else:
                        st.info("Please select a log and ensure a well name column is present to perform DCA.")

            st.divider()

            # --- Production Forecasting with ARIMA ---
            with st.expander("Production Forecasting with ARIMA", expanded=False):
                st.subheader("Forecast Production using ARIMA Model")
                
                if production_columns and well_name_column != 'None':
                    forecast_log = st.selectbox('Select Log for Forecasting:', production_columns, key="forecast_log")
                    well_for_forecast = st.selectbox('Select Well for ARIMA Forecast:', selected_wells, key="arima_well")

                    if forecast_log and well_for_forecast:
                        df_forecast = df[df[well_name_column] == well_for_forecast].dropna(subset=[forecast_log, date_column]).copy()
                        if not df_forecast.empty:
                            df_forecast.set_index(date_column, inplace=True)
                            
                            st.write("ARIMA Model Parameters:")
                            p = st.slider('p (AR term)', 0, 5, 2, key="p_arima")
                            d = st.slider('d (I term)', 0, 5, 1, key="d_arima")
                            q = st.slider('q (MA term)', 0, 5, 2, key="q_arima")
                            
                            forecast_periods = st.number_input("Number of forecast periods (days)", min_value=1, value=365, step=1, key="forecast_periods")
                            
                            try:
                                model = ARIMA(df_forecast[forecast_log], order=(p,d,q))
                                model_fit = model.fit()
                                
                                forecast = model_fit.forecast(steps=forecast_periods)
                                
                                fig_forecast = go.Figure()
                                fig_forecast.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast[forecast_log], mode='lines', name='Historical Data'))
                                
                                forecast_dates = pd.date_range(start=df_forecast.index[-1], periods=forecast_periods + 1, inclusive='right')
                                fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='ARIMA Forecast', line=dict(color='orange', dash='dash')))
                                
                                fig_forecast.update_layout(title=f'ARIMA Production Forecast for {forecast_log} on {well_for_forecast}', xaxis_title='Date', yaxis_title=f'{forecast_log} Rate')
                                st.plotly_chart(fig_forecast)
                            except Exception as e:
                                st.warning(f"Could not fit ARIMA model. Please check the data and parameters. Error: {e}")
                        else:
                            st.info(f"Insufficient data for ARIMA forecast on well: {well_for_forecast}")
                    else:
                        st.info("Please select a log and a well name column to perform forecasting.")

            st.divider()

            # --- Inflow Performance Relationship (IPR) Curve ---
            with st.expander("Inflow Performance Relationship (IPR) Curve", expanded=False):
                st.subheader('IPR Analysis')
                flow_rate_col = st.selectbox('Select Flow Rate Column:', all_columns, key="ipr_q_col")
                pwf_col = st.selectbox('Select Pwf Column:', all_columns, key="ipr_pwf_col")
                
                # New approach: Manual input for P_res
                st.info("Enter a constant value for Reservoir Pressure ($P_{res}$).")
                p_res_input_ipr = st.number_input('Enter Reservoir Pressure ($P_{res}$):', min_value=1.0, value=3000.0, step=100.0, key="ipr_p_res_input")

                if flow_rate_col in df.columns and pwf_col in df.columns and well_name_column != 'None' and selected_wells:
                    well_for_ipr = st.selectbox('Select Well for IPR Analysis:', selected_wells, key="ipr_well")
                    df_ipr = df[df[well_name_column] == well_for_ipr].dropna(subset=[flow_rate_col, pwf_col])

                    if not df_ipr.empty:
                        q_test = df_ipr[flow_rate_col].mean()
                        pwf_test = df_ipr[pwf_col].mean()
                        p_res = p_res_input_ipr
                        
                        if p_res > 0 and pwf_test < p_res:
                            q_max = q_test / (1 - 0.2 * (pwf_test / p_res) - 0.8 * (pwf_test / p_res)**2)
                            Pwf_points = np.linspace(0, p_res, 100)
                            q_points = q_max * (1 - 0.2 * (Pwf_points / p_res) - 0.8 * (Pwf_points / p_res)**2)
                            
                            fig_ipr = go.Figure()
                            fig_ipr.add_trace(go.Scatter(x=q_points, y=Pwf_points, mode='lines', name='IPR Curve'))
                            fig_ipr.add_trace(go.Scatter(x=[q_test], y=[pwf_test], mode='markers', name='Test Point'))

                            fig_ipr.update_layout(
                                xaxis_title='Flow Rate (q) [bbl/day]',
                                yaxis_title='Bottom-Hole Flowing Pressure (Pwf) [psi]',
                                title=f'Inflow Performance Relationship (IPR) Curve for {well_for_ipr}'
                            )
                            st.plotly_chart(fig_ipr)
                        else:
                            st.warning("Please ensure Reservoir Pressure is greater than Pwf.")
                    else:
                        st.warning(f"Insufficient data for IPR analysis on well: {well_for_ipr}")
                else:
                    st.info("Please select 'Flow Rate', 'Pwf', and a well to see the IPR plot.")

# --- Tab 2: Water Cut & GOR Analysis ---
with main_tab2:
    st.header("Water Cut and GOR Analysis")
    st.markdown("Please upload a CSV or Excel file containing production data.")

    uploaded_file_2 = st.file_uploader("Upload your CSV or Excel file here", type=["csv", "xlsx"], key="tab2_uploader")
    if uploaded_file_2 is not None:
        df_2 = read_data(uploaded_file_2)
        if df_2 is not None:
            all_columns_2 = df_2.columns.tolist()
            date_column_2 = st.selectbox('Select Date Column:', all_columns_2, key="date_col_2")
            well_name_column_2 = st.selectbox('Select Well Name Column:', ['None'] + all_columns_2, key="well_name_col_2")
            
            if well_name_column_2 != 'None':
                selected_wells_2 = df_2[well_name_column_2].unique().tolist()
            else:
                selected_wells_2 = []

            wc_col = st.selectbox('Select Water Cut Column:', all_columns_2, key="wc_col")
            gor_col = st.selectbox('Select GOR Column:', all_columns_2, key="gor_col")

            if date_column_2 in df_2.columns and (wc_col in df_2.columns or gor_col in df_2.columns):
                st.subheader(f"Water Cut (%) vs. Time")
                fig_wc = px.line(
                    df_2,
                    x=date_column_2,
                    y=wc_col if wc_col in df_2.columns else None,
                    color=well_name_column_2 if well_name_column_2 != 'None' else None,
                    title='Water Cut Analysis Over Time'
                )
                st.plotly_chart(fig_wc)

                st.subheader(f"Gas-Oil Ratio (GOR) vs. Time")
                fig_gor = px.line(
                    df_2,
                    x=date_column_2,
                    y=gor_col if gor_col in df_2.columns else None,
                    color=well_name_column_2 if well_name_column_2 != 'None' else None,
                    title='Gas-Oil Ratio (GOR) Analysis Over Time'
                )
                st.plotly_chart(fig_gor)
            else:
                st.info("Please select Date and at least one of Water Cut or GOR columns to view the plots.")
        else:
            st.warning("Please upload a valid production data file.")
    else:
        st.info("Please upload a production data file to view the analysis.")

# --- Tab 3: Well Test Analysis ---
with main_tab3:
    st.header("Well Test Analysis (Buildup & Drawdown)")
    st.markdown("Please select the type of well test and upload the corresponding data file.")

    test_type = st.radio("Select the type of well test:",
                         options=["Buildup Test", "Drawdown Test", "Log-Log Diagnostic"],
                         key="well_test_type_selection")

    if test_type == "Buildup Test":
        st.markdown("### Buildup Test Data Upload")
        uploaded_file = st.file_uploader("Choose a file for Buildup Test Data", type=["csv", "xlsx"], key="buildup_uploader")

        if uploaded_file is not None:
            df_test = read_data(uploaded_file)
            if df_test is not None:
                if 'TIME' in df_test.columns and 'PRESSURE' in df_test.columns:
                    st.success("Buildup test data processed successfully!")
                    
                    well_test_tab1, well_test_tab2, well_test_tab3 = st.tabs(["Plot", "Diagnostic Plot", "Calculations"])

                    with well_test_tab1:
                        st.subheader("Horner Plot (Pressure Buildup)")
                        tp = st.number_input("Enter Producing Time (tp) in hours:", min_value=1.0, value=float(max(df_test['TIME'])), help="The time the well was flowing before shutdown.")
                        df_test['HORNER_TIME_RATIO'] = (tp + df_test['TIME']) / df_test['TIME']
                        df_plot = df_test[df_test['HORNER_TIME_RATIO'] > 1].copy()
                        df_plot['LOG_HORNER'] = np.log10(df_plot['HORNER_TIME_RATIO'])
                        
                        if not df_plot.empty:
                            fig_horner = go.Figure()
                            fig_horner.add_trace(go.Scatter(x=df_plot['LOG_HORNER'], y=df_plot['PRESSURE'], mode='markers', name='Pressure Data'))
                            fig_horner.update_layout(title='Horner Plot', xaxis_title="$log_{10}((t_p + \\Delta t) / \\Delta t)$", yaxis_title="Pressure (psi)", yaxis_side="right", template="plotly_white", height=600)
                            
                            mid_index = len(df_plot) // 2
                            start_index = max(0, mid_index - len(df_plot) // 4)
                            end_index = min(len(df_plot) - 1, mid_index + len(df_plot) // 4)
                            
                            subset_df = df_plot.iloc[start_index:end_index].copy()
                            
                            if not subset_df.empty:
                                try:
                                    slope, intercept, r_value, p_value, std_err = linregress(subset_df['LOG_HORNER'], subset_df['PRESSURE'])
                                    m = abs(slope)
                                    st.success(f"Calculated slope (m): {m:.2f} psi/cycle")
                                    x_fit = np.array([subset_df['LOG_HORNER'].min(), subset_df['LOG_HORNER'].max()])
                                    y_fit = slope * x_fit + intercept
                                    fig_horner.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Best-Fit Line', line=dict(color='red', width=3)))
                                except Exception as e:
                                    st.warning(f"Could not calculate slope. Please check the selected range. Error: {e}")
                            
                            min_log = df_plot['LOG_HORNER'].min()
                            max_log = df_plot['LOG_HORNER'].max()
                            
                            fig_horner.add_vline(x=min_log + 0.2 * (max_log - min_log), line_dash="dash", line_color="gray", annotation_text="ETR", annotation_position="top left")
                            fig_horner.add_vline(x=min_log + 0.8 * (max_log - min_log), line_dash="dash", line_color="gray", annotation_text="LTR", annotation_position="top right")
                            fig_horner.add_vrect(x0=min_log + 0.2 * (max_log - min_log), x1=min_log + 0.8 * (max_log - min_log), fillcolor="LightGreen", opacity=0.2, line_width=0, annotation_text="MTR", annotation_position="top right")
                            
                            st.plotly_chart(fig_horner, use_container_width=True)
                        else:
                            st.warning("Insufficient data to create a Horner Plot with a valid Horner time ratio.")

                    with well_test_tab2:
                        st.subheader("Diagnostic Log-Log Plot")
                        df_test['PRESSURE_CHANGE'] = df_test['PRESSURE'].diff().abs()
                        df_test['TIME_CHANGE'] = df_test['TIME'].diff()
                        df_test_filtered = df_test.dropna(subset=['PRESSURE_CHANGE', 'TIME_CHANGE'])
                        df_test_filtered = df_test_filtered[df_test_filtered['TIME_CHANGE'] > 0]
                        if not df_test_filtered.empty:
                            fig_diag = go.Figure()
                            fig_diag.add_trace(go.Scatter(x=df_test_filtered['TIME_CHANGE'], y=df_test_filtered['PRESSURE_CHANGE'], mode='markers'))
                            fig_diag.update_layout(title="Diagnostic Plot (Log-Log)", xaxis_title="Time Change (log scale)", yaxis_title="Pressure Change (log scale)", xaxis_type="log", yaxis_type="log", template="plotly_white", height=600)
                            
                            min_log = np.log10(df_test_filtered['TIME_CHANGE'].min()) if df_test_filtered['TIME_CHANGE'].min() > 0 else 0
                            max_log = np.log10(df_test_filtered['TIME_CHANGE'].max()) if df_test_filtered['TIME_CHANGE'].max() > 0 else 1
                            
                            fig_diag.add_vline(x=10**(min_log + 0.2 * (max_log - min_log)), line_dash="dash", line_color="gray", annotation_text="ETR", annotation_position="top left")
                            fig_diag.add_vline(x=10**(min_log + 0.8 * (max_log - min_log)), line_dash="dash", line_color="gray", annotation_text="LTR", annotation_position="top right")
                            fig_diag.add_vrect(x0=10**(min_log + 0.2 * (max_log - min_log)), x1=10**(min_log + 0.8 * (max_log - min_log)), fillcolor="LightGreen", opacity=0.2, line_width=0, annotation_text="MTR", annotation_position="top right")
                            
                            st.plotly_chart(fig_diag, use_container_width=True)
                        else:
                            st.info("Insufficient data to create a diagnostic plot.")
                    
                    with well_test_tab3:
                        st.subheader("Key Equations and Calculations")
                        st.markdown("---")
                        st.subheader("Calculate Reservoir Properties")
                        col1, col2 = st.columns(2)
                        with col1:
                            q = st.number_input("Flow Rate (q) [bbl/day]", min_value=0.0, format="%.2f", key="q_calc_b")
                            mu = st.number_input("Oil Viscosity (μ) [cp]", min_value=0.0, format="%.2f", key="mu_calc_b")
                            Bo = st.number_input("Oil FVF (Bo) [rb/stb]", min_value=0.0, format="%.2f", key="bo_calc_b")
                            m = st.number_input("Slope (m) [psi/cycle]", min_value=0.0, format="%.2f", key='m_input_calc_b')
                        with col2:
                            h = st.number_input("Net Pay Thickness (h) [ft]", min_value=0.0, format="%.2f", key="h_calc_b")
                            phi = st.number_input("Porosity (ϕ) [fraction]", min_value=0.0, max_value=1.0, format="%.2f", key="phi_calc_b")
                            ct = st.number_input("Total Compressibility (ct) [1/psi]", min_value=0.0, format="%.2e", value=1.0E-5, key="ct_calc_b")
                            rw = st.number_input("Wellbore Radius (rw) [ft]", min_value=0.0, format="%.2f", value=0.25, key="rw_calc_b")
                            P_1hr = st.number_input("P_1hr (Pressure at 1hr) [psi]", min_value=0.0, format="%.2f", key="p1hr_calc_b")
                            P_wf_s = st.number_input("P_wf,s (Final Flowing Pressure) [psi]", min_value=0.0, format="%.2f", key="pwfs_calc_b")
                        
                        if st.button("Calculate Properties", key="calc_button_b"):
                            if m > 0 and h > 0:
                                permeability = (162.6 * q * mu * Bo) / (m * h)
                                s = 1.151 * ((P_1hr - P_wf_s) / m - math.log10(permeability / (phi * mu * ct * (rw**2))) + 3.23)
                                st.success(f"**Calculated Permeability (k):** {permeability:.2f} md")
                                st.info(f"**Calculated Skin Factor (s):** {s:.2f}")
                            else:
                                st.warning("Please enter valid positive values for 'Slope' and 'Net Pay Thickness'.")
                else:
                    st.warning("Please ensure your well test file has 'TIME' and 'PRESSURE' columns.")
            else:
                st.warning("Please upload a valid data file.")


    elif test_type == "Drawdown Test":
        st.markdown("### Drawdown Test Data Upload")
        uploaded_file = st.file_uploader("Choose a file for Drawdown Test Data", type=["csv", "xlsx"], key="drawdown_uploader")

        if uploaded_file is not None:
            df_test = read_data(uploaded_file)
            if df_test is not None:
                if 'TIME' in df_test.columns and 'PRESSURE' in df_test.columns:
                    st.success("Drawdown test data processed successfully!")
                    
                    well_test_tab1, well_test_tab2, well_test_tab3 = st.tabs(["Plot", "Diagnostic Plot", "Calculations"])

                    with well_test_tab1:
                        st.subheader("Drawdown Plot")
                        df_plot_drawdown = df_test.copy()
                        df_plot_drawdown['LOG_TIME'] = np.log10(df_plot_drawdown['TIME'])
                        
                        if not df_plot_drawdown.empty:
                            fig_drawdown = go.Figure()
                            fig_drawdown.add_trace(go.Scatter(x=df_plot_drawdown['LOG_TIME'], y=df_plot_drawdown['PRESSURE'], mode='markers', name='Pressure Data'))
                            fig_drawdown.update_layout(title='Drawdown Plot', xaxis_title="$log_{10}(\\Delta t)$", yaxis_title="Pressure (psi)", yaxis_side="right", template="plotly_white", height=600)
                            
                            mid_index = len(df_plot_drawdown) // 2
                            start_index = max(0, mid_index - len(df_plot_drawdown) // 4)
                            end_index = min(len(df_plot_drawdown) - 1, mid_index + len(df_plot_drawdown) // 4)
                            
                            subset_df_drawdown = df_plot_drawdown.iloc[start_index:end_index].copy()
                            
                            if not subset_df_drawdown.empty:
                                try:
                                    slope, intercept, r_value, p_value, std_err = linregress(subset_df_drawdown['LOG_TIME'], subset_df_drawdown['PRESSURE'])
                                    m = abs(slope)
                                    st.success(f"Calculated slope (m): {m:.2f} psi/cycle")
                                    x_fit = np.array([subset_df_drawdown['LOG_TIME'].min(), subset_df_drawdown['LOG_TIME'].max()])
                                    y_fit = slope * x_fit + intercept
                                    fig_drawdown.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Best-Fit Line', line=dict(color='red', width=3)))
                                except Exception as e:
                                    st.warning(f"Could not calculate slope. Please check the selected range. Error: {e}")
                            
                            min_log = df_plot_drawdown['LOG_TIME'].min()
                            max_log = df_plot_drawdown['LOG_TIME'].max()
                            
                            fig_drawdown.add_vline(x=10**(min_log + 0.2 * (max_log - min_log)), line_dash="dash", line_color="gray", annotation_text="ETR", annotation_position="top left")
                            fig_drawdown.add_vline(x=10**(min_log + 0.8 * (max_log - min_log)), line_dash="dash", line_color="gray", annotation_text="LTR", annotation_position="top right")
                            fig_drawdown.add_vrect(x0=10**(min_log + 0.2 * (max_log - min_log)), x1=10**(min_log + 0.8 * (max_log - min_log)), fillcolor="LightGreen", opacity=0.2, line_width=0, annotation_text="MTR", annotation_position="top right")
                            
                            st.plotly_chart(fig_drawdown, use_container_width=True)
                        else:
                            st.warning("Insufficient data to create a Drawdown Plot.")
                    
                    with well_test_tab3:
                        st.subheader("Key Equations and Calculations")
                        st.markdown("---")
                        st.subheader("Calculate Reservoir Properties")
                        col1, col2 = st.columns(2)
                        with col1:
                            q = st.number_input("Flow Rate (q) [bbl/day]", min_value=0.0, format="%.2f", key="q_calc_d")
                            mu = st.number_input("Oil Viscosity (μ) [cp]", min_value=0.0, format="%.2f", key="mu_calc_d")
                            Bo = st.number_input("Oil FVF (Bo) [rb/stb]", min_value=0.0, format="%.2f", key="bo_calc_d")
                            m = st.number_input("Slope (m) [psi/cycle]", min_value=0.0, format="%.2f", key='m_input_calc_d')
                        with col2:
                            h = st.number_input("Net Pay Thickness (h) [ft]", min_value=0.0, format="%.2f", key="h_calc_d")
                            phi = st.number_input("Porosity (ϕ) [fraction]", min_value=0.0, max_value=1.0, format="%.2f", key="phi_calc_d")
                            ct = st.number_input("Total Compressibility (ct) [1/psi]", min_value=0.0, format="%.2e", value=1.0E-5, key="ct_calc_d")
                            rw = st.number_input("Wellbore Radius (rw) [ft]", min_value=0.0, format="%.2f", value=0.25, key="rw_calc_d")
                            P_initial = st.number_input("Pi (Initial Reservoir Pressure) [psi]", min_value=0.0, format="%.2f", key="pi_calc_d")
                            P_wf = st.number_input("P_wf (Final Flowing Pressure) [psi]", min_value=0.0, format="%.2f", key="pwf_calc_d")

                        if st.button("Calculate Properties", key="calc_button_d"):
                            if m > 0 and h > 0:
                                permeability = (162.6 * q * mu * Bo) / (m * h)
                                s = 1.151 * ((P_initial - P_wf) / m - math.log10(permeability / (phi * mu * ct * (rw**2))) + 3.23)
                                st.success(f"**Calculated Permeability (k):** {permeability:.2f} md")
                                st.info(f"**Calculated Skin Factor (s):** {s:.2f}")
                            else:
                                st.warning("Please enter valid positive values for 'Slope' and 'Net Pay Thickness'.")
                else:
                    st.warning("Please ensure your well test file has 'TIME' and 'PRESSURE' columns.")
            else:
                st.warning("Please upload a valid data file.")
    
    elif test_type == "Log-Log Diagnostic":
        st.markdown("### Log-Log Diagnostic Plot")
        uploaded_file = st.file_uploader("Choose a file for Well Test Data", type=["csv", "xlsx"], key="loglog_uploader")

        if uploaded_file is not None:
            df_test = read_data(uploaded_file)
            if df_test is not None:
                if 'TIME' in df_test.columns and 'PRESSURE' in df_test.columns:
                    st.success("Well test data processed successfully!")
                    df_test['PRESSURE_CHANGE'] = df_test['PRESSURE'].diff().abs()
                    df_test['TIME_CHANGE'] = df_test['TIME'].diff()
                    df_test_filtered = df_test.dropna(subset=['PRESSURE_CHANGE', 'TIME_CHANGE'])
                    df_test_filtered = df_test_filtered[df_test_filtered['TIME_CHANGE'] > 0]
                    if not df_test_filtered.empty:
                        fig_diag = go.Figure()
                        fig_diag.add_trace(go.Scatter(x=df_test_filtered['TIME_CHANGE'], y=df_test_filtered['PRESSURE_CHANGE'], mode='markers'))
                        fig_diag.update_layout(title="Diagnostic Plot (Log-Log)", xaxis_title="Time Change (log scale)", yaxis_title="Pressure Change (log scale)", xaxis_type="log", yaxis_type="log", template="plotly_white", height=600)
                        
                        min_log = np.log10(df_test_filtered['TIME_CHANGE'].min()) if df_test_filtered['TIME_CHANGE'].min() > 0 else 0
                        max_log = np.log10(df_test_filtered['TIME_CHANGE'].max()) if df_test_filtered['TIME_CHANGE'].max() > 0 else 1
                        
                        fig_diag.add_vline(x=10**(min_log + 0.2 * (max_log - min_log)), line_dash="dash", line_color="gray", annotation_text="ETR", annotation_position="top left")
                        fig_diag.add_vline(x=10**(min_log + 0.8 * (max_log - min_log)), line_dash="dash", line_color="gray", annotation_text="LTR", annotation_position="top right")
                        fig_diag.add_vrect(x0=10**(min_log + 0.2 * (max_log - min_log)), x1=10**(min_log + 0.8 * (max_log - min_log)), fillcolor="LightGreen", opacity=0.2, line_width=0, annotation_text="MTR", annotation_position="top right")
                        
                        st.plotly_chart(fig_diag, use_container_width=True)
                    else:
                        st.info("Insufficient data to create a diagnostic plot.")
                else:
                    st.warning("Please ensure your well test file has 'TIME' and 'PRESSURE' columns.")
            else:
                st.warning("Please upload a valid data file.")
