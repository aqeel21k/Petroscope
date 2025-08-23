import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Set page configuration ---
st.set_page_config(page_title="Drilling Analysis", layout="wide")

st.title("Drilling Analysis Dashboard")
st.markdown("---")

# --- Section for File Uploader ---
st.header("1. Upload Your Drilling Data")
st.markdown("Please upload a CSV file containing your drilling logs.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# --- Conditional Logic to handle uploaded file or no file ---
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Standardize column names for consistency
        df.columns = [c.strip().replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '').upper() for c in df.columns]

        # --- Time vs Depth Curve ---
        st.markdown("---")
        st.header("2. Time vs Depth Curve")
        st.markdown("This curve shows the progress of drilling over time, which helps in evaluating drilling efficiency and identifying any delays.")

        if 'TIME' in df.columns and 'DEPTH' in df.columns:
            fig_time_depth = go.Figure()
            fig_time_depth.add_trace(go.Scatter(x=df['TIME'], y=df['DEPTH'], mode='lines+markers', name='Time vs Depth', line=dict(color='orange')))
            fig_time_depth.update_layout(title_text='Time vs Depth',
                                         xaxis_title="Time (hours)",
                                         yaxis_title="Depth (ft)",
                                         yaxis_autorange='reversed')
            st.plotly_chart(fig_time_depth, use_container_width=True)
        else:
            st.warning("Data must contain 'TIME' and 'DEPTH' columns for this plot.")

        # --- Mud Weight vs Depth Plot ---
        st.markdown("---")
        st.header("3. Mud Weight vs Depth Plot")
        st.markdown("This plot compares mud weight with depth to monitor downhole pressures.")

        if 'MUD_WEIGHT' in df.columns and 'DEPTH' in df.columns:
            fig_mud_weight = go.Figure()
            fig_mud_weight.add_trace(go.Scatter(x=df['MUD_WEIGHT'], y=df['DEPTH'], mode='lines', name='Mud Weight', line=dict(color='brown', width=2)))

            fig_mud_weight.update_layout(
                title_text='Mud Weight vs Depth',
                xaxis_title="Mud Weight (ppg)",
                yaxis_title="Depth (ft)",
                yaxis_autorange='reversed',
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
            st.plotly_chart(fig_mud_weight, use_container_width=True)
        else:
            st.warning("Data must contain 'MUD_WEIGHT' and 'DEPTH' columns for this plot.")

        # --- Bit Performance Analysis ---
        st.markdown("---")
        st.header("4. Bit Performance Analysis")
        st.markdown("This plot shows the relationship between ROP, WOB, and RPM to evaluate drilling bit efficiency.")
        
        if 'ROP' in df.columns and 'WOB' in df.columns and 'RPM' in df.columns:
            fig_bit_performance = go.Figure()
            fig_bit_performance.add_trace(go.Scatter(x=df['WOB'], y=df['ROP'], mode='markers', name='ROP vs WOB', marker=dict(size=8, color='blue')))
            fig_bit_performance.add_trace(go.Scatter(x=df['WOB'], y=df['RPM'], mode='markers', name='RPM vs WOB', marker=dict(size=8, color='green')))

            fig_bit_performance.update_layout(title='Bit Performance: ROP & RPM vs WOB',
                                               xaxis_title='WOB (Weight on Bit)',
                                               yaxis_title='Value',
                                               hovermode='x unified')
            st.plotly_chart(fig_bit_performance, use_container_width=True)
        else:
            st.warning("Data must contain 'ROP', 'WOB', and 'RPM' columns for this analysis.")

        # --- Drilling Logs Plot ---
        st.markdown("---")
        st.header("5. Drilling Logs Plot")
        st.markdown("This plot shows the main drilling logs versus depth.")

        if all(col in df.columns for col in ['ROP', 'WOB', 'TORQUE', 'RPM', 'DEPTH']):
            fig_drilling_logs = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.02,
                                              subplot_titles=('ROP', 'WOB', 'TORQUE', 'RPM'))
            
            fig_drilling_logs.add_trace(go.Scatter(x=df['ROP'], y=df['DEPTH'], name='ROP', mode='lines', line=dict(color='blue')), row=1, col=1)
            fig_drilling_logs.add_trace(go.Scatter(x=df['WOB'], y=df['DEPTH'], name='WOB', mode='lines', line=dict(color='green')), row=1, col=2)
            fig_drilling_logs.add_trace(go.Scatter(x=df['TORQUE'], y=df['DEPTH'], name='TORQUE', mode='lines', line=dict(color='purple')), row=1, col=3)
            fig_drilling_logs.add_trace(go.Scatter(x=df['RPM'], y=df['DEPTH'], name='RPM', mode='lines', line=dict(color='red')), row=1, col=4)
            
            fig_drilling_logs.update_layout(height=800, width=1200, title_text="Drilling Logs", yaxis_autorange='reversed')
            fig_drilling_logs.update_yaxes(title_text="Depth (ft)", col=1)
            
            st.plotly_chart(fig_drilling_logs, use_container_width=True)
        else:
            st.warning("Data must contain 'ROP', 'WOB', 'TORQUE', 'RPM', and 'DEPTH' columns for this plot.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Please upload a CSV file to view the analysis.")

