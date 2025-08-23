import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import requests
from streamlit_lottie import st_lottie
import io
import os 

# --- Set page configuration ---
st.set_page_config(
    page_title="Petroscope - Your Petroleum Analysis Hub",
    layout="wide"
)

# --- Function to load Lottie animation from URL ---
def load_lottieurl(url: str):
    """Loads a Lottie animation from a given URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading animation: {e}")
        return None

# --- Custom CSS for a professional look ---
st.markdown(
    """
    <style>
    /* General body and font styling */
    body {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #f7f9fc;
    }

    /* Header and Navigation Bar */
    .header-container {
        display: flex;
        align-items: center;
        padding: 10px 0;
    }
    .header-container img {
        margin-right: 20px;
    }
    
    /* Main content styling */
    .st-emotion-cache-1av5q9c {
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0c4a6e;
    }
    p, li, .stMarkdown {
        color: #333333;
    }
    .st-emotion-cache-1wb0f50 {
        font-size: 1.2rem;
        color: #333;
        line-height: 1.6;
    }
    
    /* Section dividers */
    .st-emotion-cache-121gjc6 {
        border-bottom: 2px solid #ddd;
        margin: 2rem 0;
    }

    /* Buttons and widgets */
    .stButton>button {
        background-color: #0c4a6e;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #0c4a6e;
        color: #f0f0f0;
    }
    
    /* Tabs styling to make it look like a navigation bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #333;
        font-weight: bold;
        padding: 10px 15px;
        border-radius: 5px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        color: #0c4a6e;
        border-bottom: 2px solid #0c4a6e;
    }
    
    /* Unique Footer Styling - Corrected for centering and visibility */
    .footer-container {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #4b5563;
        text-align: center;
        padding: 1rem;
        font-size: 0.875rem;
        border-top: 1px solid #e5e7eb;
        z-index: 1000;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
    }
    .footer-text {
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .by-text {
        font-weight: normal;
        margin-right: 5px; /* Adds space before the name */
    }

    /* CSS for form styling */
    form {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    form input, form textarea {
        width: 100%;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        font-size: 1em;
        box-sizing: border-box;
    }
    form button {
        background-color: #0c4a6e;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1.1em;
        align-self: flex-start;
    }
    
    /* FIX FOR LOGO RESOLUTION */
    .logo-img {
        max-width: 250px; /* or a suitable size */
        height: auto;
        margin-right: 20px;
    }
    /* NEW CSS for reduced spacing */
    .file-item-container {
        margin-bottom: 15px;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .file-name {
        font-weight: bold;
        color: #0c4a6e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header with Logo and Title ---
logo_col, title_col = st.columns([1, 5])
with logo_col:
    # We'll use the uploaded image file name directly.
    # The code now reads the image file as a binary stream.
    logo_path = "logo2.png"
    if os.path.exists(logo_path):
        logo_image = Image.open(logo_path)
        st.image(logo_image, width=250)
    else:
        st.error(f"Error: The logo file '{logo_path}' was not found.")
with title_col:
    st.markdown(
        """
        <div style='display:flex; align-items:center; height:100%;'>
            <h1>Petroscope</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Tabs ---
home_tab, about_tab, offer_tab, training_tab, contact_tab = st.tabs(["Home", "About Us", "What We Offer", "Training Data", "Contact Us"])

with home_tab:
    st.image("Banner1.png", use_container_width=True)
        
    st.header('Welcome to the Oil Field Dashboard')
    st.markdown(
      """
        This app is your comprehensive tool for analyzing **oil well data**. It's designed to help you gain deeper insights and make better decisions.
        
        ### What this app offers:
        * **Production Page:** Analyze oil and gas rates, perform Decline Curve Analysis (DCA) for forecasting, and build Inflow Performance Relationship (IPR) curves.
        * **Wireline Page:** Visualize and analyze different wireline logs to understand reservoir characteristics.
        * **Geospatial Page:** Upload well coordinates to see their locations on an interactive map.
        * **Drilling Page:** Analyze key drilling parameters like Rate of Penetration (ROP), Weight on Bit (WOB), and Torque.
        * **Pipeline Analysis:** Analyze flow assurance and hydraulic performance.
        * **Reservoir 3D:** Visualize and interact with 3D reservoir models.

        You can upload your own CSV files and use the sidebar filters to customize your view.
        """
    )

with about_tab:
    st.header("About Us")
    st.markdown(
        """
        **Petroscope** is developed by a dedicated petroleum engineer and data scientist. Our mission is to democratize data analytics in the oil and gas industry by providing accessible, powerful, and easy-to-use tools. We believe that leveraging data can drive efficiency, reduce costs, and optimize production from the wellbore to the reservoir.
        """
    )
    
    st.subheader("Our Vision & Mission")
    st.markdown(
        """
        Our vision is to empower oil and gas professionals with data-driven insights. We aim to be a leader in providing intuitive, robust, and affordable analytical solutions that transform how reservoir and production data is managed and interpreted. We are committed to fostering innovation and collaboration within the industry.
        """
    )
    
    st.subheader("Meet the Creator")
    st.markdown(
        """
        I am a petroleum engineer with a strong passion for data science and a commitment to building practical, effective tools for the energy sector. This application is a result of my belief in the power of data to solve real-world industry challenges.
        """
    )
    
with offer_tab:
    st.header("What We Offer")
    st.markdown(
        """
        Our dashboard provides a comprehensive suite of professional tools to streamline your daily workflows and unlock the full potential of your data. Each tool is meticulously designed to be intuitive and powerful, allowing you to focus on gaining actionable insights rather than tedious data processing.
        
        ### Key Services:
        """
    )
    # Placeholder for the Lottie animation
    st.image("https://placehold.co/800x300/F0F2F6/333?text=Analytics+Animation+Placeholder", use_container_width=True)
    st.markdown(
        """
        * **Production Analytics:** From decline curve analysis (DCA) to inflow performance relationships (IPR), our tools help you forecast production and optimize well performance.
            * **Water Cut & GOR Analysis:** Analyze water and gas breakthrough to manage production.
            * **Well Test Analysis:** Interpret well tests to determine reservoir properties.
            * **Material Balance Analysis:** Analyze reservoir performance and estimate original oil/gas in place.
        * **Reservoir Characterization:** Visualize and interpret wireline logs, core data, and seismic information to build a comprehensive understanding of your reservoir.
        * **Geospatial Visualization:** Map well locations and production data to gain a spatial perspective on your field's performance.
        * **Drilling Optimization:** Analyze drilling parameters in real-time to enhance drilling efficiency and reduce operational costs.
        * **Pipeline Analysis:** Analyze flow assurance and hydraulic performance in pipelines.
        * **Reservoir Simulation & Data Analysis:**
            * **3D Reservoir Model:** Visualize and interact with 3D reservoir models.
            * **PVT Data Analysis:** Analyze fluid properties to understand reservoir fluids.
            * **Core Data Analysis:** Interpret core sample data for rock properties.
        """
    )


with training_tab:
    st.header("Training Data")
    st.markdown("This section provides sample datasets for analysis and training purposes. You can download them directly to get started.")
    
    st.subheader("Available Files for Download")
    
    downloadable_files = [
        {"name": "3D reservoir data.csv", "path": "training_data/3D reservoir data.csv"},
        {"name": "buildup_test_data.csv", "path": "training_data/buildup_test_data.csv"},
        {"name": "core_data.csv", "path": "training_data/core_data.csv"},
        {"name": "geospatial_data (2).csv", "path": "training_data/geospatial_data (2).csv"},
        {"name": "ipr_test_data.csv", "path": "training_data/ipr_test_data.csv"},
        {"name": "Log_Data.csv", "path": "training_data/Log_Data.csv"},
        {"name": "production_data.csv", "path": "training_data/production_data.csv"},
        {"name": "pvt_data.csv", "path": "training_data/pvt_data.csv"},
        {"name": "reservoir_data_for_3d_model.csv", "path": "training_data/reservoir_data_for_3d_model.csv"},
        {"name": "Sample_Well_Data.csv", "path": "training_data/Sample_Well_Data.csv"},
        {"name": "simulation_data_generated (3).csv", "path": "training_data/simulation_data_generated (3).csv"},
        {"name": "water cut and GOR.csv", "path": "training_data/water cut and GOR.csv"},
        {"name": "New_Log_1.las", "path": "training_data/New_Log_1.las"},
        {"name": "New_Log_2.las", "path": "training_data/New_Log_2.las"}
    ]

    # New method to handle downloads by reading the file and providing its content directly
    for file_info in downloadable_files:
        file_name = file_info["name"]
        file_path = file_info["path"]
        
        # Check if the file exists before trying to read it
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            # Use a custom container for each file item to reduce spacing
            st.markdown(f"<div class='file-item-container'>", unsafe_allow_html=True)
            st.markdown(f"<p class='file-name'>{file_name}</p>", unsafe_allow_html=True)
            st.download_button(
                label="Download",
                data=file_content,
                file_name=file_name,
                mime="text/csv" if ".csv" in file_name else "text/plain"
            )
            st.markdown(f"</div>", unsafe_allow_html=True)
        else:
            # Use a similar container for the warning message
            st.markdown(f"<div class='file-item-container'>", unsafe_allow_html=True)
            st.markdown(f"<p class='file-name'>{file_name}</p>", unsafe_allow_html=True)
            st.warning(f"Error: The file '{file_name}' was not found in the specified path.")
            st.markdown(f"</div>", unsafe_allow_html=True)


with contact_tab:
    st.header("Contact Us")
    st.markdown("We would love to hear from you! Please fill out the form below.")

    with st.form("contact_form"):
        name = st.text_input("Your name")
        email = st.text_input("Your email")
        message = st.text_area("Your message here", height=150)
        
        submitted = st.form_submit_button("Send")
        if submitted:
            st.success("Thank you for your message! We will get back to you shortly.")

    st.markdown("---")
    st.markdown("### Our Location")
    st.markdown("Email: aqeel.engpm@gmail.com")
    st.markdown("Phone: +964 7770830306")
    st.markdown("Address: Maysan, Iraq")

# --- Unique Footer at the bottom of the page ---
def display_footer():
    """
    Displays a styled footer at the bottom of the page with correct spacing.
    """
    st.markdown("""
    <style>
    .footer-container {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #4b5563;
        text-align: center;
        padding: 1rem;
        font-size: 0.875rem;
        border-top: 1px solid #e5e7eb;
        z-index: 1000;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
    }
    .footer-text {
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .by-text {
        font-weight: normal;
        margin-right: 5px; /* Adds space before the name */
    }
    </style>
    <div class='footer-container'>
        <p class='footer-text'>
            Petroscope © 2025. All rights reserved. | <span class="by-text">Developed by</span>
            <span style="font-weight: bold;">
                Aqeel Kareem
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Call the function to display the footer
display_footer()
