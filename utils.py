# utils.py
import streamlit as st
import pandas as pd

def validate_and_read_data(uploaded_file, required_columns):
    """
    Validates and reads a data file, checking for required columns.

    Args:
        uploaded_file: The file uploaded by st.file_uploader.
        required_columns (list): A list of required column names (case-insensitive).

    Returns:
        pd.DataFrame or None: The DataFrame if validation is successful, otherwise None.
    """
    if uploaded_file is None:
        return None # Return None and wait for a file upload

    try:
        # Read the file based on its extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return None

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        st.info("Please make sure the file format is correct.")
        return None

    # Convert required column names to lowercase for flexible checking
    required_columns_lower = [col.lower() for col in required_columns]
    
    # Get the columns from the DataFrame and convert them to lowercase
    df_columns_lower = [col.lower() for col in df.columns]
    
    # Check if all required columns are present
    if all(col in df_columns_lower for col in required_columns_lower):
        st.success("Required columns found successfully!")
        
        # Map original column names to the lowercase required names
        col_map = {req_col: df.columns[df_columns_lower.index(req_col)] for req_col in required_columns_lower}
        
        # Add a new attribute to the DataFrame for easy access to mapped column names
        df.col_map = col_map
        return df
    else:
        # Identify and list the missing columns for the user
        missing_columns = [col for col in required_columns_lower if col not in df_columns_lower]
        st.error(f"Error: The file does not contain the required columns.")
        st.info(f"Missing columns are: {', '.join(missing_columns)}")
        st.warning("Please check your file, rename the columns, or use a different file.")
        return None
