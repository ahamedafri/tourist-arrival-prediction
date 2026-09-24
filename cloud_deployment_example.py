import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tensorflow.keras.models import load_model, Model
import warnings
import io
import os

# Import train_lstm functions
from train_lstm import merge_csv_files, prepare_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# This helps Streamlit Cloud find your data files
def get_project_root():
    """Get the path to the project root folder"""
    return os.path.dirname(os.path.abspath(__file__))

# Modified file loading function
def load_model_and_data():
    """Load model and data with proper paths for Streamlit Cloud"""
    root_dir = get_project_root()
    
    # Load model
    model_path = os.path.join(root_dir, 'tourism_lstm_model.h5')
    model = load_model(model_path)
    
    # For data loading - adjust merge_csv_files if needed to use root_dir
    # This depends on how your merge_csv_files function works
    merged_data = merge_csv_files()
    
    return model, merged_data

# Example of how to adjust your main function for Streamlit Cloud
def main():
    # Set page config with proper metadata
    st.set_page_config(
        page_title="Tourism Analytics Dashboard",
        page_icon="🏝️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS the same way as your current app
    st.markdown("""
    <style>
    /* Your existing CSS here */
    </style>
    """, unsafe_allow_html=True)

    # Continue with your app logic...
    
# This part remains unchanged
if __name__ == "__main__":
    main()