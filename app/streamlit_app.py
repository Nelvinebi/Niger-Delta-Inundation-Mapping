import streamlit as st
import rasterio
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="NDIMS", layout="wide")

tab1, tab2 = st.tabs(["Phase 1: Inundation Map", "Phase 2: Temporal Change"])

with tab1:
    st.header("Current Inundation Mapping")
    st.info("Maps surface water extent using SAR backscatter, observed rainfall, and elevation.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with rasterio.open("data/raw/features_stacked.tif") as src:
            sar = src.read(1)
        fig, ax = plt.subplots()
        ax.imshow(sar[::5, ::5], cmap='gray', vmin=-25, vmax=0)
        ax.set_title("SAR Backscatter (VV)")
        st.pyplot(fig)
    
    with col2:
        with rasterio.open("data/raw/features_stacked.tif") as src:
            rain = src.read(2)
        fig, ax = plt.subplots()
        ax.imshow(rain[::5, ::5], cmap='Blues')
        ax.set_title("CHIRPS Rainfall (mm)")
        st.pyplot(fig)
    
    with col3:
        with rasterio.open("data/raw/features_stacked.tif") as src:
            elev = src.read(3)
        fig, ax = plt.subplots()
        ax.imshow(elev[::5, ::5], cmap='terrain')
        ax.set_title("SRTM Elevation (m)")
        st.pyplot(fig)
    
    st.subheader("Model Prediction")
    with rasterio.open("outputs/figures/inundation_map.tif") as src:
        pred = src.read(1)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(pred[::5, ::5], cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_title("Predicted Inundation")
    st.pyplot(fig)
    
    st.markdown("""
    **Model:** XGBoost trained on 2M samples | **AUC:** 0.9951
    **Features:** Sentinel-1 SAR, CHIRPS rainfall, SRTM elevation
    **Note:** Maps current water extent, not future flood risk.
    """)

with tab2:
    st.header("Temporal Change Detection")
    st.info("Compares dry season vs. flood period to identify true flood events.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with rasterio.open("data/raw/sar_dry_2022.tif") as src:
            dry = src.read(1)
        fig, ax = plt.subplots()
        ax.imshow(dry[::5, ::5], cmap='gray', vmin=-25, vmax=0)
        ax.set_title("Dry Season (Jan 2022)")
        st.pyplot(fig)
    
    with col2:
        with rasterio.open("data/raw/sar_flood_2022.tif") as src:
            flood = src.read(1)
        fig, ax = plt.subplots()
        ax.imshow(flood[::5, ::5], cmap='gray', vmin=-25, vmax=0)
        ax.set_title("Flood Period (Sep-Oct 2022)")
        st.pyplot(fig)
    
    st.subheader("True Flood Detection")
    with rasterio.open("data/raw/flood_change_2022.tif") as src:
        change = src.read(1)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(change[::5, ::5], cmap='Reds', vmin=0, vmax=1)
    ax.set_title("Areas That Became Wet During Flood")
    st.pyplot(fig)
    
    st.markdown("""
    **Method:** Dry season water subtracted from flood period water.
    Red areas = true flood events. Gray areas = permanent water.
    **Model:** AUC = 0.9857 using pre-flood SAR, elevation, and observed rainfall.
    """)

st.sidebar.header("About NDIMS")
st.sidebar.markdown("""
**Niger Delta Inundation Mapping System**

- **Phase 1:** Current water extent mapping
- **Phase 2:** Temporal change detection (retrospective)
- **Phase 3:** Early warning — requires NWP forecast integration (6-8 weeks)
""")