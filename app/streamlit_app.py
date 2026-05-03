import streamlit as st
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.set_page_config(page_title="NDIMS", layout="wide")

# ── Resolve paths relative to this script so the app works
#    both locally and on Streamlit Cloud regardless of working directory ──────
APP_DIR  = Path(__file__).parent          # .../app/
ROOT_DIR = APP_DIR.parent                 # repo root

RAW      = ROOT_DIR / "data" / "raw"
OUTPUTS  = ROOT_DIR / "outputs" / "figures"


# ── Cached raster loader — reads each file once per session ─────────────────
@st.cache_resource
def load_band(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band)


def show_raster(path: Path, band: int, cmap: str, title: str, **imshow_kw):
    """Render a raster band inside whatever column/container is active."""
    if not path.exists():
        st.warning(f"File not found: `{path.relative_to(ROOT_DIR)}`")
        return
    arr = load_band(path, band)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(arr[::5, ::5], cmap=cmap, **imshow_kw)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Phase 1: Inundation Map", "Phase 2: Temporal Change"])

# ── PHASE 1 ─────────────────────────────────────────────────────────────────
with tab1:
    st.header("Current Inundation Mapping")
    st.info(
        "Maps surface water extent using Sentinel-1 SAR backscatter, "
        "CHIRPS observed rainfall, and SRTM elevation."
    )

    col1, col2, col3 = st.columns(3)
    stacked = RAW / "features_stacked.tif"

    with col1:
        show_raster(stacked, 1, "gray",    "SAR Backscatter (VV)",  vmin=-25, vmax=0)
    with col2:
        show_raster(stacked, 2, "Blues",   "CHIRPS Rainfall (mm)")
    with col3:
        show_raster(stacked, 3, "terrain", "SRTM Elevation (m)")

    st.subheader("Model Prediction")
    show_raster(
        OUTPUTS / "inundation_map.tif", 1,
        "RdYlBu_r", "Predicted Inundation Probability",
        vmin=0, vmax=1,
    )

    st.markdown(
        "**Model:** XGBoost trained on 2 M samples &nbsp;|&nbsp; **AUC:** 0.9951  \n"
        "**Features:** Sentinel-1 SAR · CHIRPS rainfall · SRTM elevation  \n"
        "**Note:** Maps current water extent — not future flood risk."
    )

# ── PHASE 2 ─────────────────────────────────────────────────────────────────
with tab2:
    st.header("Temporal Change Detection")
    st.info(
        "Compares dry season vs. flood period SAR to isolate "
        "true flood events from permanent water bodies."
    )

    col1, col2 = st.columns(2)

    with col1:
        show_raster(
            RAW / "sar_dry_2022.tif", 1,
            "gray", "Dry Season (Jan 2022)",
            vmin=-25, vmax=0,
        )
    with col2:
        show_raster(
            RAW / "sar_flood_2022.tif", 1,
            "gray", "Flood Period (Sep–Oct 2022)",
            vmin=-25, vmax=0,
        )

    st.subheader("True Flood Detection")
    show_raster(
        RAW / "flood_change_2022.tif", 1,
        "Reds", "Areas That Became Wet During the Flood Event",
        vmin=0, vmax=1,
    )

    st.markdown(
        "**Method:** Dry-season water mask subtracted from flood-period water mask.  \n"
        "Red = true flood events &nbsp;|&nbsp; Grey = permanent water bodies  \n"
        "**Model:** AUC 0.9857 — features: pre-flood SAR change magnitude, elevation, rainfall."
    )

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("About NDIMS")
st.sidebar.markdown(
    """
**Niger Delta Inundation Mapping System**

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Current water extent mapping | ✅ Live |
| **2** | Temporal change detection | ⚙️ In progress |
| **3** | Early warning (NWP forecast) | 🗓️ Planned |
"""
)
