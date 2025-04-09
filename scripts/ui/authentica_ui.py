import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import os
from numpy.linalg import norm

# -- CONFIG --
st.set_page_config(page_title="Authentica", layout="wide")

ICP_COLS = [
    # your 35 ICP columns
    'AL_ICP40', 'CA_ICP40', 'FE_ICP40', 'K_ICP40', 'NA_ICP40', 'TI_ICP40', 'AG_ICP40', 'CO_ICP40', 'CE_ICP40',
    'CD_ICP40', 'BI_ICP40', 'BE_ICP40', 'LI_ICP40', 'LA_ICP40', 'MO_ICP40', 'TH_ICP40', 'SN_ICP40', 'CR_ICP40',
    'CU_ICP40', 'GA_ICP40', 'PB_ICP40', 'SC_ICP40', 'NI_ICP40', 'Y_ICP40', 'V_ICP40', 'U_ICP40', 'ZN_ICP40',
    'BA_ICP40', 'MG_ICP40', 'SR_ICP40', 'MN_ICP40', 'P_ICP40', 'AU_ICP40', 'AS_ICP40', 'NB_ICP40'
]

@st.cache_data
def load_models_and_data():
    """Load scaler, classifier, KMeans, and scaled training data."""
    scaler = joblib.load("scripts/models/scaler.pkl")
    clf = joblib.load("scripts/models/classifier.pkl")
    kmeans = joblib.load("scripts/models/kmeans.pkl")
    df_train_scaled = pd.read_csv("data/training_data_scaled.csv")
    return scaler, clf, kmeans, df_train_scaled

def get_region_icp_centroids(df_scaled):
    """Compute mean scaled ICP vector for each region."""
    region_centroids = {}
    for region_label in sorted(df_scaled['region_label'].unique()):
        region_data = df_scaled[df_scaled['region_label'] == region_label]
        mean_vec = region_data[ICP_COLS].mean().values
        region_centroids[region_label] = mean_vec
    return region_centroids

def compute_purity(sample_scaled_vec, region_centroid):
    """Distance-based purity in scaled ICP space, mapped to 0-100."""
    dist = norm(sample_scaled_vec - region_centroid)
    alpha = 10.0
    purity = 100 - alpha * dist
    return max(0, min(100, purity))  # Clamp between [0, 100]

# -- HEADER --
st.title("Authentica")
st.write("**Instant AI verification of rare earth authenticity, purity, and ethical sourcing**")

# -------------------------------------------------
# 1. Region Selection & File Uploader Side-by-Side
# -------------------------------------------------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("""
        <div style="width:350px;"background-color:#F0F9FF; padding:20px; border-radius:10px;">
            <h4 style="margin-top:0; color:#2B547E;">Select Claimed Region</h4>
    """, unsafe_allow_html=True)
    claimed_region = st.selectbox(
        "",
        [f"Region {i}" for i in range(10)]
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
        <div style="width:350px;"background-color:#F0F9FF; padding:20px; border-radius:10px;">
            <h4 style="margin-top:0;">Upload Your Data</h4>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("REE Isotope Data (CSV)", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

# Only proceed if a file is uploaded
if uploaded_file is not None:
    try:
        sample_df = pd.read_csv(uploaded_file)
        
        # Show a quick preview
        st.write("### Data Preview")
        st.dataframe(sample_df.head())

        # Load models & data
        scaler, clf, kmeans, df_train_scaled = load_models_and_data()
        region_centroids_icp = get_region_icp_centroids(df_train_scaled)

        # Scale the input sample
        sample_scaled_arr = scaler.transform(sample_df[ICP_COLS])
        sample_scaled_vec = sample_scaled_arr[0]  # single row assumption

        # Predict region from chemistry
        predicted_region = clf.predict([sample_scaled_vec])[0]
        original_region = predicted_region  # rename "predicted" to "original"

        # Compare user claim
        claimed_idx = int(claimed_region.split()[-1])
        mismatch = (claimed_idx != original_region)

        # Purity
        centroid_vec = region_centroids_icp[original_region]
        purity_score = compute_purity(sample_scaled_vec, centroid_vec)

        # Lat/lon from KMeans cluster centers
        original_lat, original_lon = kmeans.cluster_centers_[original_region]
        claimed_lat, claimed_lon = kmeans.cluster_centers_[claimed_idx]

        # ---------------------------------------------------------
        # 2. Display Results in Two Columns (Purity & Compliance)
        # ---------------------------------------------------------
        colA, colB = st.columns(2, gap="large")

        with colA:
            # Purity Card
            st.markdown(f"""
                <div style="background-color: #F0FFF0; padding:20px; border-radius:10px;">
                    <h4 style="margin-top:0; color:#2B547E;">Purity Score</h4>
                    <p style="font-size: 32px; margin:0; color:#2B547E;"><strong>{purity_score:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with colB:
            # Compliance Card
            if mismatch:
                st.markdown("""
                    <div style="background-color: #FFF2F2; padding:20px; border-radius:10px;">
                        <h4 style="margin-top:0; color:#C00000;">Compliance Status</h4>
                        <p style="font-size: 20px; margin:0;"><strong>Mismatch Detected</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background-color: #E6FFEB; padding:20px; border-radius:10px;">
                        <h4 style="margin-top:0; color:#007600;">Compliance Status</h4>
                        <p style="font-size: 20px; margin:0;"><strong>Verified</strong></p>
                    </div>
                """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 3. Additional Info Cards Side-by-Side
        # ---------------------------------------------------------
        colC, colD = st.columns(2, gap="large")

        with colC:
            st.markdown(f"""
                <div style="background-color: #FFF8E5; padding:20px; border-radius:10px; margin-top: 15px;">
                    <h4 style="margin-top:0;">User-Claimed Region</h4>
                    <p style="font-size:24px; margin:0;"><strong>Region {claimed_idx}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with colD:
            st.markdown(f"""
                <div style="background-color: #F5F5F5; padding:20px; border-radius:10px; margin-top: 15px;">
                    <h4 style="margin-top:0;">Original Region</h4>
                    <p style="font-size:24px; margin:0;"><strong>Region {original_region}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 4. Map Display
        # ---------------------------------------------------------
        st.markdown("### Map: Original vs. Claimed Region")
        folium_map = folium.Map(location=[original_lat, original_lon], zoom_start=5)

        # Marker for Original region
        folium.Marker(
            [original_lat, original_lon],
            popup=f"Original Location (Region {original_region})",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(folium_map)

        # Marker for Claimed region
        if mismatch:
            folium.Marker(
                [claimed_lat, claimed_lon],
                popup=f"User-Claimed Location (Region {claimed_idx})",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(folium_map)
        else:
            folium.Marker(
                [claimed_lat, claimed_lon],
                popup=f"Claimed == Original (Region {claimed_idx})",
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(folium_map)

        st_folium(folium_map, width=900, height=500)

    except Exception as e:
        st.error(f"An error occurred: {e}")
