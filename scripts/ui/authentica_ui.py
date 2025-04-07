import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import os

# -- Title & Tagline --
st.title("Authentica")
st.subheader("Instant AI verification of rare earth authenticity, purity, and ethical sourcing")
st.markdown("### Upload REE isotope data to know sample origin")

# -- File Upload --
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        sample = pd.read_csv(uploaded_file)
        st.write("### Data Preview", sample.head())

        # Load the saved models
        scaler = joblib.load("scripts/models/scaler.pkl")
        clf = joblib.load("scripts/models/classifier.pkl")
        kmeans = joblib.load("scripts/models/kmeans.pkl")
        
        # Preprocess sample (assumes CSV has the required 35 ICP40 columns)
        sample_scaled = scaler.transform(sample)
        predicted_region = clf.predict(sample_scaled)[0]
        
        # Get predicted cluster centroid from KMeans (order: [LATITUDE, LONGITUDE])
        centroid = kmeans.cluster_centers_[predicted_region]
        centroid_lat = centroid[0]
        centroid_lon = centroid[1]
        
        st.success(f"Predicted Region: {predicted_region} at ({centroid_lat:.4f}, {centroid_lon:.4f})")
        
        # Create a clean Folium map centered on the predicted location
        m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=7)
        
        # Mark the predicted region centroid on the map
        folium.Marker(
            location=[centroid_lat, centroid_lon],
            popup=f"Predicted Region: {predicted_region}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        st.markdown("### Predicted Sample Location on Map")
        st_folium(m, width=700, height=500)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
