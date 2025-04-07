from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import folium

# Initialize the FastAPI app
app = FastAPI(title="Authentica Project: Sample Region Predictor")

# Load the pre-trained models
scaler = joblib.load("scripts/models/scaler.pkl")
clf = joblib.load("scripts/models/classifier.pkl")
kmeans = joblib.load("scripts/models/kmeans.pkl")

# Define the expected input format for a sample (35 ICP40 features)
class SampleData(BaseModel):
    AL_ICP40: float
    CA_ICP40: float
    FE_ICP40: float
    K_ICP40: float
    NA_ICP40: float
    TI_ICP40: float
    AG_ICP40: float
    CO_ICP40: float
    CE_ICP40: float
    CD_ICP40: float
    BI_ICP40: float
    BE_ICP40: float
    LI_ICP40: float
    LA_ICP40: float
    MO_ICP40: float
    TH_ICP40: float
    SN_ICP40: float
    CR_ICP40: float
    CU_ICP40: float
    GA_ICP40: float
    PB_ICP40: float
    SC_ICP40: float
    NI_ICP40: float
    Y_ICP40: float
    V_ICP40: float
    U_ICP40: float
    ZN_ICP40: float
    BA_ICP40: float
    MG_ICP40: float
    SR_ICP40: float
    MN_ICP40: float
    P_ICP40: float
    AU_ICP40: float
    AS_ICP40: float
    NB_ICP40: float

@app.post("/predict")
def predict(sample: SampleData):
    # Convert the input to a DataFrame
    sample_df = pd.DataFrame([sample.dict()])
    
    # Scale the input features using the loaded scaler
    sample_scaled = scaler.transform(sample_df)
    
    # Predict the region label using the classifier
    region_pred = clf.predict(sample_scaled)[0]
    
    # Retrieve the cluster centroid from the KMeans model
    # Since kmeans was trained on [LATITUDE, LONGITUDE],
    # the centroid order is [latitude, longitude]
    centroid = kmeans.cluster_centers_[region_pred]
    centroid_lat = centroid[0]
    centroid_lon = centroid[1]
    
    # Create a folium map centered on the centroid
    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=10)
    folium.Marker(
        location=[centroid_lat, centroid_lon],
        tooltip=f"Predicted Region: {region_pred}"
    ).add_to(m)
    
    # Save the map to an HTML file in the models folder (you can adjust the path)
    map_filename = f"scripts/models/map_region_{region_pred}.html"
    m.save(map_filename)
    
    # Return the prediction details as JSON
    return {
        "predicted_region": int(region_pred),
        "centroid_latitude": centroid_lat,
        "centroid_longitude": centroid_lon,
        "map_file": map_filename
    }
