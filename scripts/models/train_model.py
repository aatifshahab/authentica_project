import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import joblib

# -------------------------------
# 1. Load Data
# -------------------------------
# Adjust the paths if necessary
df_features = pd.read_csv(r"data/output_filename.csv")         # Contains LABNO and ICP40 columns
df_locations = pd.read_csv(r"data/ngs_shapefile.csv")           # Contains LABNO and geometry

# -------------------------------
# 2. Merge Data on LABNO
# -------------------------------
df_merged = pd.merge(df_features, df_locations[['LABNO', 'geometry']], on='LABNO')

# -------------------------------
# 3. Extract Latitude and Longitude
# -------------------------------
# Assumes geometry is like: "POINT (longitude latitude)"
df_merged[['LONGITUDE', 'LATITUDE']] = (
    df_merged['geometry']
    .str.extract(r'POINT \((-?\d+\.\d+)\s+(-?\d+\.\d+)\)')
    .astype(float)
)

# -------------------------------
# 4. Select ICP40 Features and Clean Data
# -------------------------------
selected_icp40 = [
    'AL_ICP40', 'CA_ICP40', 'FE_ICP40', 'K_ICP40', 'NA_ICP40', 'TI_ICP40', 'AG_ICP40', 'CO_ICP40', 'CE_ICP40',
    'CD_ICP40', 'BI_ICP40', 'BE_ICP40', 'LI_ICP40', 'LA_ICP40', 'MO_ICP40', 'TH_ICP40', 'SN_ICP40', 'CR_ICP40',
    'CU_ICP40', 'GA_ICP40', 'PB_ICP40', 'SC_ICP40', 'NI_ICP40', 'Y_ICP40', 'V_ICP40', 'U_ICP40', 'ZN_ICP40',
    'BA_ICP40', 'MG_ICP40', 'SR_ICP40', 'MN_ICP40', 'P_ICP40', 'AU_ICP40', 'AS_ICP40', 'NB_ICP40'
]

# Drop rows with missing values in ICP40 columns or in lat/lon
df_model = df_merged.dropna(subset=selected_icp40 + ['LATITUDE', 'LONGITUDE'])

# -------------------------------
# 5. Unsupervised Clustering on Location
# -------------------------------
n_clusters = 10  # For MVP, you can adjust this between 10-15 as needed
# IMPORTANT: kmeans is trained on [LATITUDE, LONGITUDE] (order matters!)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_model['region_label'] = kmeans.fit_predict(df_model[['LATITUDE', 'LONGITUDE']])

# -------------------------------
# 6. Train Supervised Classifier (RandomForest)
# -------------------------------
X = df_model[selected_icp40]
y = df_model['region_label']

# Scale the chemical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Optional: Evaluate the model (print classification report and confusion matrix)
from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. Save the Models
# -------------------------------
joblib.dump(scaler, "scripts/models/scaler.pkl")
joblib.dump(clf, "scripts/models/classifier.pkl")
joblib.dump(kmeans, "scripts/models/kmeans.pkl")

df_model.to_csv("data/training_data_with_clusters.csv", index=False)

print("Models saved successfully in 'scripts/models/'")
