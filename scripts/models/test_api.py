import pandas as pd
import requests

# Load your dummy sample CSV (ensure it has the required 35 columns exactly as defined in the SampleData model)
sample = pd.read_csv("dummy_sample.csv")

# Assuming the CSV has one sample row, convert it to a dictionary.
# If there are multiple rows, you might want to loop through them.
sample_data = sample.to_dict(orient="records")[0]

# URL of your FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Send a POST request to the endpoint
response = requests.post(url, json=sample_data)

# Print out the response from the API
print("API Response:")
print(response.json())
