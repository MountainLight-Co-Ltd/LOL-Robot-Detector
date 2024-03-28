import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib  # For saving the model

# Path to the folder containing the CSV files
folder_path = 'analysis_results'

# Initialize an empty DataFrame to hold all reconstruction errors
combined_data = pd.DataFrame()

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Load the current CSV file
        current_data = pd.read_csv(file_path)
        # Combine the current data with the combined dataset
        combined_data = pd.concat([combined_data, current_data], ignore_index=True)

# Assuming the CSV files contain a single column with reconstruction errors
X = combined_data.values  # Convert DataFrame to numpy array

# Initialize the Isolation Forest model
# Adjust the 'contamination' parameter as needed based on your understanding of the data
isolation_forest = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)

# Train the Isolation Forest model on the combined reconstruction error data
isolation_forest.fit(X)

# Save the trained Isolation Forest model
model_filename = 'anomaly_detection_isolation_forest.joblib'
joblib.dump(isolation_forest, model_filename)

print("The Isolation Forest model has been trained and saved as", model_filename)