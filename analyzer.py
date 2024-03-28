import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from tensorflow.keras.models import load_model

raw_data_dir = 'material_for_analysis'
results_dir = 'analysis_results'
model_path = 'mouse_movement_anomaly_detection_model.keras'
scaler_path = 'universal_scaler.joblib'

os.makedirs(results_dir, exist_ok=True)

model = load_model(model_path)
scaler = load(scaler_path)

sequence_length = 30
n_features = 9

def calculate_features(data):
    data['Time_Delta'] = data['Time'].diff().bfill()

    data['Velocity_X'] = data['X'].diff().fillna(0) / data['Time_Delta']
    data['Velocity_Y'] = data['Y'].diff().fillna(0) / data['Time_Delta']

    data['Acceleration_X'] = data['Velocity_X'].diff().fillna(0) / data['Time_Delta']
    data['Acceleration_Y'] = data['Velocity_Y'].diff().fillna(0) / data['Time_Delta']

    data['Angle'] = np.arctan2(data['Velocity_Y'], data['Velocity_X'])
    data['Angular_Change'] = data['Angle'].diff().bfill()

    data['Distance'] = np.sqrt(data['X'].diff().fillna(0) ** 2 + data['Y'].diff().fillna(0) ** 2)

    data = data.drop(['Time', 'Angle'], axis=1)

    return data

def analyze_data(file_path):
    data = pd.read_csv(file_path)
    features = calculate_features(data)
    normalized_features = scaler.transform(features)

    rows_to_use = (normalized_features.shape[0] // sequence_length) * sequence_length
    normalized_features = normalized_features[:rows_to_use]

    sequences = normalized_features.reshape(-1, sequence_length, n_features)

    reconstructed = model.predict(sequences)

    mse = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
    return mse

for filename in os.listdir(raw_data_dir):
    if filename.endswith('.csv'):
        try:
            print(f"Analyzing {filename}...")
            file_path = os.path.join(raw_data_dir, filename)

            reconstruction_errors = analyze_data(file_path)

            result_file_path = os.path.join(results_dir, f"analysis_{filename}")
            pd.DataFrame(reconstruction_errors, columns=['Reconstruction_Error']).to_csv(result_file_path, index=False)
            print(f"Analysis results saved to {result_file_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Analysis complete.")