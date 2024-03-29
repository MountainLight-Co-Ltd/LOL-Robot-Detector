import os
import pandas as pd
import numpy as np
from joblib import load

source_dir = 'mouse_positions'
target_dir = 'ready_for_training'
universal_scaler_filename = 'universal_scaler.joblib'
sequence_length = 10

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

scaler = load(universal_scaler_filename)


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


def segment_into_sequences(data_normalized):
    sequences = []
    for start in range(0, data_normalized.shape[0] - sequence_length + 1, sequence_length):
        seq = data_normalized.iloc[start:start + sequence_length].values
        sequences.append(seq)
    return np.array(sequences)


def process_file(file_path):
    data = pd.read_csv(file_path)
    data = calculate_features(data)
    data_normalized = pd.DataFrame(scaler.transform(data), columns=data.columns)
    return segment_into_sequences(data_normalized)


def save_sequences(sequences, target_file):
    flattened_data = sequences.reshape(-1, sequences.shape[-1])
    df = pd.DataFrame(flattened_data,
                      columns=['X', 'Y', 'Time_Delta', 'Velocity_X', 'Velocity_Y', 'Acceleration_X', 'Acceleration_Y', 'Angular_Change',
                               'Distance'])
    df.to_csv(target_file, index=False)


for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        print(f'Processing {filename}...')
        file_path = os.path.join(source_dir, filename)
        sequences = process_file(file_path)

        target_file = os.path.join(target_dir, f'processed_{filename}')
        save_sequences(sequences, target_file)
        print(f'Saved processed data to {target_file}')

print("Data preparation complete.")
