import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directories
training_dir = 'ready_for_training'

def load_and_combine_csv(directory):
    combined_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

# Load and combine data from all CSV files
combined_data = load_and_combine_csv(training_dir)
flattened_sequences = combined_data.values

# Parameters
sequence_length = 30  # Number of time steps in each sequence (ensure consistency with preprocessing)
n_features = flattened_sequences.shape[1]  # Number of features per time step
test_size = 0.2  # Proportion of data for testing

# Reshape the combined data into sequences
num_sequences = len(flattened_sequences) // sequence_length
print(num_sequences)
sequences = flattened_sequences[:num_sequences * sequence_length].reshape((num_sequences, sequence_length, n_features))

# Split the data into training and testing sets
X_train, X_test = train_test_split(sequences, test_size=test_size, random_state=42)

# Define the LSTM Autoencoder model
model = Sequential([
    LSTM(128, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
    LSTM(64, activation='relu', return_sequences=False),
    RepeatVector(sequence_length),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(128, activation='relu', return_sequences=True),
    TimeDistributed(Dense(n_features))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model for future anomaly detection
model.save('mouse_movement_anomaly_detection_model.keras')

print("Model training complete and saved.")