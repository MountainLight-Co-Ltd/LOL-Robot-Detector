import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = 'ready_for_training'

def load_and_combine_csv(directory):
    combined_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

def train_model(training_dir, sequence_length, test_size):
    combined_data = load_and_combine_csv(training_dir)
    flattened_sequences = combined_data.values

    n_features = flattened_sequences.shape[1]

    num_sequences = len(flattened_sequences) // sequence_length
    sequences = flattened_sequences[:num_sequences * sequence_length].reshape(
        (num_sequences, sequence_length, n_features))

    X_train, X_test = train_test_split(sequences, test_size=test_size, random_state=42)

    model = Sequential([
        LSTM(128, activation='tanh', input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),  # Added dropout for regularization
        LSTM(64, activation='tanh', return_sequences=False),
        RepeatVector(sequence_length),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.2),  # Added dropout for regularization
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(n_features, activation='linear')) 
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    history = model.fit(X_train, X_train, epochs=500, validation_data=(X_test, X_test), callbacks=[early_stopping])
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.save('mouse_movement_anomaly_detection_model.keras')

train_model(data_path, 10, 0.2)

print("Model training complete and saved.")
