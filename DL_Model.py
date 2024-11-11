import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import joblib

# Directories
raw_data_dir = 'Raw_Data_2023'
preprocessed_data_dir = 'Preprocessed_Data_2023'
os.makedirs(preprocessed_data_dir, exist_ok=True)

# Summary of rows removed
removed_rows_summary = []

def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(columns=['Slave_Device1_CH4_Voltage V'], inplace=True, errors='ignore')
    
    # Remove rows with zeros or negative values
    initial_rows = len(df)
    df = df[(df['Slave_Device1_CH1_FLOW m3/h'] > 0) &
            (df['Slave_Device1_CH2_VELOCITY m/s'] > 0) &
            (df['Slave_Device1_CH3_Pressure BAR'] > 0)]
    
    removed_rows = initial_rows - len(df)

    # Calculate additional features
    df['delta_flow'] = df['Slave_Device1_CH1_FLOW m3/h'].diff()
    df['delta_velocity'] = df['Slave_Device1_CH2_VELOCITY m/s'].diff()
    df['delta_pressure'] = df['Slave_Device1_CH3_Pressure BAR'].diff()
    
    df.dropna(inplace=True)
    return df, removed_rows

def train_svr_model(df):
    X = df[['Slave_Device1_CH1_FLOW m3/h', 'Slave_Device1_CH2_VELOCITY m/s']]
    y = df['Slave_Device1_CH3_Pressure BAR']
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Train SVR
    svr = SVR(kernel='rbf')
    svr.fit(X_scaled, y_scaled)

    # Save the scaler and model
    joblib.dump(scaler, 'scaler_svr.pkl')
    joblib.dump(svr, 'svr_model.pkl')
    
    return svr, scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(df, sequence_length=10):
    features = ['Slave_Device1_CH1_FLOW m3/h', 'Slave_Device1_CH2_VELOCITY m/s', 'Slave_Device1_CH3_Pressure BAR']
    data = df[features].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length, 2])  # Predicting next pressure value

    X, y = np.array(X), np.array(y)
    
    model = create_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
    
    # Save scaler and model
    joblib.dump(scaler, 'scaler_lstm.pkl')
    model.save('lstm_model.h5')
    
    return model, scaler

def detect_anomalies(df, svr, svr_scaler, lstm, lstm_scaler, sequence_length=10):
    # Prepare input for SVR
    X_svr = df[['Slave_Device1_CH1_FLOW m3/h', 'Slave_Device1_CH2_VELOCITY m/s']]
    X_svr_scaled = svr_scaler.transform(X_svr)
    pressure_pred_svr = svr.predict(X_svr_scaled)
    pressure_actual_scaled = svr_scaler.transform(df[['Slave_Device1_CH3_Pressure BAR']])
    
    # Prepare input for LSTM
    data = df[['Slave_Device1_CH1_FLOW m3/h', 'Slave_Device1_CH2_VELOCITY m/s', 'Slave_Device1_CH3_Pressure BAR']].values
    data_scaled = lstm_scaler.transform(data)
    
    X_lstm = []
    for i in range(len(data_scaled) - sequence_length):
        X_lstm.append(data_scaled[i:i + sequence_length])
    X_lstm = np.array(X_lstm)
    
    pressure_pred_lstm = lstm.predict(X_lstm).flatten()
    pressure_actual = data_scaled[sequence_length:, 2]
    
    # Define anomaly threshold
    svr_deviation = np.abs(pressure_pred_svr - pressure_actual_scaled.flatten())
    lstm_deviation = np.abs(pressure_pred_lstm - pressure_actual)
    
    threshold_svr = 0.2
    threshold_lstm = 0.15
    anomalies = (svr_deviation > threshold_svr) & (lstm_deviation > threshold_lstm)
    
    # Results
    df['Anomaly'] = np.concatenate([[False] * sequence_length, anomalies])
    return df

try:
    # Loop through each file in Raw_Data_2023
    for subdir, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                relative_subdir = os.path.relpath(subdir, raw_data_dir)
                output_subdir = os.path.join(preprocessed_data_dir, relative_subdir)
                os.makedirs(output_subdir, exist_ok=True)

                # Load and preprocess data
                df = pd.read_csv(file_path)
                df_preprocessed, removed_rows = preprocess_data(df)

                # Train models and detect anomalies
                svr, svr_scaler = train_svr_model(df_preprocessed)
                lstm, lstm_scaler = train_lstm_model(df_preprocessed)
                df_result = detect_anomalies(df_preprocessed, svr, svr_scaler, lstm, lstm_scaler)

                # Save the preprocessed file
                output_file_path = os.path.join(output_subdir, file.replace('.csv', '_corrected.csv'))
                df_result.to_csv(output_file_path, index=False)
                
                # Log removed rows
                removed_rows_summary.append(f"{file}: {removed_rows} rows removed")

    # Save the summary
    with open(os.path.join(preprocessed_data_dir, 'removed_rows_summary.txt'), 'w') as f:
        f.write("\n".join(removed_rows_summary))

except Exception as e:
    print(f"An error occurred: {e}")