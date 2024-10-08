import numpy as np
import pandas as pd
from keras.models import load_model
from data.data import process_data
import os
from datetime import datetime, timedelta


def prepare_input_sequence(data, date_time, lag=12):
    """
    Prepare input sequence for prediction

    Args:
        data (DataFrame): Historical data with 'DateTime' and 'Volume' columns
        date_time (datetime): Target datetime for prediction
        lag (int): Number of previous time steps to use

    Returns:
        numpy array of shape (1, lag) or (1, lag, 1) for LSTM/GRU
    """
    # Get the lag number of previous values
    end_idx = data[data['DateTime'] == date_time].index[0] if date_time in data['DateTime'].values else len(data)
    start_idx = max(0, end_idx - lag)

    sequence = data['Volume'].iloc[start_idx:end_idx].values

    # Pad with zeros if we don't have enough historical data
    if len(sequence) < lag:
        sequence = np.pad(sequence, (lag - len(sequence), 0), 'constant')

    return sequence[-lag:]  # Take the last 'lag' values


def predict_traffic_by_time(site_id, model_type, start_time, end_time, historical_data_file, lag=12):
    """
    Make predictions for a specific time period

    Args:
        site_id (str): The SCATS site ID (e.g., '2000')
        model_type (str): Type of model ('lstm', 'gru', or 'saes')
        start_time (datetime): Start time for predictions
        end_time (datetime): End time for predictions
        historical_data_file (str): Path to historical data file
        lag (int): Number of time steps to use for prediction

    Returns:
        DataFrame with timestamps and predictions
    """
    # Load the trained model
    model_path = os.path.join('model', 'sites_models', f'{model_type}_{site_id}.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for site {site_id}")

    model = load_model(model_path)

    # Load historical data
    historical_data = pd.read_csv(historical_data_file)

    # Convert DateTime column to datetime type
    historical_data['DateTime'] = pd.to_datetime(historical_data['DateTime'])

    # Sort data by datetime
    historical_data = historical_data.sort_values('DateTime')

    # Generate prediction timestamps
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=15)  # Assuming 15-minute intervals

    predictions = []

    for t in timestamps:
        # Prepare input sequence
        sequence = prepare_input_sequence(historical_data, t, lag)

        # Reshape input based on model type
        if model_type in ['lstm', 'gru']:
            sequence = np.reshape(sequence, (1, lag, 1))
        else:  # saes
            sequence = np.reshape(sequence, (1, lag))

        # Make prediction
        pred = model.predict(sequence, verbose=0)
        predictions.append(pred[0][0])

    # Create results DataFrame
    results = pd.DataFrame({
        'DateTime': timestamps,
        'Predicted_Volume': predictions
    })

    # Add additional time-based columns
    results['Date'] = results['DateTime'].dt.date
    results['Time'] = results['DateTime'].dt.time
    results['Day_of_Week'] = results['DateTime'].dt.day_name()
    results['Hour'] = results['DateTime'].dt.hour
    results['Minute'] = results['DateTime'].dt.minute

    return results


# Example usage
if __name__ == "__main__":
    site_id = '2000'
    model_type = 'lstm'

    # Example: Predict for next hour
    start_time = datetime(2024, 10, 4, 8, 0)  # October 4, 2024, 8:00 AM
    end_time = datetime(2024, 10, 4, 9, 0)  # October 4, 2024, 9:00 AM

    try:
        predictions = predict_traffic_by_time(
            site_id=site_id,
            model_type=model_type,
            start_time=start_time,
            end_time=end_time,
            historical_data_file='data/splitted_data/2000_test.csv'  # Use your test data file
        )

        # Display predictions
        print("\nPredictions:")
        print(predictions[['DateTime', 'Predicted_Volume', 'Time']])

    except Exception as e:
        print(f"Error making predictions: {str(e)}")