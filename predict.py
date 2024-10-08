import numpy as np
import pandas as pd
from keras.models import load_model
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


def prepare_future_sequence(data, scaler, lag=12):
    sequence = data['Lane 1 Flow (Veh/5 Minutes)'].tail(lag).values
    sequence = sequence.reshape(-1, 1)
    return scaler.transform(sequence)


def get_day_name(dt):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days[dt.weekday()]


def add_time_features(df):
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    return df


def predict_future_traffic(site_id, model_type, start_time, end_time, historical_data_file, lag=12):
    model_path = os.path.join('model', 'sites_models', f'{model_type}_{site_id}.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for site {site_id}")

    model = load_model(model_path)

    # Load and prepare historical data
    historical_data = pd.read_csv(historical_data_file)
    historical_data['datetime'] = pd.to_datetime(historical_data['5 Minutes'], format='%d/%m/%Y %H:%M')
    historical_data = historical_data.sort_values('datetime')

    # Create and fit the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(historical_data['Lane 1 Flow (Veh/5 Minutes)'].values.reshape(-1, 1))

    # Prepare the initial sequence
    initial_sequence = prepare_future_sequence(historical_data, scaler, lag)

    # Generate future timestamps
    future_timestamps = pd.date_range(start=start_time, end=end_time, freq='15T')

    predictions = []

    for timestamp in future_timestamps:
        # Prepare input features
        time_features = pd.DataFrame({'DateTime': [timestamp]})
        time_features = add_time_features(time_features)

        # Combine sequence with time features
        if model_type in ['lstm', 'gru']:
            model_input = [initial_sequence.reshape(1, lag, 1),
                           time_features[['Hour', 'Minute', 'DayOfWeek']].values]
        else:  # saes
            model_input = [initial_sequence.reshape(1, lag),
                           time_features[['Hour', 'Minute', 'DayOfWeek']].values]

        # Make prediction
        pred = model.predict(model_input, verbose=0)

        # Inverse transform the prediction
        pred_unscaled = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
        predictions.append(pred_unscaled)

        # Update the sequence for the next prediction
        initial_sequence = np.roll(initial_sequence, -1)
        initial_sequence[-1] = pred

    # Create results DataFrame
    results = pd.DataFrame({
        'DateTime': future_timestamps,
        'Predicted_Flow': np.round(predictions).astype(int),
        'Date': [t.strftime('%d/%m/%Y') for t in future_timestamps],
        'Time': [t.strftime('%H:%M') for t in future_timestamps],
        'Day_of_Week': [get_day_name(t) for t in future_timestamps]
    })

    return results


def analyze_historical_patterns(historical_data_file):
    data = pd.read_csv(historical_data_file)
    data['datetime'] = pd.to_datetime(data['5 Minutes'], format='%d/%m/%Y %H:%M')
    data = add_time_features(data.rename(columns={'datetime': 'DateTime'}))

    # Analyze patterns
    hourly_pattern = data.groupby('Hour')['Lane 1 Flow (Veh/5 Minutes)'].mean()
    daily_pattern = data.groupby('DayOfWeek')['Lane 1 Flow (Veh/5 Minutes)'].mean()

    return hourly_pattern, daily_pattern


# Example usage
if __name__ == "__main__":
    try:
        site_id = '2000'
        historical_data_file = 'data/splitted_data/2000_test.csv'

        # Analyze historical patterns
        hourly_pattern, daily_pattern = analyze_historical_patterns(historical_data_file)
        print("\nAverage Hourly Traffic Pattern:")
        print(hourly_pattern)
        print("\nAverage Daily Traffic Pattern:")
        print(daily_pattern)

        # Make future predictions
        predictions = predict_future_traffic(
            site_id=site_id,
            model_type='lstm',
            start_time=datetime(2024, 10, 7, 8, 0),  # Future date
            end_time=datetime(2024, 10, 7, 9, 0),
            historical_data_file=historical_data_file
        )

        print("\nPredicted Future Traffic Flow (vehicles per 5 minutes):")
        print(predictions[['Date', 'Time', 'Predicted_Flow']].to_string(index=False))

        print("\nPrediction Statistics:")
        print(f"Mean flow: {predictions['Predicted_Flow'].mean():.1f}")
        print(f"Min flow: {predictions['Predicted_Flow'].min()}")
        print(f"Max flow: {predictions['Predicted_Flow'].max()}")
        print(f"Standard deviation: {predictions['Predicted_Flow'].std():.1f}")

    except Exception as e:
        print(f"Error making predictions: {str(e)}")