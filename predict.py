import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta


def load_neighbors():
    df = pd.read_csv('neighbouring_intersections.csv')
    neighbors = {}
    for _, row in df.iterrows():
        scats = str(row['Scats_number'])
        neighbors[scats] = row['Neighbours'].split(';')
    return neighbors


def find_path(start, end, neighbors):
    queue = [(start, [start])]
    visited = set()

    while queue:
        (node, path) = queue.pop(0)
        if node not in visited:
            if node == end:
                return path
            visited.add(node)
            for neighbor in neighbors.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None


def load_model_for_site(site, model_type):
    model_path = f'model/sites_models/{model_type.lower()}_{site}.h5'

    try:
        model = load_model(model_path)
        print(f"Loaded {model_type} model for site {site}")
        return model
    except:
        print(f"No {model_type} model found for site {site}")
        return None


def prepare_input_data(date_time, input_shape, model_type):
    if model_type in ['LSTM', 'GRU']:
        data = []
        for i in range(input_shape[0]):
            dt = date_time + timedelta(minutes=5*i)
            features = [
                dt.hour / 24.0,
                dt.minute / 60.0,
                dt.weekday() / 6.0,
                dt.day / 31.0,
                dt.month / 12.0
            ]
            data.append(features[:input_shape[1]])
        return np.array(data).reshape((1,) + input_shape)
    elif model_type == 'SAES':
        # For SAES, we'll create an 18-feature input
        # First 5 features are time-based, last 13 are placeholders for recent traffic data
        features = [
            date_time.hour / 24.0,
            date_time.minute / 60.0,
            date_time.weekday() / 6.0,
            date_time.day / 31.0,
            date_time.month / 12.0
        ]
        # Add 13 placeholder values for recent traffic data
        features.extend([0.5] * 13)  # Using 0.5 as a neutral placeholder value
        return np.array(features).reshape(1, 18)


def denormalize_prediction(prediction, min_value=0, max_value=500):
    return int(round(min_value + prediction * (max_value - min_value)))


def interpret_traffic_flow(value):
    if value < 50:
        return "Low traffic"
    elif value < 150:
        return "Moderate traffic"
    elif value < 250:
        return "High traffic"
    else:
        return "Very high traffic"


def predict_traffic_flow(path, date_time, model_type, recent_traffic_data=None):
    predictions = []
    for site in path:
        model = load_model_for_site(site, model_type)
        if model:
            if model_type in ['LSTM', 'GRU']:
                input_shape = model.input_shape[1:]
            else:  # SAES
                input_shape = model.input_shape[1]

            input_data = prepare_input_data(date_time, input_shape, model_type)
            try:
                prediction = model.predict(input_data)
                denormalized_prediction = denormalize_prediction(prediction[0][0], 500, 0)
                predictions.append((site, denormalized_prediction, input_shape))
            except Exception as e:
                print(f"Error predicting for site {site}: {str(e)}")
                predictions.append((site, None, input_shape))
        else:
            print(f"No {model_type} model available for site {site}")
            predictions.append((site, None, None))
    return predictions


def traffic_flow_prediction():
    neighbors = load_neighbors()

    start = input("Enter starting SCATS site number: ")
    end = input("Enter ending SCATS site number: ")

    model_type = input("Enter model type (LSTM, GRU, or SAES): ").upper()
    while model_type not in ['LSTM', 'GRU', 'SAES']:
        print("Invalid model type. Please enter LSTM, GRU, or SAES.")
        model_type = input("Enter model type (LSTM, GRU, or SAES): ").upper()

    date_time_str = input("Enter date and time (YYYY-MM-DD HH:MM), or press Enter for current date and time: ")
    if date_time_str.strip() == "":
        date_time = datetime.now()
    else:
        try:
            date_time = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
        except ValueError:
            print("Invalid date format. Using current date and time.")
            date_time = datetime.now()

    path = find_path(start, end, neighbors)

    if path:
        print(f"Path found: {' -> '.join(path)}")
        predictions = predict_traffic_flow(path, date_time, model_type)

        print(f"\nPredicted traffic flow for {date_time.strftime('%Y-%m-%d %H:%M')} using {model_type} models:")
        for site, pred, input_shape in predictions:
            if pred is not None:
                interpretation = interpret_traffic_flow(pred)
                print(f"SCATS site {site}: {pred} vehicles/5min - {interpretation} (Input shape: {input_shape})")
            else:
                print(f"SCATS site {site}: No {model_type} model available")
    else:
        print("No path found between the given SCATS sites.")


if __name__ == "__main__":
    traffic_flow_prediction()