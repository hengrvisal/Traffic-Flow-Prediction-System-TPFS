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
                dt.hour / 23.0,  # Hour normalized to [0, 1]
                dt.minute / 59.0,  # Minute normalized to [0, 1]
                dt.weekday() / 6.0,  # Weekday normalized to [0, 1]
                int(dt.weekday() < 5),  # Is weekday (1) or weekend (0)
                dt.day / 31.0,  # Day of month normalized to [0, 1]
                (dt.month - 1) / 11.0  # Month normalized to [0, 1]
            ]
            data.append(features[:input_shape[1]])
        return np.array(data).reshape((1,) + input_shape)
    elif model_type == 'SAES':
        features = [
            date_time.hour / 23.0,
            date_time.minute / 59.0,
            date_time.weekday() / 6.0,
            int(date_time.weekday() < 5),
            date_time.day / 31.0,
            (date_time.month - 1) / 11.0
        ]
        features.extend([0.5] * 12)  # Placeholder for recent traffic data
        return np.array(features).reshape(1, 18)


def denormalize_prediction(prediction, min_value=0, max_value=500):
    return int(round(min_value + prediction * (max_value - min_value)))


def interpret_traffic_flow(value):
    if value < 30:
        return "Very low traffic"
    elif value < 75:
        return "Low traffic"
    elif value < 150:
        return "Moderate traffic"
    elif value < 250:
        return "High traffic"
    else:
        return "Very high traffic"


def apply_time_adjustment(prediction, hour, is_weekday):
    # Base factors
    time_factors = {
        0: 0.3, 1: 0.2, 2: 0.15, 3: 0.15, 4: 0.2, 5: 0.4,  # Early morning
        6: 0.6, 7: 0.9, 8: 1.1, 9: 1.0,  # Morning rush
        10: 0.9, 11: 0.9, 12: 1.0, 13: 1.0, 14: 1.0,  # Midday
        15: 1.1, 16: 1.2, 17: 1.2, 18: 1.1,  # Evening rush
        19: 0.9, 20: 0.8, 21: 0.7, 22: 0.5, 23: 0.4  # Night
    }

    # Adjust factors for weekends
    if not is_weekday:
        time_factors = {h: max(0.5, f * 0.7) for h, f in time_factors.items()}

    return max(0, prediction * time_factors.get(hour, 1.0))


def predict_traffic_flow(path, date_time, model_type, recent_traffic_data=None):
    predictions = []
    is_weekday = date_time.weekday() < 5
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
                print(f"Raw prediction for site {site}: {prediction[0][0]}")  # Debug output
                denormalized_prediction = denormalize_prediction(prediction[0][0], 0, 500)
                print(f"Denormalized prediction for site {site}: {denormalized_prediction}")  # Debug output
                adjusted_prediction = apply_time_adjustment(denormalized_prediction, date_time.hour, is_weekday)
                print(f"Time-adjusted prediction for site {site}: {adjusted_prediction}")  # Debug output
                predictions.append((site, int(adjusted_prediction), input_shape))
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