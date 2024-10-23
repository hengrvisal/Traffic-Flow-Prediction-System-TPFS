import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta
from functools import lru_cache
import os

# Global variables
model_cache = {}
neighbors = None


def load_neighbors():
    global neighbors
    if neighbors is None:
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
    global model_cache
    key = f"{model_type.lower()}_{site}"
    if key not in model_cache:
        # model_path = f'model/sites_models/{key}.h5'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'sites_models', f'{model_type}_{site}.h5')
        try:
            model = load_model(model_path)
            print(f"Loaded {model_type} model for site {site}")
            model_cache[key] = model
        except:
            print(f"No {model_type} model found for site {site}")
            model_cache[key] = None
    return model_cache[key]


def prepare_input_data(date_time, input_shape, model_type):
    base_features = [
        date_time.hour / 23.0,
        date_time.minute / 59.0,
        date_time.weekday() / 6.0,
        int(date_time.weekday() < 5),
        date_time.day / 31.0,
        (date_time.month - 1) / 11.0
    ]

    if model_type in ['LSTM', 'GRU', 'RNN']:
        data = [base_features[:input_shape[1]]] * input_shape[0]
        return np.array(data).reshape((1,) + input_shape)
    elif model_type in ['SAES', 'SAES_FIXED']:
        features = base_features + [0.5] * 12  # Placeholder for recent traffic data
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
    time_factors = {
        0: 0.6, 1: 0.5, 2: 0.4, 3: 0.4, 4: 0.5, 5: 0.7,  # Early morning
        6: 0.9, 7: 1.4, 8: 1.5, 9: 1.3,  # Morning rush
        10: 1.1, 11: 1.1, 12: 1.2, 13: 1.2, 14: 1.1,  # Midday
        15: 1.2, 16: 1.5, 17: 1.6, 18: 1.4,  # Evening rush
        19: 1.2, 20: 1.0, 21: 0.9, 22: 0.8, 23: 0.7  # Night
    }

    if not is_weekday:
        time_factors = {h: max(0.6, f * 0.8) for h, f in time_factors.items()}

    return prediction * time_factors.get(hour, 1.0)


@lru_cache(maxsize=10000)
def cached_predict(site, date_time, model_type):
    model = load_model_for_site(site, model_type)
    if model:
        if model_type in ['LSTM', 'GRU', 'RNN']:
            input_shape = model.input_shape[1:]
        else:  # SAES
            input_shape = model.input_shape[1]

        input_data = prepare_input_data(date_time, input_shape, model_type)
        try:
            prediction = model.predict(input_data)

            # Model-specific processing
            if model_type in ['LSTM', 'GRU', 'RNN']:
                # LSTM and GRU might output a sequence, take the last value
                prediction = prediction[0][-1] if len(prediction[0]) > 1 else prediction[0][0]
            else:  # SAES and SAES_FIXED
                prediction = prediction[0][0]

            # Denormalize prediction
            denormalized_prediction = denormalize_prediction(prediction, 0, 500)

            # Apply time adjustment
            is_weekday = date_time.weekday() < 5
            adjusted_prediction = apply_time_adjustment(denormalized_prediction, date_time.hour, is_weekday)

            # Clamp prediction to a reasonable range
            final_prediction = max(0, min(adjusted_prediction, 500))

            return int(final_prediction), input_shape
        except Exception as e:
            print(f"Error predicting for site {site}: {str(e)}")
    return None, None


def predict_traffic_flow(path, date_time, model_type):
    return [cached_predict(site, date_time, model_type) + (site,) for site in path]


def traffic_flow_prediction():
    neighbors = load_neighbors()

    start = input("Enter starting SCATS site number: ")
    end = input("Enter ending SCATS site number: ")

    model_type = input("Enter model type (LSTM, GRU, or SAES): ").upper()
    while model_type not in ['LSTM', 'GRU', 'SAES', 'SAES_FIXED', 'RNN']:
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
        for pred, input_shape, site in predictions:
            if pred is not None:
                interpretation = interpret_traffic_flow(pred)
                print(f"SCATS site {site}: {pred} vehicles/5min - {interpretation} (Input shape: {input_shape})")
            else:
                print(f"SCATS site {site}: No {model_type} model available")
    else:
        print("No path found between the given SCATS sites.")


if __name__ == "__main__":
    traffic_flow_prediction()