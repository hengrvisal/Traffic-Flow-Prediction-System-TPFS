import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta
from functools import lru_cache

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
        model_path = f'model/sites_models/{key}.h5'
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

    if model_type in ['LSTM', 'GRU']:
        data = [base_features[:input_shape[1]]] * input_shape[0]
        return np.array(data).reshape((1,) + input_shape)
    elif model_type == 'SAES':
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
        0: 0.3, 1: 0.2, 2: 0.15, 3: 0.15, 4: 0.2, 5: 0.4,  # Early morning
        6: 0.6, 7: 0.9, 8: 1.1, 9: 1.0,  # Morning rush
        10: 0.9, 11: 0.9, 12: 1.0, 13: 1.0, 14: 1.0,  # Midday
        15: 1.1, 16: 1.2, 17: 1.2, 18: 1.1,  # Evening rush
        19: 0.9, 20: 0.8, 21: 0.7, 22: 0.5, 23: 0.4  # Night
    }

    if not is_weekday:
        time_factors = {h: max(0.5, f * 0.7) for h, f in time_factors.items()}

    return max(0, prediction * time_factors.get(hour, 1.0))

@lru_cache(maxsize=10000)
def cached_predict(site, date_time, model_type):
    model = load_model_for_site(site, model_type)
    if model:
        if model_type in ['LSTM', 'GRU']:
            input_shape = model.input_shape[1:]
        else:  # SAES
            input_shape = model.input_shape[1]

        input_data = prepare_input_data(date_time, input_shape, model_type)
        try:
            prediction = model.predict(input_data)
            denormalized_prediction = denormalize_prediction(prediction[0][0], 0, 500)
            is_weekday = date_time.weekday() < 5
            adjusted_prediction = apply_time_adjustment(denormalized_prediction, date_time.hour, is_weekday)
            return int(adjusted_prediction), input_shape
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