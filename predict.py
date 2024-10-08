import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta


# Load neighboring intersections data
def load_neighbors():
    df = pd.read_csv('neighbouring_intersections.csv')
    neighbors = {}
    for _, row in df.iterrows():
        scats = str(row['Scats_number'])
        neighbors[scats] = row['Neighbours'].split(';')
    return neighbors


# Find path between two SCATS sites
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


# Load appropriate model for a SCATS site
def load_model_for_site(site, model_type):
    model_path = f'model/sites_models/{model_type.lower()}_{site}.h5'

    try:
        model = load_model(model_path)
        print(f"Loaded {model_type} model for site {site}")
        return model
    except:
        print(f"No {model_type} model found for site {site}")
        return None


# Prepare input data for the model
def prepare_input_data(date_time, input_shape):
    data = []
    for i in range(input_shape[0]):  # Use the first dimension of input_shape
        dt = date_time + timedelta(hours=i)
        features = [
            dt.hour / 24.0,  # Normalize hour to [0, 1]
            dt.minute / 60.0,  # Normalize minute to [0, 1]
            dt.weekday() / 6.0,  # Normalize weekday to [0, 1]
            dt.day / 31.0,  # Normalize day to [0, 1]
            dt.month / 12.0  # Normalize month to [0, 1]
        ]
        data.append(features[:input_shape[1]])  # Use only as many features as the input shape allows
    return np.array(data).reshape((1,) + input_shape)


# Predict traffic flow along the path
def denormalize_prediction(prediction, max_value=500):
    """
    Convert the normalized prediction back to the original scale.
    Adjust max_value if a different scaling was used during training.
    """
    return int(round(prediction * max_value))


def interpret_traffic_flow(value):
    if value < 50:
        return "Low traffic"
    elif value < 150:
        return "Moderate traffic"
    elif value < 250:
        return "High traffic"
    else:
        return "Very high traffic"


# Predict traffic flow along the path
def predict_traffic_flow(path, date_time, model_type):
    predictions = []
    for site in path:
        model = load_model_for_site(site, model_type)
        if model:
            input_shape = model.input_shape[1:]  # Get input shape excluding batch size
            input_data = prepare_input_data(date_time, input_shape)
            prediction = model.predict(input_data)
            denormalized_prediction = denormalize_prediction(prediction[0][0])
            predictions.append((site, denormalized_prediction, input_shape))
        else:
            predictions.append((site, None, None))
    return predictions


# Main function
def main():
    neighbors = load_neighbors()

    start = input("Enter starting SCATS site number: ")
    end = input("Enter ending SCATS site number: ")

    model_type = input("Enter model type (LSTM or GRU): ").upper()
    while model_type not in ['LSTM', 'GRU']:
        print("Invalid model type. Please enter either LSTM or GRU.")
        model_type = input("Enter model type (LSTM or GRU): ").upper()

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
    main()