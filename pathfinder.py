import heapq
from typing import List, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
from predict import load_neighbors, predict_traffic_flow, load_model_for_site
from distance import load_intersection_data, calculate_intersection_distance

# Global variables
neighbors = load_neighbors()
intersection_data = load_intersection_data("neighbouring_intersections.csv")
precomputed_distances = {}
models = {}
global_model_type = ""

@lru_cache(maxsize=1000000)
def get_distance(site1, site2):
    key = (min(site1, site2), max(site1, site2))
    if key not in precomputed_distances:
        precomputed_distances[key] = calculate_intersection_distance(site1, site2, intersection_data)
    return precomputed_distances[key]

def load_all_models(model_type: str):
    global models, global_model_type
    global_model_type = model_type
    all_sites = set(neighbors.keys()) | set(site for sublist in neighbors.values() for site in sublist)
    for site in all_sites:
        if site not in models:
            models[site] = load_model_for_site(site, model_type)

def calculate_speed(site, current_time):
    # Constants
    CAPACITY_FLOW = 500   # vehicles/hour
    CAPACITY_SPEED = 32   # km/hr
    SPEED_LIMIT = 60      # km/hr
    FLOW_AT_SPEED_LIMIT = 117  # vehicles/hour
    MIN_SPEED = 10        # km/hr
    MAX_FLOW = 1000       # vehicles/hour

    # Get traffic prediction
    predictions = predict_traffic_flow([site], current_time, global_model_type)
    if predictions and predictions[0][1] is not None:
        hourly_flow = predictions[0][1] * 12  # Convert from vehicles/5min to vehicles/hour
    else:
        hourly_flow = FLOW_AT_SPEED_LIMIT  # Default to flow at speed limit if no prediction

    if hourly_flow <= FLOW_AT_SPEED_LIMIT:
        return SPEED_LIMIT
    elif hourly_flow <= CAPACITY_FLOW:
        # Under capacity: Parabolic decrease from speed limit to capacity speed
        flow_diff = hourly_flow - FLOW_AT_SPEED_LIMIT
        max_flow_diff = CAPACITY_FLOW - FLOW_AT_SPEED_LIMIT
        speed_diff = SPEED_LIMIT - CAPACITY_SPEED
        return SPEED_LIMIT - (flow_diff / max_flow_diff) ** 2 * speed_diff
    else:
        # Over capacity: Linear decrease from capacity speed to minimum speed
        over_capacity = hourly_flow - CAPACITY_FLOW
        max_over_capacity = MAX_FLOW - CAPACITY_FLOW
        speed_decrease = (CAPACITY_SPEED - MIN_SPEED) * (over_capacity / max_over_capacity)
        return max(MIN_SPEED, CAPACITY_SPEED - speed_decrease)

def calculate_travel_time(site, distance, current_time):
    speed = calculate_speed(site, current_time)
    travel_time = (distance / speed) * 60  # time in minutes
    intersection_delay = 0.5  # minutes
    return travel_time + intersection_delay

def find_multiple_paths(start: str, end: str, start_time: datetime, num_paths: int = 5) -> List[Tuple[float, float, List[str], float]]:
    heap = [(0, 0, [start], start_time, 0)]  # (estimated_time, distance, path, current_time, total_flow)
    paths = []
    visited = set()

    while heap and len(paths) < num_paths:
        (estimated_time, current_distance, path, current_time, total_flow) = heapq.heappop(heap)
        current = path[-1]

        if current == end:
            avg_traffic = total_flow / len(path) if len(path) > 0 else 0
            paths.append((estimated_time, current_distance, path, avg_traffic))
            continue

        visit_key = (current, current_time.strftime("%Y-%m-%d %H:%M"))  # Round to nearest minute
        if visit_key in visited:
            continue
        visited.add(visit_key)

        for neighbor in neighbors.get(current, []):
            if neighbor in path:
                continue

            segment_distance = get_distance(current, neighbor)
            segment_time = calculate_travel_time(current, segment_distance, current_time)

            new_estimated_time = estimated_time + segment_time
            new_distance = current_distance + segment_distance
            new_current_time = current_time + timedelta(minutes=segment_time)

            # Get traffic prediction for total flow calculation
            predictions = predict_traffic_flow([current], new_current_time, global_model_type)
            if predictions and predictions[0][1] is not None:
                flow_prediction = predictions[0][1] * 12  # Convert from vehicles/5min to vehicles/hour
            else:
                flow_prediction = 0  # Use 0 if no prediction available
            new_total_flow = total_flow + flow_prediction

            heapq.heappush(heap, (new_estimated_time, new_distance, path + [neighbor], new_current_time, new_total_flow))

    return sorted(paths, key=lambda x: x[0])  # Sort by estimated time

def pathfinder(start: str, end: str, start_time: datetime, model_type: str) -> List[Tuple[float, float, List[str], float]]:
    global global_model_type
    global_model_type = model_type
    load_all_models(model_type)
    return find_multiple_paths(start, end, start_time)

if __name__ == "__main__":
    start = input("Enter starting SCATS site number: ")
    end = input("Enter ending SCATS site number: ")
    model_type = input("Enter model type (LSTM, GRU, or SAES): ").upper()
    date_time_str = input("Enter date and time (YYYY-MM-DD HH:MM), or press Enter for current date and time: ")

    start_time = datetime.now() if not date_time_str.strip() else datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")

    efficient_paths = pathfinder(start, end, start_time, model_type)

    print(f"\nTop 5 most time-efficient routes from {start} to {end} at {start_time}:")
    for i, (estimated_time, total_distance, path, avg_traffic) in enumerate(efficient_paths, 1):
        print(f"{i}. Estimated time: {estimated_time:.2f} minutes")
        print(f"   Number of intersections: {len(path) - 1}")
        print(f"   Total distance: {total_distance:.2f} km")
        print(f"   Average traffic: {avg_traffic:.2f} vehicles/hour")
        print(f"   Path: {' -> '.join(path)}")
        print()

    if len(efficient_paths) < 5:
        print(f"Note: Only {len(efficient_paths)} unique paths were found.")