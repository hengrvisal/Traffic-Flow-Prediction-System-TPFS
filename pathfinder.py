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

def calculate_speed(traffic_flow, is_peak_hour):
    # Constants
    CAPACITY_FLOW = 250   # vehicles/5min (3000 vehicles/hour)
    CAPACITY_SPEED = 35   # km/hr
    SPEED_LIMIT = 60 if not is_peak_hour else 50  # km/hr
    FLOW_AT_SPEED_LIMIT = 60  # vehicles/5min (720 vehicles/hour)
    MIN_SPEED = 25        # km/hr

    # Clamp traffic flow to a maximum value
    traffic_flow = min(traffic_flow, CAPACITY_FLOW * 2)

    if traffic_flow <= FLOW_AT_SPEED_LIMIT:
        return SPEED_LIMIT
    elif traffic_flow <= CAPACITY_FLOW:
        # Non-linear decrease from speed limit to capacity speed
        flow_ratio = (traffic_flow - FLOW_AT_SPEED_LIMIT) / (CAPACITY_FLOW - FLOW_AT_SPEED_LIMIT)
        speed_diff = SPEED_LIMIT - CAPACITY_SPEED
        return SPEED_LIMIT - (flow_ratio ** 1.5) * speed_diff
    else:
        # Sharper decrease for over-capacity
        over_capacity = traffic_flow - CAPACITY_FLOW
        speed_decrease = min(20, over_capacity / 10)  # Max 20 km/h decrease for very high traffic
        return max(MIN_SPEED, CAPACITY_SPEED - speed_decrease)

def find_multiple_paths(start: str, end: str, start_time: datetime, num_paths: int = 5) -> List[Tuple[float, float, List[str], float]]:
    heap = [(0, 0, [start], start_time, 0)]  # (estimated_time, distance, path, current_time, total_flow)
    paths = []
    visited = {}

    while heap and len(paths) < num_paths:
        (estimated_time, current_distance, path, current_time, total_flow) = heapq.heappop(heap)
        current = path[-1]

        if current == end:
            avg_traffic = total_flow / len(path) if len(path) > 0 else 0
            paths.append((estimated_time, current_distance, path, avg_traffic))
            continue

        visit_key = (current, current_time.strftime("%Y-%m-%d %H:%M"))  # Round to nearest minute
        if visit_key in visited and visited[visit_key] <= estimated_time:
            continue
        visited[visit_key] = estimated_time

        predictions = predict_traffic_flow([current], current_time, global_model_type)
        if predictions and predictions[0][0] is not None:
            flow_prediction = predictions[0][0]
        else:
            flow_prediction = 20  # Default value if prediction fails

        is_peak_hour = 7 <= current_time.hour <= 9 or 16 <= current_time.hour <= 18

        for neighbor in neighbors.get(current, []):
            if neighbor in path:
                continue

            segment_distance = get_distance(current, neighbor)
            speed = calculate_speed(flow_prediction, is_peak_hour)
            segment_time = (segment_distance / speed) * 60  # time in minutes

            new_estimated_time = estimated_time + segment_time
            new_distance = current_distance + segment_distance
            new_current_time = current_time + timedelta(minutes=segment_time)
            new_total_flow = total_flow + flow_prediction

            heapq.heappush(heap, (new_estimated_time, new_distance, path + [neighbor], new_current_time, new_total_flow))

    return sorted(paths, key=lambda x: x[0])[:num_paths]  # Sort by estimated time and return top num_paths

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

    print(f"\nTop {len(efficient_paths)} most time-efficient routes from {start} to {end} at {start_time}:")
    for i, (estimated_time, total_distance, path, avg_traffic) in enumerate(efficient_paths, 1):
        print(f"{i}. Estimated time: {estimated_time:.2f} minutes")
        print(f"   Number of intersections: {len(path) - 1}")
        print(f"   Total distance: {total_distance:.2f} km")
        print(f"   Average traffic: {avg_traffic:.2f} vehicles/5min")
        print(f"   Path: {' -> '.join(path)}")
        print()

    if len(efficient_paths) < 5:
        print(f"Note: Only {len(efficient_paths)} unique paths were found.")