import heapq
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
from functools import lru_cache
from predict import load_neighbors, predict_traffic_flow, load_model_for_site
from distance import load_intersection_data, calculate_intersection_distance
import random


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

@lru_cache(maxsize=1024)
def calculate_speed(traffic_flow):
    # Constants
    CAPACITY_FLOW = 500   # vehicles/hour (scaled down from 1500)
    CAPACITY_SPEED = 32   # km/hr
    SPEED_LIMIT = 60      # km/hr
    FLOW_AT_SPEED_LIMIT = 117  # vehicles/hour (scaled down from 351)
    MIN_SPEED = 20        # km/hr
    MAX_FLOW = 1000       # vehicles/hour (scaled down from 3000)

    try:
        # Convert flow from vehicles/5min to vehicles/hour
        hourly_flow = float(traffic_flow) * 12

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
    except (TypeError, ValueError) as e:
        print(f"Error in calculate_speed: {e}. Using default speed.")
        return SPEED_LIMIT  # Return speed limit as a default


def calculate_travel_time(flow, distance):
    # Use the flow of the starting SCATS site as per the report
    speed = calculate_speed(flow)
    travel_time = (distance / speed) * 60  # time in minutes

    # Add a fixed delay for the intersection (e.g., 30 seconds)
    intersection_delay = 0.5  # minutes

    return travel_time + intersection_delay

@lru_cache(maxsize=100000)
def cached_predict_traffic_flow(site, time_key):
    if site not in models:
        return None
    predictions = predict_traffic_flow([site], datetime.fromtimestamp(time_key), global_model_type)
    return predictions[0][1] if predictions and predictions[0][1] is not None else None


def find_efficient_paths(start: str, end: str, start_time: datetime, num_paths: int = 5) -> List[Tuple[float, float, List[str]]]:
    heap = [(0, 0, [start], start_time, set())]
    visited = {}
    efficient_paths = []
    path_diversity_penalty = {}
    unique_paths = set()

    def apply_diversity_penalty(path):
        penalty = 0
        for i, node in enumerate(path):
            penalty += path_diversity_penalty.get(node, 0) * (0.5 ** i)  # Exponential decay
        return penalty

    while heap and len(efficient_paths) < num_paths:
        (estimated_time, current_distance, path, current_time, visited_set) = heapq.heappop(heap)

        current = path[-1]
        if current == end:
            if tuple(path) not in unique_paths:
                efficient_paths.append((estimated_time, current_distance, path))
                unique_paths.add(tuple(path))
                for i, node in enumerate(path):
                    path_diversity_penalty[node] = path_diversity_penalty.get(node, 0) + 1
            continue

        time_key = int(current_time.timestamp() / 300) * 300
        if current in visited and visited[current] <= time_key:
            continue
        visited[current] = time_key

        # Introduce forced diversity for the first few steps
        if len(path) <= 3:
            neighbors_list = list(neighbors.get(current, []))
            random.shuffle(neighbors_list)
        else:
            neighbors_list = neighbors.get(current, [])

        for neighbor in neighbors_list:
            if neighbor in visited_set:
                continue

            segment_distance = get_distance(current, neighbor)
            flow_prediction = cached_predict_traffic_flow(current, time_key)

            if isinstance(flow_prediction, tuple) and len(flow_prediction) > 0:
                flow_current = flow_prediction[0]
            elif isinstance(flow_prediction, (int, float)):
                flow_current = flow_prediction
            else:
                print(f"Warning: Unexpected flow prediction for site {current}: {flow_prediction}")
                flow_current = None

            if flow_current is not None:
                try:
                    segment_time = calculate_travel_time(flow_current, segment_distance)
                    # Increase randomness
                    segment_time *= random.uniform(0.8, 1.2)
                    # Apply diversity penalty to partial path
                    segment_time += apply_diversity_penalty(path + [neighbor]) * 2
                except TypeError as e:
                    print(f"Error calculating travel time for site {current}: {e}")
                    segment_time = segment_distance / 0.5
            else:
                segment_time = segment_distance / 0.5

            new_estimated_time = estimated_time + segment_time
            new_distance = current_distance + segment_distance
            new_current_time = current_time + timedelta(minutes=int(segment_time))
            new_path = path + [neighbor]
            new_visited_set = visited_set.copy()
            new_visited_set.add(neighbor)

            heapq.heappush(heap, (new_estimated_time, new_distance, new_path, new_current_time, new_visited_set))

    # If we haven't found enough paths, relax the constraints and continue searching
    while len(efficient_paths) < num_paths and heap:
        (estimated_time, current_distance, path, current_time, visited_set) = heapq.heappop(heap)
        if path[-1] == end and tuple(path) not in unique_paths:
            efficient_paths.append((estimated_time, current_distance, path))
            unique_paths.add(tuple(path))

    return sorted(efficient_paths)[:num_paths]


def pathfinder(start: str, end: str, start_time: datetime, model_type: str) -> List[Tuple[float, float, List[str]]]:
    load_all_models(model_type)
    random.seed(0)  # Reset random seed for consistency
    return find_efficient_paths(start, end, start_time)


if __name__ == "__main__":
    start = input("Enter starting SCATS site number: ")
    end = input("Enter ending SCATS site number: ")
    model_type = input("Enter model type (LSTM, GRU, or SAES): ").upper()
    date_time_str = input("Enter date and time (YYYY-MM-DD HH:MM), or press Enter for current date and time: ")

    start_time = datetime.now() if not date_time_str.strip() else datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")

    efficient_paths = pathfinder(start, end, start_time, model_type)

    print(f"\nTop {len(efficient_paths)} most time-efficient routes from {start} to {end}:")
    for i, (estimated_time, total_distance, path) in enumerate(efficient_paths, 1):
        print(f"{i}. Estimated time: {estimated_time:.2f} minutes")
        print(f"   Number of intersections: {len(path) - 1}")
        print(f"   Total distance: {total_distance:.2f} km")
        print(f"   Path: {' -> '.join(path)}")
        print()

    if len(efficient_paths) < 5:
        print(f"Note: Only {len(efficient_paths)} unique paths were found.")