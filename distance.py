import pandas as pd
import folium
from geopy.distance import geodesic 
from sklearn.model_selection import train_test_split
import math
import shutil
from predict import load_neighbors , find_path


# Function to load the intersection data from the CSV file
def load_intersection_data(file_path):
    df = pd.read_csv(file_path)
    intersection_data = {}
    for _, row in df.iterrows():
        scats = str(row['Scats_number'])
        intersection_data[scats] = (float(row['Latitude']), float(row['Longitude']))
    return intersection_data

# Function to calculate distance between two intersections
def calculate_intersection_distance(intersection1, intersection2, intersection_data):
    if intersection1 not in intersection_data or intersection2 not in intersection_data:
        return 0.0
    coord1 = intersection_data[intersection1]
    coord2 = intersection_data[intersection2]
    return geodesic(coord1, coord2).km

# Function to calculate the path distance from Starting point to End point 
def distance_path(path):
    # Check if the path is None
    if path is None:
        print("No valid path found between the intersections.")
        return 0.0
    
    file_path = "C:/Users/DELL/OneDrive/Documents/GitHub/Traffic-Flow-Prediction-System-TPFS/data/neighbouring_intersections.csv"
    intersection_data = load_intersection_data(file_path) 
    distance = 0.0

    # Calculate the distance for consecutive pairs of intersections in the path
    for i in range(len(path) - 1):
        distance += calculate_intersection_distance(path[i], path[i + 1], intersection_data)

    
    return (distance)
    