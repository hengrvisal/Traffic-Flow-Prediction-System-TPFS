import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from tkinter import messagebox
from pathfinder import pathfinder
from predict import load_neighbors
from PIL import Image, ImageTk, ImageEnhance
import os
import webbrowser
import folium
import csv
import json

# Load neighbors data once at the start
neighbors = load_neighbors()
TRAFFIC_NETWORK = 'neighbouring_intersections.csv'

class TrafficFlowGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window title and size
        self.title("Traffic Flow Prediction System - TFPS")
        self.geometry("700x500")
        self.configure(bg='#f0f0f0')

        # Main frame to hold canvas and status bar
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Canvas frame for background image and content
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        # Load background image
        image_path = "gui_image/traffic.jpg"
        if os.path.exists(image_path):
            self.background_image = Image.open(image_path).convert("RGBA")
            enhancer = ImageEnhance.Brightness(self.background_image)
            self.background_image = enhancer.enhance(0.7)
            self.background_image = self.background_image.resize((700, 500), Image.LANCZOS)
            self.background_image_tk = ImageTk.PhotoImage(self.background_image)

            # Create a canvas for the background
            self.canvas = tk.Canvas(self.canvas_frame, width=700, height=480)
            self.canvas.pack(fill="both", expand=True)
            self.canvas.create_image(0, 0, image=self.background_image_tk, anchor="nw")
        else:
            messagebox.showerror("Error", f"Background image not found: {image_path}")
            self.canvas = tk.Canvas(self.canvas_frame, width=700, height=480)
            self.canvas.pack(fill="both", expand=True)

        # Frame for the input fields
        self.input_frame = tk.Frame(self, bg="#ffffff", highlightbackground="#f0f0f0", highlightthickness=2)
        self.input_frame.place(relx=0.5, rely=0.3, anchor="center")

        self.path_finder_label = tk.Label(self.input_frame, text="Route Detection", font=("Helvetica", 13, "bold"), bg='#ffffff', fg='#333')
        self.path_finder_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Model selection dropdown
        model_label = tk.Label(self.input_frame, text="Select Model:", font=("Helvetica", 10), bg="#ffffff", fg="#333")
        model_label.grid(row=1, column=0, padx=10, pady=5, sticky='e')

        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self.input_frame, textvariable=self.model_var, font=("Helvetica", 10))
        self.model_dropdown['values'] = ("LSTM", "GRU", "SAES")
        self.model_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky='ew')
        self.model_dropdown.current(0)

        # Source SCATS input
        source_label = tk.Label(self.input_frame, text="Origin Node:", font=("Helvetica", 10), bg="#ffffff", fg="#333")
        source_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')

        self.source_entry = tk.Entry(self.input_frame, font=("Helvetica", 10))
        self.source_entry.grid(row=2, column=1, padx=10, pady=5, sticky='ew')

        # Destination SCATS input
        destination_label = tk.Label(self.input_frame, text="Destination Node:", font=("Helvetica", 10), bg="#ffffff", fg="#333")
        destination_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')

        self.destination_entry = tk.Entry(self.input_frame, font=("Helvetica", 10))
        self.destination_entry.grid(row=3, column=1, padx=10, pady=5, sticky='ew')

        # Date/Time input
        datetime_label = tk.Label(self.input_frame, text="Date/Time:", font=("Helvetica", 10), bg="#ffffff", fg="#333")
        datetime_label.grid(row=4, column=0, padx=10, pady=5, sticky='e')

        self.datetime_entry = tk.Entry(self.input_frame, font=("Helvetica", 10), fg="grey")
        self.datetime_entry.insert(0, "YYYY-MM-DD HH:MM")
        self.datetime_entry.bind("<FocusIn>", self.clear_placeholder)
        self.datetime_entry.bind("<FocusOut>", self.add_placeholder)
        self.datetime_entry.grid(row=4, column=1, padx=10, pady=5, sticky='ew')

        # Configure column weights for better resizing behavior
        self.input_frame.columnconfigure(0, weight=1)
        self.input_frame.columnconfigure(1, weight=3)

        # Generate Route Button
        generate_button = tk.Button(self.input_frame, text="Generate Route", command=self.generate_route, font=("Helvetica", 10), bg="#4CAF50", fg="white", bd=0)
        generate_button.grid(row=5, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

        # Result display area with a scrollbar
        self.result_text = ScrolledText(self, height=8, wrap='word', bg='#f5f5f5', font=('Arial', 9))
        self.canvas.create_window(350, 350, window=self.result_text)
        self.result_text.config(state='disabled')

        # "View Route" Button placed below the text box
        self.view_route_button = tk.Button(self, text="View Route", command=self.view_route, font=("Helvetica", 10), bg="#4CAF50", fg="white", bd=0)
        self.canvas.create_window(350, 430, window=self.view_route_button)

        # Create a frame for the status bar and add it at the bottom of the main_frame
        self.status_frame = tk.Frame(self.main_frame)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Status bar
        self.status_bar = tk.Label(self.status_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Store generated paths
        self.generated_paths = []

    def clear_placeholder(self, event):
        if self.datetime_entry.get() == "YYYY-MM-DD HH:MM":
            self.datetime_entry.delete(0, tk.END)
            self.datetime_entry.config(fg="black")

    def add_placeholder(self, event):
        if not self.datetime_entry.get():
            self.datetime_entry.insert(0, "YYYY-MM-DD HH:MM")
            self.datetime_entry.config(fg="grey")

    def get_date_time(self):
        date_time_str = self.datetime_entry.get().strip()
        if date_time_str == "YYYY-MM-DD HH:MM" or not date_time_str:
            return datetime.now()
        try:
            return datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showerror("Input Error", "Invalid date/time format. Use YYYY-MM-DD HH:MM.")
            return None

    def generate_route(self):
        self.status_bar.config(text="Generating route...")
        src = self.source_entry.get().strip()
        dest = self.destination_entry.get().strip()
        model = self.model_var.get()

        if not src or not dest:
            messagebox.showerror("Input Error", "Please fill in the Origin and Destination Nodes.")
            self.status_bar.config(text="Error: Missing input fields.")
            return

        date_time = self.get_date_time()
        if date_time is None:
            self.status_bar.config(text="Error: Invalid date/time format.")
            return

        try:
            self.generated_paths = pathfinder(src, dest, date_time, model)
            if not self.generated_paths:
                result = "No routes found."
            else:
                result = f"Routes from {src} to {dest} using {model} model on {date_time.strftime('%Y-%m-%d %H:%M')}:\n\n"
                for i, (estimated_time, total_distance, path, avg_traffic) in enumerate(self.generated_paths, 1):
                    result += f"Route {i}\n"
                    result += f"   Estimated time: {estimated_time:.2f} minutes\n"
                    result += f"   Total distance: {total_distance:.2f} km\n"
                    result += f"   Avg traffic: {avg_traffic:.2f} vehicles/5min\n"
                    result += f"   Path: {' -> '.join(path)}\n\n"
            self.status_bar.config(text="Route generation complete.")
        except Exception as e:
            result = f"Error generating route: {str(e)}"
            self.status_bar.config(text="Error generating route.")

        self.display_result(result)

    def display_result(self, text):
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state='disabled')

    def view_route(self):
        if not self.generated_paths:
            # If no routes are generated, show SCATS locations
            self.status_bar.config(text="No route available, showing SCATS locations...")
            self.render_map_with_scat_sites()
        else:
            self.status_bar.config(text="Displaying generated route...")
            self.render_map_with_routes(self.generated_paths)

    def getCoords(self, scat):
        """Fetch the coordinates of the SCATS location."""
        with open(TRAFFIC_NETWORK, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Scats_number'] == str(scat):
                    # Adding a small offset to simulate more precise locations
                    lat = float(row['Latitude']) + 0.00123
                    lon = float(row['Longitude']) + 0.00123
                    return lon, lat  # Return longitude, latitude

        print("Unable to find SCATS location")
        return 0, 0

    def generate_geojson(self, routes):
        """Generate GeoJSON data for routes."""
        data = {
            "type": "FeatureCollection",
            "features": []
        }

        for index, route in reversed(list(enumerate(routes))):
            weight = 5
            color = "#3484F0" if index == 0 else "#757575"  # Blue for the best route, grey for others
            coords = []
            for scat in route:
                lon, lat = self.getCoords(scat)
                coords.append([lon, lat])

            feature = {
                "type": "Feature",
                "properties": {
                    "stroke": color,
                    "stroke-width": weight
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }
            data["features"].append(feature)

        return json.dumps(data)

    def draw_markers(self, map_obj, src, dest):
        """Draw markers for the source and destination."""
        src_lon, src_lat = self.getCoords(src)
        dest_lon, dest_lat = self.getCoords(dest)

        folium.Marker([src_lat, src_lon], popup=f"<strong>Start</strong> SCATS: {src}",
                      icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(map_obj)
        folium.Marker([dest_lat, dest_lon], popup=f"<strong>Finish</strong> SCATS: {dest}",
                      icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(map_obj)

    def draw_nodes(self, map_obj):
        """Draw all SCATS nodes on the map."""
        with open(TRAFFIC_NETWORK, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                lon, lat = float(row['Longitude']) + 0.001, float(row['Latitude']) + 0.001
                folium.Circle(
                    radius=5,
                    location=[lat, lon],
                    popup=f"SCATS: {row['Scats_number']}",
                    color="#5A5A5A",
                    fill=False
                ).add_to(map_obj)

    def render_map_with_routes(self, routes):
        """Render the map with the generated routes."""
        src = routes[0][0]
        dest = routes[0][-1]

        geojson_data = self.generate_geojson(routes)

        # Create the map centered around a specific location
        map_obj = folium.Map(location=[-37.831219, 145.056965], zoom_start=13, tiles="cartodbpositron")

        # Plot the routes
        folium.GeoJson(geojson_data, style_function=lambda x: {
            'color': x['properties']['stroke'],
            'weight': x['properties']['stroke-width']
        }).add_to(map_obj)

        # Add markers and nodes
        self.draw_markers(map_obj, src, dest)
        self.draw_nodes(map_obj)

        # Save to an HTML file and open it in a web browser
        map_obj.save("index.html")
        webbrowser.open("index.html")

    def render_map_with_scat_sites(self):
        """Render the map with only the SCATS locations."""
        # Create the map centered around a specific location
        map_obj = folium.Map(location=[-37.831219, 145.056965], zoom_start=13, tiles="cartodbpositron")

        # Draw SCATS nodes
        self.draw_nodes(map_obj)

        # Add a message about no route found
        folium.Marker(
            location=[-37.831219, 145.056965],
            popup="<strong>No route found</strong>",
            icon=folium.Icon(color='red')
        ).add_to(map_obj)

        # Save to an HTML file and open it in a web browser
        map_obj.save("index.html")
        webbrowser.open("index.html")

# Run the application
if __name__ == "__main__":
    app = TrafficFlowGUI()
    app.mainloop()

