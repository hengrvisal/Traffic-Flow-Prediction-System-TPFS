import tkinter as tk
from tkinter import ttk
from datetime import datetime
from tkinter import messagebox
from pathfinder import pathfinder
from predict import load_neighbors
from PIL import Image, ImageTk, ImageEnhance
import os
import webbrowser

# Load neighbors data once at the start
neighbors = load_neighbors()

class TrafficFlowGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window title and size
        self.title("Traffic Flow Prediction System - TFPS")
        self.geometry("700x500")  
        self.configure(bg='#f0f0f0')

        # Load background image
        image_path = "gui_image/traffic.jpg"
        if os.path.exists(image_path):
            self.background_image = Image.open(image_path).convert("RGBA")
            enhancer = ImageEnhance.Brightness(self.background_image)
            self.background_image = enhancer.enhance(0.7)
            self.background_image = self.background_image.resize((700, 500), Image.LANCZOS)
            self.background_image_tk = ImageTk.PhotoImage(self.background_image)

            # Create a canvas for the background
            self.canvas = tk.Canvas(self, width=700, height=500)
            self.canvas.pack(fill="both", expand=True)
            self.canvas.create_image(0, 0, image=self.background_image_tk, anchor="nw")
        else:
            messagebox.showerror("Error", f"Background image not found: {image_path}")
            self.canvas = tk.Canvas(self, width=700, height=500)
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

        # Result display area
        self.result_text = tk.Text(self, height=8, wrap='word', bg='#f5f5f5', font=('Arial', 9))
        self.canvas.create_window(350, 350, window=self.result_text)
        self.result_text.config(state='disabled')

        # "View Route" Button placed below the text box
        self.view_route_button = tk.Button(self, text="View Route", command=self.view_route, font=("Helvetica", 10), bg="#2196F3", fg="white", bd=0)
        self.canvas.create_window(350, 430, window=self.view_route_button)

        # Status bar
        self.status_bar = tk.Label(self, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

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
                result = f"Routes from {src} to {dest} using {model} model on {date_time.strftime('%Y-%m-%d %H:%M')}:\n"
                for i, (estimated_time, total_distance, path, avg_traffic) in enumerate(self.generated_paths, 1):
                    result += f"\n{i}. Estimated time: {estimated_time:.2f} minutes\n"
                    result += f"   Total distance: {total_distance:.2f} km\n"
                    result += f"   Avg traffic: {avg_traffic:.2f} vehicles/5min\n"
                    result += f"   Path: {' -> '.join(path)}\n"
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
            messagebox.showerror("No Route Available", "Please generate a route first.")
            return



# Run the application
if __name__ == "__main__":
    app = TrafficFlowGUI()
    app.mainloop()
