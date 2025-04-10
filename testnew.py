# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:58:36 2024

@author: OKhalid
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ThermalImageAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal Image Analyzer")

        # Button to upload image
        self.upload_button = tk.Button(root, text="Upload Thermal Image", command=self.upload_image)
        self.upload_button.pack()

        # Canvas to display the image
        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        # Variables to hold the image and points
        self.image = None
        self.image_tk = None
        self.start_point = None
        self.end_point = None
        self.line_id = None

    def upload_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tiff")])
        if not file_path:
            return

        # Load the image using OpenCV and convert to grayscale
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize image for display if too large
        max_size = 800
        if self.image.shape[1] > max_size or self.image.shape[0] > max_size:
            scale = max_size / max(self.image.shape[1], self.image.shape[0])
            self.image = cv2.resize(self.image, (int(self.image.shape[1] * scale), int(self.image.shape[0] * scale)))

        # Convert the image to Tkinter format
        self.image_tk = ImageTk.PhotoImage(Image.fromarray(self.image))
        
        # Display the image on the canvas
        self.canvas.config(width=self.image.shape[1], height=self.image.shape[0])
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

        # Bind mouse events for drawing line
        self.canvas.bind("<Button-1>", self.set_start_point)
        self.canvas.bind("<B1-Motion>", self.draw_line)

    def set_start_point(self, event):
        # Store the starting point of the line
        self.start_point = (event.x, event.y)
        self.end_point = None
        if self.line_id:
            self.canvas.delete(self.line_id)

    def draw_line(self, event):
        # Draw a line from the starting point to the current mouse position
        self.end_point = (event.x, event.y)
        if self.line_id:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1], fill="red", width=2)

        # After drawing the line, plot the temperature variation along it
        self.plot_temperature_along_line()

    def plot_temperature_along_line(self):
        if self.start_point and self.end_point:
            # Extract the points of the line in the image
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            line_points = list(zip(np.linspace(x1, x2, num=100).astype(int), np.linspace(y1, y2, num=100).astype(int)))

            # Get the pixel intensity (temperature) along the line
            temperatures = [self.image[y, x] for x, y in line_points]

            # Plot the temperature variation
            plt.figure("Temperature Variation")
            plt.plot(temperatures, color="blue")
            plt.title("Temperature Variation along the Line")
            plt.xlabel("Distance along the line")
            plt.ylabel("Temperature (Pixel Intensity)")
            plt.show()

# Create the Tkinter root window
root = tk.Tk()

# Create the application
app = ThermalImageAnalyzer(root)

# Run the application
root.mainloop()
