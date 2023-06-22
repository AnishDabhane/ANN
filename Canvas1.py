import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('path_to_your_model.pkl')

# Constants
CANVAS_WIDTH = 300
CANVAS_HEIGHT = 300

# Function to preprocess the user's input
def preprocess_input(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)
    image = image.flatten()
    image = image / 255.0
    return image

# Function to predict the digit
def predict_digit():
    # Get the user's input from the canvas
    image = Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), fill='black')

    canvas_image = canvas.postscript(colormode='gray')
    image.paste(Image.open(canvas_image), (50, 50))

    # Preprocess the input
    preprocessed_image = preprocess_input(image)

    # Make the prediction
    prediction = model.predict([preprocessed_image])
    predicted_digit = prediction[0]

    # Display the predicted digit
    messagebox.showinfo("Prediction", f"The predicted digit is: {predicted_digit}")

# Create the main window
window = tk.Tk()
window.title("Handwritten Digit Recognition")

# Create the canvas
canvas = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack()

# Function to handle mouse movements
def on_mouse_drag(event):
    x, y = event.x, event.y
    radius = 8
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black')

# Bind mouse drag event to canvas
canvas.bind('<B1-Motion>', on_mouse_drag)

# Create the predict button
predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.pack()

# Start the main event loop
window.mainloop()
