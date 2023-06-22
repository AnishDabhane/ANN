import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import joblib
import numpy as np
# Load the pre-trained model
model = joblib.load('TensorFlow.py')

# Constants
CANVAS_WIDTH = 300
CANVAS_HEIGHT = 300
BRUSH_SIZES = {
    'small': 4,
    'medium': 8,
    'large': 12
}
BRUSH_STYLES = {
    'round': 'round',
    'square': 'square',
    'diamond': 'diamond',
    'pencil': 'pencil',
    'highlighter': 'highlighter'
}
DEFAULT_BRUSH_SIZE = 'medium'
DEFAULT_BRUSH_STYLE = 'round'

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

# Function to clear the canvas
def clear_canvas():
    canvas.delete('all')

# Function to set the brush size
def set_brush_size(size):
    global brush_size
    brush_size = BRUSH_SIZES[size]

# Function to set the brush style
def set_brush_style(style):
    global brush_style
    brush_style = BRUSH_STYLES[style]

# Create the main window
window = tk.Tk()
window.title("Handwritten Digit Recognition")

# Create the canvas
canvas = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack()

# Function to handle mouse movements
def on_mouse_drag(event):
    x, y = event.x, event.y
    radius = brush_size
    brush_type = brush_style

    if brush_type == 'round':
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black')
    elif brush_type == 'square':
        canvas.create_rectangle(x - radius, y - radius, x + radius, y + radius, fill='black')
    elif brush_type == 'diamond':
        canvas.create_polygon(x, y - radius, x - radius, y, x, y + radius, x + radius, y, fill='black')
    elif brush_type == 'pencil':
        opacity = 0.3
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black', outline='',
                           stipple='gray12', width=int(opacity * radius))
    elif brush_type == 'highlighter':
        opacity = 0.5
        highlighter_colors = ['yellow', 'green', 'cyan', 'pink']

# Bind mouse drag event to canvas
canvas.bind('<B1-Motion>', on_mouse_drag)

# Create the predict button
predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.pack()

# Create the clear button
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

## Create the brush size buttons
brush_size_frame = tk.Frame(window)
brush_size_frame.pack()

for size in BRUSH_SIZES:
    size_button = tk.Button(brush_size_frame, text=size.capitalize(), command=lambda s=size: set_brush_size(s))
    size_button.pack(side='left', padx=5, pady=5)

# Create the brush style buttons
brush_style_frame = tk.Frame(window)
brush_style_frame.pack()

for style in BRUSH_STYLES:
    style_button = tk.Button(brush_style_frame, text=style.capitalize(), command=lambda s=style: set_brush_style(s))
    style_button.pack(side='left', padx=5, pady=5)

window.mainloop()