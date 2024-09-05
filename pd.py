import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# Load the pre-trained model
model = load_model('C:/Users/UsEr/Videos/garbage_classification_model.h5')

# Define class names (update based on your model's output classes)
class_names = ['Garbage', 'Paper', 'Plastic']

# Define a function for image prediction
def predict_image(img):
    img = img.resize((150, 150))  # Resize the image to fit the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_names[predicted_class[0]]

# Streamlit app interface
st.title("Live Image Garbage Classification App")
st.write("Capture an image using your webcam to classify it into Garbage, Paper, or Plastic.")

# Capture live image input from the webcam
live_image = st.camera_input("Capture an image")

# If an image is captured
if live_image is not None:
    # Convert the captured image to a PIL image
    img = Image.open(live_image)

    # Display the captured image
    st.image(img, caption='Captured Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make a prediction
    prediction = predict_image(img)
    
    # Show the prediction result
    st.write(f"Predicted Class: {prediction}")
