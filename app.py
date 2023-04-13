# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:30:14 2023

@author: kesavan
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

# Define function to perform object detection
def detect_objects(image):
    # Convert image to RGB format
    image = image.convert('RGB')
    # Perform object detection
    results = model(image)
    # Get labels and number of objects detected
    labels = results.names
    num_objects = len(results.xyxy[0])
    # Draw bounding boxes around detected objects
    draw = ImageDraw.Draw(image)
    for result in results.xyxy[0]:
        draw.rectangle(result[0:4], outline='red', width=2)
    return image, labels, num_objects

# Define Streamlit app
def app():
    st.set_page_config(page_title='YOLOv5 Object Detection', page_icon=':detective:', layout='wide')
    st.title('YOLOv5 Object Detection')

    # Upload image
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader('', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
        # Perform object detection on uploaded image
        st.sidebar.title('Detection Results')
        result_image, labels, num_objects = detect_objects(image)
        st.image(result_image, caption=f'{num_objects} objects detected', use_column_width=True)
        st.sidebar.write(f'{num_objects} objects detected:')
        for label in labels:
            st.sidebar.write(f'- {label}')

if __name__ == '__main__':
    app()
