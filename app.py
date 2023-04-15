# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:23:40 2023

@author: kesavan
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw

#from ultralytics import YOLO

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/kesavan/Desktop/AI project 1/codes/streamlit app/best.pt', force_reload=True)
#model = YOLO('C:/Users/kesavan/Desktop/AI project 1/codes/streamlit app/bestyv8.pt')
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
        center_x = result[0]
        center_y = result[1]
        width = result[2] * 0.07  # reduce width by 20%
        height = result[3] * 0.07  # reduce height by 20%
        x1 = int(center_x - width/2)
        y1 = int(center_y - height/2)
        x2 = int(center_x + width/2)
        y2 = int(center_y + height/2)
        draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
    return image, labels, num_objects

# Define Streamlit app
def app():
    st.set_page_config(page_title='YOLOv5 TMT ROD COUNTER', page_icon=':detective:', layout='wide')
    st.title('TMT ROD COUNTER')

    # Upload image
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader('', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
        result_image, labels, num_objects = detect_objects(image)
        st.image(result_image, caption=f'{num_objects} Steel bars Detected', use_column_width=True)
        # Perform object detection on uploaded image
        st.title('Detection Results')      
        st.title(f'{num_objects} Steel bars Detected')
        #for label in labels:
            #st.sidebar.write(f'- {label}')

if __name__ == '__main__':
    app()
