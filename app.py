import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image

st.header('Flower Classification CNN Model')


flower_names = ['dandelion', 'rose', 'sunflower', 'tulip']


model = load_model('Flower_Recog_Model.h5')


def classify_images(image_file):
    
    input_image = Image.open(image_file).convert('RGB')
    input_image = input_image.resize((180, 180))  
    
    
    input_image_array = np.array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)  

    
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f"The Image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%."
    return outcome

uploaded_file = st.file_uploader('Upload an Image')

if uploaded_file is not None:
    
    st.image(uploaded_file, width=200)

    
    result = classify_images(uploaded_file)
    st.write(result)