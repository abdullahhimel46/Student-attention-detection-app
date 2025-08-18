import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = tf.keras.models.load_model('student_engagement_model.h5')

# Classes and colors
classes = ['Not Attentive', 'Attentive']
class_colors = {
    'Not Attentive': 'red',
    'Attentive': 'green'
}

# Custom CSS
st.markdown("""
    <style>
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #306998;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction {
        font-size: 2rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .confidence {
        font-size: 1.2rem;
        color: #555555;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">🎓 Student Engagement Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or use your webcam to check student engagement</div>', unsafe_allow_html=True)

# Tabs for Upload or Webcam
tab1, tab2 = st.tabs(["Upload Image", "Live Webcam"])

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    input_arr = np.expand_dims(img_array, axis=0)
    return input_arr

with tab1:
    uploaded_file = st.file_uploader("Upload an image here 👇", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')  # <- No use_container_width or clamp here

        input_arr = preprocess_image(image)

        with st.spinner('Analyzing...'):
            prediction = model.predict(input_arr)
            class_idx = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][class_idx]
            predicted_class = classes[class_idx]

        st.markdown(f'<div class="prediction" style="color:{class_colors[predicted_class]}">Prediction: {predicted_class}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)

with tab2:
    live_image = st.camera_input("Take a selfie or point your camera to a student")

    if live_image is not None:
        image = Image.open(live_image)
        st.image(image, caption='Live Camera Capture')  # <- No use_container_width or clamp here

        input_arr = preprocess_image(image)

        with st.spinner('Analyzing live image...'):
            prediction = model.predict(input_arr)
            class_idx = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][class_idx]
            predicted_class = classes[class_idx]

        st.markdown(f'<div class="prediction" style="color:{class_colors[predicted_class]}">Prediction: {predicted_class}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <hr>
    <p style='text-align:center; color:#888; font-size:0.9rem;'>
    Made with ❤️ using Streamlit & TensorFlow
    </p>
    """, unsafe_allow_html=True
)
