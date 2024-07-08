import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_labels' not in st.session_state:
    st.session_state.class_labels = []

st.title('Teachable Machine Classifier')

# Model Upload Section
st.header('1. Upload Model and Set Labels')
model_file = st.file_uploader("Choose a .h5 model file", type="h5")
labels_input = st.text_input("Enter class labels (comma-separated)")

if st.button('Upload Model'):
    if model_file is not None and labels_input:
        # Load the model
        model_bytes = io.BytesIO(model_file.read())
        st.session_state.model = tf.keras.models.load_model(model_bytes)
        # Set the labels
        st.session_state.class_labels = [label.strip() for label in labels_input.split(',')]
        st.success(f"Model uploaded successfully. Labels: {', '.join(st.session_state.class_labels)}")
    else:
        st.error("Please upload a model file and enter class labels.")

# Image Classification Section
st.header('2. Classify Image')
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if st.button('Classify'):
    if st.session_state.model is None:
        st.error("Please upload a model first.")
    elif image_file is None:
        st.error("Please upload an image to classify.")
    else:
        # Preprocess the image
        img = Image.open(image_file).convert('RGB')
        img = img.resize((224, 224))  # Resize to match the input size of the model
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = st.session_state.model.predict(img_array)[0]

        # Display results
        st.subheader("Classification Results:")
        for label, prob in zip(st.session_state.class_labels, predictions):
            st.write(f"{label}: {prob:.2%}")

        # Display the image
        st.image(img, caption='Uploaded Image', use_column_width=True)

