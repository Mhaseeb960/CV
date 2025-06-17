import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import tempfile
import os

# ------------------------------------------------------------------------------
# Streamlit App Styling
# ------------------------------------------------------------------------------
st.set_page_config(page_title='Disease Detection Suite', layout='centered')

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        color: #fff;
        background: #0E1117;
    }
    .css-18e3th9 {
        padding-bottom: 30px;
    }
    .stImage {
        border: 5px solid #fff;
        border-radius: 20px;
        box-shadow: 0 0 30px #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
st.sidebar.title("🧭 Disease Detection Suite")
page = st.sidebar.selectbox("Choose a detector", ["Fruit classifier 🍉", "OCT Eye Disease Detector 👁"])


# ------------------------------------------------------------------------------
# Main Section
# ------------------------------------------------------------------------------
if page == "Fruit classifier 🍉":
    st.title("🍇🍉 Fruit Disease Detector")
    st.write("Analyze your fruit images for disease or abnormalities with our trained classifier")

    # ------------------------------------------------------------------------------
    # Loading Model
    # ------------------------------------------------------------------------------
    @st.cache_resource
    def load_fruit_model():
        return load_model("my_fruit_classifier.keras")  # Update to your model's path


    model = load_fruit_model()

    # ------------------------------------------------------------------------------
    # Define class_labels matching your training data
    # ------------------------------------------------------------------------------
    class_labels = ['apple', 'banana', 'cherry', 'chickoo', 'grapes', 'kiwi', 'mango', 'orange', 'strawberry']

    # ------------------------------------------------------------------------------
    # Image Size for Model Input
    # ------------------------------------------------------------------------------
    img_size = (224, 224)


    # ------------------------------------------------------------------------------
    # Prediction function
    # ------------------------------------------------------------------------------
    def predict_fruit(image_path):
        # Prepare the image for the classifier
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)[0]

        # Determine the predicted class and confidence
        max_index = np.argmax(predictions)
        label = class_labels[max_index]
        confidence = predictions[max_index]

        return label, confidence


    # ------------------------------------------------------------------------------
    # File upload and prediction
    # ------------------------------------------------------------------------------
    uploaded_file = st.file_uploader("📁 Upload a fruit image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())  # save the upload temporarily
            temp_path = tmp.name

        st.image(temp_path, caption='Uploaded Image', use_container_width=True)

        with st.spinner("🔬 Analyzing..."):
            label, confidence = predict_fruit(temp_path)
            st.success(f"Predicted label: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}**")

        os.remove(temp_path)


# ------------------------------------------------------------------------------
# OCT Eye Disease Detector
# ------------------------------------------------------------------------------
elif page == "OCT Eye Disease Detector 👁":
    st.title("👁 OCT Eye Disease Detector — VGG16")
    st.write("Analyze your OCT retina images for abnormalities with our trained classifier")

    # ------------------------------------------------------------------------------
    # Loading OCT Model
    # ------------------------------------------------------------------------------
    @st.cache_resource
    def load_oct_model():
        return load_model("vgg16_oct_model2.keras")

    oct_model = load_oct_model()
    oct_class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    oct_img_size = (160, 160)


    # ------------------------------------------------------------------------------
    # Preprocessing and Prediction
    # ------------------------------------------------------------------------------
    def preprocess_oct_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(oct_img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.cast(img_array, tf.float32) / 255.0
        return tf.expand_dims(img_array, axis=0)


    def predict_oct(image_path):
        input_img = preprocess_oct_image(image_path)
        prediction = oct_model.predict(input_img)
        label = oct_class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        return label, confidence


    # ------------------------------------------------------------------------------
    # File upload and prediction
    # ------------------------------------------------------------------------------
    uploaded_file = st.file_uploader("📁 Upload an OCT retina image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())  # save temporarily
            temp_path = tmp.name

        st.image(temp_path, caption='Uploaded Image', use_container_width=True)

        with st.spinner("🔬 Analyzing..."):
            label, confidence = predict_oct(temp_path)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2%}**")

        os.remove(temp_path)
