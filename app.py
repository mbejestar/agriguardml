import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")

@st.cache_resource
def load_disease_model():
    # Loads the specific Keras model from your directory
    return tf.keras.models.load_model("best_plant_disease_model.keras")

@st.cache_data
def load_class_map():
    # Attempts to load the mapping file to translate indices to names
    try:
        with open("leaf-map.json", "r") as f:
            return json.load(f)
    except Exception:
        return None

model = load_disease_model()
class_map = load_class_map()

st.title("🌿 Plant Disease Predictor")
st.write("Upload a photo of a plant leaf to identify potential diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            # Resize to match your model's expected input (adjust 224 to your model's input shape)
            img_resized = image.resize((224, 224))
            
            # Convert to array and expand dimensions for the model batch size
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Scale pixels if your model was trained with normalized data (e.g., 0-1)
            # img_array = img_array / 255.0 
            
            predictions = model.predict(img_array)
            predicted_index = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions))
            
            st.success("Prediction Complete!")
            st.write(f"**Predicted Class Index:** {predicted_index}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            if class_map:
                st.write("*(Use your leaf-map.json data to map this index to the exact disease name)*")
