
import streamlit as st
import tensorflow as tf
import keras
import inspect
st_version = None
from PIL import Image
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿"
)

MODEL_PATH = Path(__file__).parent / "best_plant_disease_model.keras"
CLASS_MAP_PATH = Path(__file__).parent / "leaf-map.json"


@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False
    )


@st.cache_data
def load_class_map():
    try:
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


try:
    
    st.write("TensorFlow version:", tf.__version__)
    st.write("Keras version:", keras.__version__)
    st.write(
    "GlorotUniform parameters:",
    str(inspect.signature(keras.initializers.GlorotUniform))


    model = load_disease_model()
    class_map = load_class_map()
except Exception as e:
    st.error("The model could not be loaded.")
    st.code(str(e))
    st.info(
        "Check the model file, TensorFlow version, and "
        "preprocess_input serialization."
    )
    st.stop()


st.title("🌿 Plant Disease Predictor")
st.write(
    "Upload a photo of a plant leaf to identify potential diseases."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded leaf",
            use_container_width=True
        )

        if st.button("Predict Disease", type="primary"):
            with st.spinner("Analyzing leaf..."):
                image = image.resize((224, 224))

                img_array = tf.keras.utils.img_to_array(image)
                img_array = np.expand_dims(img_array, axis=0)

                # Correct for MobileNetV2 when this matches
                # the preprocessing used during model training.
                img_array = preprocess_input(img_array)

                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))

                disease_name = f"Class {predicted_index}"

                if isinstance(class_map, dict):
                    disease_name = class_map.get(
                        str(predicted_index),
                        class_map.get(
                            predicted_index,
                            disease_name
                        )
                    )

                elif isinstance(class_map, list):
                    if 0 <= predicted_index < len(class_map):
                        disease_name = class_map[predicted_index]

                st.success("Prediction complete!")
                st.subheader(f"Prediction: {disease_name}")
                st.metric("Model confidence", f"{confidence:.2%}")

                if confidence < 0.60:
                    st.warning(
                        "The model is uncertain. Try a clearer "
                        "photo with good lighting and a visible leaf."
                    )

                st.caption(
                    "This is an AI prediction, not a confirmed "
                    "plant disease diagnosis."
                )

    except Exception as e:
        st.error("Could not process the image or make a prediction.")
        st.exception(e)
