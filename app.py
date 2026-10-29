
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# ==========================================
# AGRIGUARD CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AgriGuard Plant Disease Scanner",
    page_icon="🌿",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_plant_disease_model.keras"
MAP_PATH = BASE_DIR / "leaf-map.json"


# ==========================================
# PREPROCESSING FROM COLAB
# ==========================================

@tf.keras.utils.register_keras_serializable()
def preprocess_input(x):
    return x


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )


# ==========================================
# LOAD DISEASE CLASS MAPPING
# ==========================================

@st.cache_data
def load_labels():
    with open(MAP_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🌿 AgriGuard Plant Disease Scanner")

st.write(
    "Upload a photograph of a plant leaf "
    "to identify possible crop diseases."
)


# ==========================================
# LOAD MODEL AND CLASS MAP
# ==========================================

try:
    model = load_model()

except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


try:
    class_map = load_labels()

except Exception as e:
    st.error("Unable to load leaf-map.json.")
    st.exception(e)
    st.stop()


# ==========================================
# SYSTEM DIAGNOSTICS
# ==========================================

with st.expander("🔧 System diagnostics"):

    st.write("TensorFlow version:", tf.__version__)

    st.write(
        "Model exists:",
        MODEL_PATH.is_file()
    )

    st.write(
        "Class mapping exists:",
        MAP_PATH.is_file()
    )

    st.write(
        "Model output shape:",
        model.output_shape
    )

    st.write(
        "JSON entries:",
        len(class_map)
    )

    if isinstance(class_map, dict):

        unique_labels = sorted({
            str(value[0]).split(":::")[0]
            for value in class_map.values()
            if isinstance(value, list) and value
        })

        st.write(
            "Unique disease labels:",
            len(unique_labels)
        )

        st.write(
            "Disease names:",
            unique_labels
        )


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            original_image,
            caption="Uploaded plant image",
            use_container_width=True
        )

    except Exception as e:
        st.error("Unable to open the uploaded image.")
        st.exception(e)
        st.stop()


    # ======================================
    # PREDICT DISEASE
    # ======================================

    if st.button(
        "🔍 Predict Disease",
        type="primary"
    ):

        with st.spinner("Analyzing plant image..."):

            try:

                # Resize image
                image = original_image.resize(
                    (224, 224)
                )

                # Convert to NumPy array
                img_array = tf.keras.utils.img_to_array(
                    image
                )

                # Add batch dimension
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Match Colab image scaling
                img_array = img_array / 255.0

                # Run trained model
                predictions = model.predict(
                    img_array,
                    verbose=0
                )[0]

                # Determine predicted class
                predicted_index = int(
                    np.argmax(predictions)
                )

                confidence = float(
                    predictions[predicted_index]
                )

                # ==================================
                # CHECK CLASS MAPPING
                # ==================================

                st.write(
                    "Model output classes:",
                    len(predictions)
                )

                st.write(
                    "JSON entries:",
                    len(class_map)
                )

                if isinstance(class_map, dict):

                    unique_labels = sorted({
                        str(value[0]).split(":::")[0]
                        for value in class_map.values()
                        if isinstance(value, list) and value
                    })

                    st.write(
                        "Unique disease labels:",
                        len(unique_labels)
                    )

                # Do not assume image-file mappings
                # correspond to model class indexes.

                valid_labels = None

                if isinstance(class_map, list):
                    valid_labels = class_map

                elif isinstance(class_map, dict):

                    keys = {
                        str(key)
                        for key in class_map.keys()
                    }

                    expected_keys = {
                        str(i)
                        for i in range(len(predictions))
                    }

                    if keys == expected_keys:

                        valid_labels = [
                            class_map[str(i)]
                            for i in range(len(predictions))
                        ]

                if (
                    valid_labels is None
                    or len(valid_labels) != len(predictions)
                ):

                    st.error(
                        "The class mapping does not match "
                        "the trained model's output indexes."
                    )

                    st.warning(
                        "Your leaf-map.json may contain "
                        "image filenames rather than the "
                        "38 ordered model class labels. "
                        "We need the original training "
                        "class order to display accurate "
                        "disease names."
                    )

                    st.write(
                        "Predicted class index:",
                        predicted_index
                    )

                    st.write(
                        "Raw model score:",
                        confidence
                    )

                    st.stop()

                # ==================================
                # FORMAT DISEASE NAME
                # ==================================

                disease_name = valid_labels[
                    predicted_index
                ]

                if isinstance(disease_name, list):
                    disease_name = (
                        disease_name[0]
                        if disease_name
                        else "Unknown"
                    )

                disease_name = str(
                    disease_name
                ).split(":::")[0]

                disease_name = disease_name.replace(
                    "___", " - "
                )

                disease_name = disease_name.replace(
                    "_", " "
                )

                # ==================================
                # DISPLAY PREDICTION
                # ==================================

                st.success("Prediction complete!")

                st.subheader(
                    f"🌱 Prediction: {disease_name}"
                )

                st.metric(
                    "Model confidence",
                    f"{confidence:.2%}"
                )

                if confidence < 0.60:

                    st.warning(
                        "The model is uncertain. "
                        "Try another clear photograph "
                        "with good lighting."
                    )

                st.caption(
                    "This is an AI-generated prediction, "
                    "not a confirmed plant disease diagnosis. "
                    "A high confidence score does not "
                    "guarantee accuracy."
                )

            except Exception as e:

                st.error(
                    "An error occurred while "
                    "analyzing the image."
                )

                st.exception(e)
