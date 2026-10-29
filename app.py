
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="AgriGuard Plant Disease Scanner",
    page_icon="🌿"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_plant_disease_model.keras"
MAP_PATH = BASE_DIR / "leaf-map.json"


st.write("Application directory:", str(BASE_DIR))
st.write("Model exists:", MODEL_PATH.is_file())

st.write(
    "Files available:",
    [p.name for p in BASE_DIR.iterdir()]
)

if MODEL_PATH.is_file():
    st.write(
        "Model size:",
        MODEL_PATH.stat().st_size,
        "bytes"
    )


# Same custom function used in your Colab notebook
@tf.keras.utils.register_keras_serializable()
def preprocess_input(x):
    return x

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )

@st.cache_data
def load_labels():
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()

try:
    class_map = load_labels()
except Exception:
    st.error(
        "Missing or invalid leaf-map.json. "
        "Upload the original class mapping from Colab."
    )
    st.stop()

st.title("🌿 AgriGuard Plant Disease Scanner")
st.write("Upload a clear photograph of a plant leaf.")

uploaded_file = st.file_uploader(
    "Choose a plant image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Disease", type="primary"):
        with st.spinner("Analyzing plant image..."):
            img = img.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Match original notebook preprocessing
            img_array = img_array / 255.0

            predictions = model.predict(
                img_array, verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index])

            # Preserve the notebook's class order
            if isinstance(class_map, list):
                labels = class_map
            elif isinstance(class_map, dict):
                labels = list(class_map.values())
            else:
                st.error("Unsupported class mapping format.")
                st.stop()

            if len(labels) != len(predictions):
                st.error(
                    "Class mapping does not match model outputs."
                )
                st.stop()

            label = labels[predicted_index]

            if isinstance(label, list):
                label = label[0] if label else "Unknown"

            label = str(label).split(":::")[0]
            label = label.replace("___", " - ")
            label = label.replace("_", " ")

            st.subheader(f"Prediction: {label}")
            st.metric(
                "Model confidence",
                f"{confidence:.2%}"
            )

            if confidence < 0.60:
                st.warning(
                    "Low confidence. Try another clear image."
                )

            st.caption(
                "AI prediction only. A high confidence "
                "does not guarantee a correct diagnosis."
            )
