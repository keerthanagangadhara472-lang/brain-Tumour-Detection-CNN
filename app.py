"""Simple Streamlit interface for the trained MRI image classifier."""
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "brain_tumour_cnn.keras"

st.set_page_config(page_title="Brain Tumour Detection", page_icon="🧠")
st.title("🧠 Brain Tumour Detection Using CNN")
st.write("Upload a brain MRI image to obtain an educational model classification.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI image", use_container_width=True)
    if st.button("Analyse MRI"):
        if not MODEL_PATH.exists():
            st.error("Trained model not found. First run `python src/train.py`.")
        else:
            resized = image.resize((150, 150))
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            score = float(load_model().predict(np.expand_dims(arr, 0), verbose=0)[0][0])
            if score >= 0.5:
                st.warning("Model result: Tumour class")
            else:
                st.success("Model result: No-tumour class")
            st.metric("Tumour-class model score", f"{score:.1%}")
            st.caption("The score is not a clinical probability or diagnosis.")

st.info("Educational demonstration only. A qualified radiologist must interpret MRI scans.")
