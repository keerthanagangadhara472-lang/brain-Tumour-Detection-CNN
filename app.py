<<<<<<< HEAD
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
=======
from flask import Flask, render_template, request
from pathlib import Path
from werkzeug.utils import secure_filename
from predict import predict_image
from config import BASE_DIR, MODEL_PATH

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = BASE_DIR / "uploads"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp"}
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_name = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            error = "Please select an MRI image."
        elif Path(file.filename).suffix.lower() not in ALLOWED:
            error = "Allowed image formats: PNG, JPG, JPEG, BMP."
        elif not MODEL_PATH.exists():
            error = "CNN model is not trained yet. Run python train.py first."
        else:
            image_name = secure_filename(file.filename)
            path = app.config["UPLOAD_FOLDER"] / image_name
            file.save(path)
            try:
                label, confidence, raw = predict_image(path)
                result = {"label": label.replace("_", " ").title(),
                          "confidence": f"{confidence:.2%}",
                          "probability": f"{raw:.4f}"}
            except Exception as exc:
                error = f"Could not process image: {exc}"
    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 12946e920a2f7d36a126a0784b7d2f9b384c55fc
