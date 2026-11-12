import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from config import IMAGE_SIZE, MODEL_PATH

def predict_image(path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Train first: python train.py")
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    model = tf.keras.models.load_model(MODEL_PATH)
    probability = float(model.predict(arr[None, ...], verbose=0)[0][0])
    label = "tumor" if probability >= 0.5 else "no_tumor"
    confidence = probability if label == "tumor" else 1.0 - probability
    return label, confidence, probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict binary MRI class with CNN.")
    parser.add_argument("--image", required=True, help="Path to MRI image")
    args = parser.parse_args()
    label, confidence, p = predict_image(args.image)
    print(f"Prediction: {label}")
    print(f"Confidence (model score): {confidence:.2%}")
    print(f"Raw tumour probability: {p:.4f}")
    print("Educational output only; not a clinical diagnosis.")
