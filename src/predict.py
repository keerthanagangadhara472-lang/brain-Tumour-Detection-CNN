"""Predict tumour/no-tumour for one MRI image."""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "brain_tumour_cnn.keras"
CLASS_NAMES = ["no_tumor", "tumor"]  # alphabetical folder order

def predict(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py")
    image = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    array = tf.keras.utils.img_to_array(image) / 255.0
    probability = float(tf.keras.models.load_model(MODEL_PATH).predict(
        np.expand_dims(array, axis=0), verbose=0
    )[0][0])
    label = "Tumour detected" if probability >= 0.5 else "No tumour detected"
    print(f"Result: {label}")
    print(f"Tumour model score: {probability:.2%}")
    print("This is an educational model output, not a medical diagnosis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an MRI image")
    args = parser.parse_args()
    predict(args.image)
