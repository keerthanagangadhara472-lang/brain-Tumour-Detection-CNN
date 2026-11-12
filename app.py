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
