import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import TEST_DIR, IMAGE_SIZE, BATCH_SIZE, ARTIFACTS_DIR, MODEL_PATH

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Train first with: python train.py")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, class_names=["no_tumor", "tumor"], image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, label_mode="binary", shuffle=False
    ).map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)
    model = tf.keras.models.load_model(MODEL_PATH)
    y_true, y_prob = [], []
    for x, y in test_ds:
        y_true.extend(y.numpy().reshape(-1).astype(int))
        y_prob.extend(model.predict(x, verbose=0).reshape(-1))
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, target_names=["no_tumor", "tumor"], digits=4)
    cm = confusion_matrix(y_true, y_pred)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n\n{report}\n")
        f.write(f"Confusion matrix (rows=true, columns=predicted):\n{cm}\n")
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.xticks([0, 1], ["No tumour", "Tumour"])
    plt.yticks([0, 1], ["No tumour", "Tumour"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
    print(report)
    print("Saved report and confusion matrix in artifacts/")

if __name__ == "__main__":
    main()
