"""
Train a binary CNN classifier for brain MRI images.
Expected folders:
data/train/tumor, data/train/no_tumor
data/test/tumor,  data/test/no_tumor

Class convention from directory names:
no_tumor -> 0, tumor -> 1 (alphabetical order).
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (150, 150)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 25

def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", color_mode="rgb", shuffle=True, seed=SEED,
        validation_split=0.2, subset="training"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", color_mode="rgb", shuffle=True, seed=SEED,
        validation_split=0.2, subset="validation"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", color_mode="rgb", shuffle=False
    )
    class_names = train_ds.class_names
    print("Class mapping:", {name: i for i, name in enumerate(class_names)})

    # Normalize pixel values to [0, 1].
    normalizer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)

    # Architecture follows the layer sequence and dimensions in the supplied paper.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="valid"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "brain_tumour_cnn.keras",
            monitor="val_loss", save_best_only=True
        )
    ]

    history = model.fit(
        train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks
    )

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().astype(int).ravel().tolist())
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    with open(OUTPUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("Test metrics:", model.evaluate(test_ds, verbose=0))
    print("Classification report saved to outputs/classification_report.json")
    print("Model saved to models/brain_tumour_cnn.keras")

if __name__ == "__main__":
    main()
