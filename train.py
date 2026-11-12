import json
import tensorflow as tf
import matplotlib.pyplot as plt
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, SEED, ARTIFACTS_DIR, MODEL_PATH, CLASS_NAMES_PATH
from model import build_cnn

def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folders missing. Expected {TRAIN_DIR} and {TEST_DIR}. "
            "Read README.md for folder structure."
        )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, class_names=["no_tumor", "tumor"], image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, label_mode="binary", shuffle=True, seed=SEED
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, class_names=["no_tumor", "tumor"], image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, label_mode="binary", shuffle=False
    )
    # Reserve 20% of the training batches for validation.
    batches = tf.data.experimental.cardinality(train_ds).numpy()
    val_count = max(1, int(batches * 0.2))
    val_ds = train_ds.take(val_count)
    fit_ds = train_ds.skip(val_count)
    norm = tf.keras.layers.Rescaling(1./255)
    fit_ds = fit_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

    model = build_cnn()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    history = model.fit(fit_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump({"0": "no_tumor", "1": "tumor"}, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "training_history.png", dpi=160)
    print(f"Model saved to: {MODEL_PATH}")
    print("Run python evaluate.py for held-out test metrics.")

if __name__ == "__main__":
    main()
