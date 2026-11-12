import tensorflow as tf
from config import IMAGE_SIZE

def build_cnn(input_shape=(*IMAGE_SIZE, 3)):
    # CNN layers and parameter counts mirror the architecture table in the supplied paper.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(96, kernel_size=(3, 3), activation="relu", padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="brain_tumor_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model

if __name__ == "__main__":
    build_cnn().summary()
