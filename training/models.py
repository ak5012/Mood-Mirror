"""Model architectures: ANN baseline and CNN.

Built with `tf_keras` (Keras 2) via compat, not `tensorflow.keras` (Keras 3).
The tensorflowjs converter can only read Keras 2 models, so a Keras 3 model
here would train perfectly and then be unexportable. See compat.py.

Every layer used below is also implemented by TF.js's Layers runtime. That is
not automatic - Keras has layers with no TF.js equivalent, and hitting one is
only discovered at conversion time. `smoke_test_export.py` proves the whole
architecture round-trips before any real training starts.
"""
from compat import keras

layers = keras.layers


def build_ann_baseline(input_shape=(48, 48, 1), num_classes=7):
    """Dense-only baseline. This is the floor the CNN has to beat."""
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="ann_baseline",
    )


def build_cnn(input_shape=(48, 48, 1), num_classes=7):
    """Three-block VGG-style CNN for FER2013.

    A third block (128 filters) is added over the original two-block design:
    at 48x48 the two-block version pools down to 12x12 and flattens 9216
    features straight into Dense(512), which puts ~4.7M of the model's
    parameters in a single fully-connected layer and overfits early. Pooling
    once more to 6x6 and using GlobalAveragePooling instead of Flatten cuts the
    model to a few hundred thousand parameters and generalizes better.
    """
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1: 48 -> 24
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            # Block 2: 24 -> 12
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.3),
            # Block 3: 12 -> 6
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.4),
            # Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn",
    )


if __name__ == "__main__":
    from compat import describe

    print(describe(), "\n")
    for build in (build_ann_baseline, build_cnn):
        m = build()
        m.summary()
        print()
