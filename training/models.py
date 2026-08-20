"""Model architectures: ANN baseline and CNN."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_ann_baseline(input_shape=(48, 48, 1), num_classes=7):
    """Simple dense-only baseline. Flattens the image, passes through 2 dense layers.

    This is the floor — we expect the CNN to beat this by a meaningful margin.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ], name='ann_baseline')
    return model


def build_cnn(input_shape=(48, 48, 1), num_classes=7):
    """CNN model for FER2013.

    Architecture:
      - Two conv blocks: 32/64 filters, 3x3 kernels, max pooling
      - Batch normalization and dropout for regularization
      - Two dense layers before the output
      - Total params: ~250k (reasonable for 48x48 input)
    """
    model = keras.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name='cnn')
    return model


if __name__ == "__main__":
    ann = build_ann_baseline()
    cnn = build_cnn()
    print("ANN Baseline:")
    ann.summary()
    print("\nCNN:")
    cnn.summary()
