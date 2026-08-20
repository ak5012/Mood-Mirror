"""Train ANN baseline and CNN on FER2013. Save models and metrics."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import tensorflow as tf
from tensorflow import keras

from data import load_fer2013, preprocess_batch, EMOTIONS
from models import build_ann_baseline, build_cnn


OUTPUT_DIR = Path(__file__).parent.parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train and return history."""
    # Class weights to handle imbalance (disgust has ~550 examples vs ~9000 for happy)
    class_weights = {}
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    for cls, cnt in zip(unique, counts):
        class_weights[cls] = total / (len(unique) * cnt)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=1,
    )
    return history


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate on test set, save metrics."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=EMOTIONS, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} Test Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

    # Save confusion matrix as image
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} Confusion Matrix\nTest Accuracy: {acc:.4f}')
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=100)
    plt.close()

    # Save JSON metrics
    metrics = {
        'model': model_name,
        'test_accuracy': float(acc),
        'per_class_precision': {e: float(report[e]['precision']) for e in EMOTIONS},
        'per_class_recall': {e: float(report[e]['recall']) for e in EMOTIONS},
        'per_class_f1': {e: float(report[e]['f1-score']) for e in EMOTIONS},
    }
    (OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    return acc, report, cm


def plot_training_history(history, model_name):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train loss')
    ax1.plot(history.history['val_loss'], label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_training.png", dpi=100)
    plt.close()


def main():
    print("Loading FER2013...")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_fer2013()

    print("\nPreprocessing (CLAHE)...")
    X_train = preprocess_batch(X_train, augment=True)
    X_test = preprocess_batch(X_test, augment=False)
    X_val = preprocess_batch(X_val, augment=False)

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # Baseline
    print("\n" + "="*60)
    print("Training ANN Baseline")
    print("="*60)
    ann = build_ann_baseline()
    ann_history = train_model(ann, X_train, y_train, X_val, y_val, epochs=30)
    ann_acc, ann_report, ann_cm = evaluate_model(ann, X_test, y_test, "ANN Baseline")
    plot_training_history(ann_history, "ANN Baseline")
    ann.save(OUTPUT_DIR / "ann_baseline.keras")

    # CNN
    print("\n" + "="*60)
    print("Training CNN")
    print("="*60)
    cnn = build_cnn()
    cnn_history = train_model(cnn, X_train, y_train, X_val, y_val, epochs=50)
    cnn_acc, cnn_report, cnn_cm = evaluate_model(cnn, X_test, y_test, "CNN")
    plot_training_history(cnn_history, "CNN")
    cnn.save(OUTPUT_DIR / "cnn.keras")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Majority-class baseline (always predict 'happy'):  {np.sum(y_test == 3) / len(y_test):.4f}")
    print(f"ANN test accuracy:                                 {ann_acc:.4f}")
    print(f"CNN test accuracy:                                 {cnn_acc:.4f}")
    print(f"Improvement (CNN vs ANN):                          {cnn_acc - ann_acc:+.4f}")
    print("\nArtifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
