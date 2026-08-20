"""Validate that Python model and TF.js model give identical predictions.

This runs BEFORE Phase 2 to make sure the export was lossless.
"""
import numpy as np
from pathlib import Path
import tensorflow as tf
import json

from data import load_fer2013, preprocess_batch, EMOTIONS


ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def parity_check_python(model_path: str, num_samples: int = 10):
    """Load Keras model, test on held-out samples, save predictions for JS comparison."""
    print("Loading FER2013 test set...")
    (_, X_test, y_test), _, _ = load_fer2013()

    print("Preprocessing...")
    X_test = preprocess_batch(X_test, augment=False)[..., np.newaxis]

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Take first num_samples
    X_sample = X_test[:num_samples]
    y_sample = y_test[:num_samples]

    print(f"\nRunning predictions on {num_samples} test images...")
    preds = model.predict(X_sample, verbose=0)

    results = []
    for i, (img, true_label, pred_probs) in enumerate(zip(X_sample, y_sample, preds)):
        pred_idx = np.argmax(pred_probs)
        pred_label = EMOTIONS[pred_idx]
        true_emotion = EMOTIONS[true_label]
        confidence = float(pred_probs[pred_idx])

        results.append({
            'index': i,
            'true_emotion': true_emotion,
            'predicted_emotion': pred_label,
            'confidence': confidence,
            'all_logits': {EMOTIONS[j]: float(pred_probs[j]) for j in range(7)},
        })

        match = "✓" if pred_idx == true_label else "✗"
        print(f"  {match} [{i}] True: {true_emotion:8} | Pred: {pred_label:8} ({confidence:.3f})")

    # Save for JS to compare
    parity_file = ARTIFACTS_DIR / "parity_check.json"
    with open(parity_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {parity_file}")
    print("Now load this in the browser and compare TF.js predictions.")
    return results


if __name__ == "__main__":
    model_path = ARTIFACTS_DIR / "cnn.keras"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run train.py first.")
        exit(1)
    parity_check_python(str(model_path), num_samples=10)
