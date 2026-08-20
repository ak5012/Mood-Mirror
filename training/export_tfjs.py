"""Export trained Keras model to TensorFlow.js format."""
import json
from pathlib import Path
import tensorflow as tf
import tensorflowjs as tfjs


ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
WEB_MODEL_DIR = Path(__file__).parent.parent / "web" / "model"


def export_model(model_path: str, output_name: str = "cnn"):
    """Convert .keras model to TF.js format.

    Args:
        model_path: path to trained .keras file
        output_name: name for web output (e.g., "cnn", "ann_baseline")
    """
    # Create web/model/ directory
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load Keras model
    print(f"Loading {model_path}...")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Convert to TF.js
    output_path = WEB_MODEL_DIR / output_name
    print(f"Exporting to {output_path}/...")
    tfjs.converters.save_keras_model(model, output_path)

    # Create model.json with metadata so the browser can load it
    # (tfjs.converters already creates model.json, but we'll add our own metadata)
    metadata = {
        "format": "keras_v3",
        "modelName": output_name,
        "emotions": [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ],
        "inputShape": [1, 48, 48, 1],
        "preprocessing": {
            "method": "clahe_normalized",
            "note": "Face crop resized to 48x48, CLAHE applied, normalized to [0,1]"
        }
    }
    model_json = output_path / "model.json"
    with open(model_json) as f:
        existing = json.load(f)
    existing.update(metadata)
    with open(model_json, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"✓ Exported to web/model/{output_name}/")
    print(f"  Load in browser with:")
    print(f"  await tf.loadLayersModel('model/{output_name}/model.json');")


if __name__ == "__main__":
    model_path = ARTIFACTS_DIR / "cnn.keras"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run train.py first.")
        exit(1)
    export_model(str(model_path), "cnn")
    print("\nNow run: cd web && python -m http.server 8000")
    print("Then test in browser console: window.moodMirror.setClassifier(...)")
