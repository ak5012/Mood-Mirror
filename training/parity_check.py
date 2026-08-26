"""Verify the exported model predicts identically to the Python model.

Run:  .venv\\Scripts\\python.exe training\\parity_check.py

Two levels of checking happen here.

1. **Python vs converted model.** The .h5 is reloaded through the tensorflowjs
   loader and both are run on the same inputs. This catches conversion damage.

2. **Browser fixture.** A JSON fixture is written to web/model/parity_check.json
   holding the raw 48x48 uint8 pixels and the expected probability vector for
   each sample. In the browser, run

       await window.moodMirror.verifyParity()

   which feeds those exact pixels through the TF.js model and reports the max
   deviation. That is the only check that exercises the real runtime, the real
   JS standardize(), and the real weights together.

The original version of this file began with

    (_, X_test, y_test), _, _ = load_fer2013()

which unpacks a 2-tuple into three names and raises ValueError before doing any
work. It has never run successfully.
"""
import json
import sys
from pathlib import Path

import numpy as np

from compat import MODEL_EXT, EMOTIONS, describe, keras
from data import load_fer2013, standardize

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
WEB_MODEL_DIR = ROOT / "web" / "model"

TOL = 1e-5


def parity_check(model_path, num_samples=24, seed=0):
    print(describe(), "\n")

    print("Loading FER2013 test split ...")
    _, (X_test, y_test), _ = load_fer2013()

    # Sample across classes rather than taking the first N, which in the folder
    # release would all be 'angry' and would not exercise the output layer.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    raw = X_test[idx]
    truth = y_test[idx]

    X = standardize(raw)[..., np.newaxis]

    print(f"Loading {model_path} ...")
    model = keras.models.load_model(model_path)
    probs = model.predict(X, verbose=0)

    # --- level 1: reload through the tensorflowjs loader --------------------
    # Reads the *layers* copy in artifacts/, not web/model/cnn/. The browser
    # copy is a graph model (Human's bundled tfjs has no tfjs-layers), and
    # load_keras_model cannot read that format. Both are converted from the same
    # .h5 in the same run, so checking the layers copy still validates the
    # conversion; the graph copy is verified against the real tfjs runtime by
    # the Node harness and by verifyParity() in the browser.
    tfjs_dir = ARTIFACTS_DIR / "tfjs_layers" / "cnn"
    delta = None
    if (tfjs_dir / "model.json").exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from export_tfjs import _install_optional_stubs

        _install_optional_stubs()
        from tensorflowjs.converters import load_keras_model

        print(f"Reloading converted model from {tfjs_dir} ...")
        restored = load_keras_model(str(tfjs_dir / "model.json"))
        probs2 = restored.predict(X, verbose=0)
        delta = float(np.abs(probs - probs2).max())
        status = "OK" if delta <= TOL else "MISMATCH"
        print(f"  Python vs converted: max delta {delta:.3e}  [{status}]")
    else:
        print(f"  (skipped: {tfjs_dir}/model.json not found - run export_tfjs.py)")

    # --- report -------------------------------------------------------------
    pred = probs.argmax(axis=1)
    correct = int((pred == truth).sum())
    print(f"\nSample predictions ({correct}/{len(truth)} correct):")
    for i in range(len(truth)):
        mark = "OK " if pred[i] == truth[i] else "  x"
        print(
            f"  {mark} true={EMOTIONS[truth[i]]:9} pred={EMOTIONS[pred[i]]:9} "
            f"p={probs[i][pred[i]]:.3f}"
        )

    # --- level 2: browser fixture ------------------------------------------
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fixture = {
        "note": (
            "Raw uint8 48x48 pixels and the probabilities the Python model "
            "produces for them. Run window.moodMirror.verifyParity() in the "
            "browser to compare against the TF.js runtime."
        ),
        "classOrder": EMOTIONS,
        "tolerance": 1e-3,
        "samples": [
            {
                "trueLabel": EMOTIONS[int(truth[i])],
                "pixels": raw[i].reshape(-1).astype(int).tolist(),
                "expected": [float(v) for v in probs[i]],
            }
            for i in range(len(truth))
        ],
    }
    out = WEB_MODEL_DIR / "parity_check.json"
    out.write_text(json.dumps(fixture))
    print(f"\nBrowser fixture -> {out}  ({out.stat().st_size / 1024:.0f} KB)")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "parity_check.json").write_text(
        json.dumps(
            {
                "python_vs_converted_max_delta": delta,
                "accuracy_on_sample": correct / len(truth),
                "num_samples": len(truth),
            },
            indent=2,
        )
    )

    if delta is not None and delta > TOL:
        print("\nFAIL: conversion changed the model's outputs.")
        return 1
    print("\nNext: open http://localhost:8000, click 'Load custom CNN', then run")
    print("      await window.moodMirror.verifyParity()")
    return 0


if __name__ == "__main__":
    path = ARTIFACTS_DIR / f"cnn{MODEL_EXT}"
    if not path.exists():
        print(f"ERROR: {path} not found. Run training/train.py first.", file=sys.stderr)
        sys.exit(1)
    sys.exit(parity_check(path))
