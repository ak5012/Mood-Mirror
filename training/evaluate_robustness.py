"""Score a model on the clean test set and under fixed webcam-like conditions.

Run:  .venv\\Scripts\\python.exe training\\evaluate_robustness.py
      .venv\\Scripts\\python.exe training\\evaluate_robustness.py --model artifacts/cnn.h5

Test accuracy on clean FER2013 says nothing about how a model behaves on a dim,
soft webcam feed - the two distributions are not the same. This measures that
gap directly.

The conditions below are DETERMINISTIC, unlike the randomized `degrade_image`
used in training. Evaluation needs the same corruption applied the same way
every run, or before/after numbers are not comparable. Each condition is
seeded, so re-running reproduces the identical corrupted test set.

Note which corruptions are expected to matter. standardize() is linear, so it
removes brightness and contrast entirely - a purely darker image is, after
normalization, the same image. Only nonlinear or information-destroying
corruptions (gamma, noise, blur, resolution, clipping, JPEG) should move the
numbers. "dim_only" is included precisely to confirm that reasoning holds: if
it drops much, something is wrong with the parity assumption.
"""
import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from compat import MODEL_EXT, EMOTIONS, describe, keras
from data import load_fer2013, standardize, IMG_SIZE

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# Deterministic corruptions
# ---------------------------------------------------------------------------
def _gamma(x, gamma, gain):
    return 255.0 * gain * np.power(np.clip(x, 0, 255) / 255.0, gamma)


def _blur(x, radius):
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        pil = Image.fromarray(np.clip(x[i], 0, 255).astype(np.uint8), "L")
        out[i] = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    return out


def _downscale(x, size):
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        pil = Image.fromarray(np.clip(x[i], 0, 255).astype(np.uint8), "L")
        pil = pil.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32)
    return out


def _jpeg(x, quality):
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        buf = BytesIO()
        Image.fromarray(np.clip(x[i], 0, 255).astype(np.uint8), "L").save(
            buf, format="JPEG", quality=quality
        )
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("L"), dtype=np.float32)
    return out


def _noise(x, sigma, seed):
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, sigma, x.shape).astype(np.float32)


CONDITIONS = {
    "clean": lambda x: x,
    # Linear only. standardize() should erase this almost entirely - if it does
    # not, the parity assumption is broken somewhere.
    "dim_only": lambda x: 0.45 * x,
    "dim_gamma": lambda x: _gamma(x, 2.0, 0.55),
    "noisy": lambda x: _noise(x, 10.0, 11),
    "dim_noisy": lambda x: _noise(_gamma(x, 2.0, 0.55), 9.0, 12),
    "blurry": lambda x: _blur(x, 1.2),
    "low_res": lambda x: _downscale(x, 24),
    "jpeg": lambda x: _jpeg(x, 25),
    # The realistic composite: a dim room, a soft cheap sensor, compression.
    "webcam_hard": lambda x: _jpeg(
        _noise(_blur(_gamma(x, 1.8, 0.6), 0.9), 7.0, 13), 40
    ),
}


def evaluate(model, X, y):
    probs = model.predict(standardize(X)[..., np.newaxis], batch_size=256, verbose=0)
    return float((probs.argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ARTIFACTS_DIR / f"cnn{MODEL_EXT}"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(describe(), "\n")
    _, (X_test, y_test), _ = load_fer2013()
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Model:    {args.model}\n")

    model = keras.models.load_model(args.model)
    base = X_test.astype(np.float32)

    results = {}
    print(f"{'condition':<14} {'accuracy':>9}   {'vs clean':>9}")
    print("-" * 38)
    clean_acc = None
    for name, fn in CONDITIONS.items():
        Xc = np.clip(fn(base.copy()), 0, 255)
        acc = evaluate(model, Xc, y_test)
        results[name] = acc
        if clean_acc is None:
            clean_acc = acc
            print(f"{name:<14} {acc:>8.2%}   {'—':>9}")
        else:
            print(f"{name:<14} {acc:>8.2%}   {acc - clean_acc:>+8.2%}")

    worst = min(results.values())
    print("-" * 38)
    print(f"clean {clean_acc:.2%} | worst case {worst:.2%} | spread {clean_acc - worst:.2%}")

    out = Path(args.out) if args.out else ARTIFACTS_DIR / "robustness.json"
    out.write_text(json.dumps({"model": args.model, "conditions": results}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
