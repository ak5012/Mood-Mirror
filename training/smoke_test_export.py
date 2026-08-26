"""Prove the Keras -> TF.js export chain works BEFORE spending hours training.

Run:  .venv\\Scripts\\python.exe training\\smoke_test_export.py

The failure this guards against is the expensive one: an architecture trains
for hours, then turns out to contain a layer the TF.js Layers runtime has no
kernel for, or gets saved in a format the converter refuses. Both are only
discovered at export time - after the compute is spent.

So: build the *real* architecture, train it for one step on random noise, run
it through the *real* export path, and load the result back. It costs a few
seconds and it exercises every format boundary the real run will hit.

Exit code 0 means the pipeline is safe to run for real.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from compat import MODEL_EXT, EMOTIONS, describe, keras
from models import build_cnn, build_ann_baseline
from data import standardize

PASS = "  [ok]  "
FAIL = "  [FAIL]"


def _check(cond, msg, detail=""):
    print((PASS if cond else FAIL) + " " + msg + (f"  ({detail})" if detail else ""))
    return bool(cond)


def main():
    print("=" * 72)
    print("EXPORT CHAIN SMOKE TEST")
    print("=" * 72)
    print(describe())
    print()

    ok = True

    # --- 1. Keras 2 is really what we are building on ------------------------
    ok &= _check(
        keras.__name__ == "tf_keras",
        "building on tf_keras (Keras 2)",
        keras.__version__,
    )

    # --- 2. Preprocessing contract ------------------------------------------
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    std = standardize(raw)
    per_img_mean = std.mean(axis=(1, 2))
    per_img_std = std.std(axis=(1, 2))
    ok &= _check(
        np.allclose(per_img_mean, 0, atol=1e-4) and np.allclose(per_img_std, 1, atol=1e-3),
        "standardize() yields per-image mean 0 / std 1",
        f"mean={per_img_mean.max():.2e} std={per_img_std.mean():.4f}",
    )

    # --- 3. Build + train one step on noise ----------------------------------
    X = std[..., np.newaxis].astype(np.float32)
    y = rng.integers(0, len(EMOTIONS), size=len(X))

    for name, build in (("cnn", build_cnn), ("ann_baseline", build_ann_baseline)):
        print(f"\n--- {name} ---")
        model = build()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(X, y, epochs=1, batch_size=2, verbose=0)
        ok &= _check(True, "trains one step", f"{model.count_params():,} params")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # --- 4. Save in the format the converter accepts ------------------
            h5 = tmp / f"{name}{MODEL_EXT}"
            model.save(h5)
            ok &= _check(h5.exists(), f"saves as {MODEL_EXT}", f"{h5.stat().st_size/1024:.0f} KB")

            # --- 5. The real export path -------------------------------------
            sys.path.insert(0, str(Path(__file__).parent))
            from export_tfjs import export_model, WEB_MODEL_DIR

            import export_tfjs

            orig = export_tfjs.WEB_MODEL_DIR
            export_tfjs.WEB_MODEL_DIR = tmp / "web"
            try:
                out = export_model(h5, name)
            except Exception as e:  # noqa: BLE001 - we want the message verbatim
                ok &= _check(False, "export to TF.js Layers", f"{type(e).__name__}: {e}")
                export_tfjs.WEB_MODEL_DIR = orig
                continue
            finally:
                export_tfjs.WEB_MODEL_DIR = orig

            # --- 6. Validate the artifact the browser will fetch -------------
            mj = out / "model.json"
            ok &= _check(mj.exists(), "model.json written")
            spec = json.loads(mj.read_text())

            ok &= _check(
                spec.get("format") == "layers-model",
                "format is 'layers-model' (tf.loadLayersModel-compatible)",
                str(spec.get("format")),
            )
            ok &= _check(
                "modelTopology" in spec and "weightsManifest" in spec,
                "has modelTopology + weightsManifest",
            )

            shards = [
                p for g in spec.get("weightsManifest", []) for p in g.get("paths", [])
            ]
            missing = [p for p in shards if not (out / p).exists()]
            ok &= _check(
                shards and not missing,
                "all weight shards present",
                f"{len(shards)} shard(s)",
            )

            # --- 7. Round-trip back through the TF.js loader -----------------
            from tensorflowjs.converters import load_keras_model

            try:
                restored = load_keras_model(str(mj))
                a = model.predict(X, verbose=0)
                b = restored.predict(X, verbose=0)
                ok &= _check(
                    np.allclose(a, b, atol=1e-5),
                    "round-trip predictions identical",
                    f"max delta {np.abs(a - b).max():.2e}",
                )
            except Exception as e:  # noqa: BLE001
                ok &= _check(False, "round-trip load", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    if ok:
        print("PASS - the export chain is sound. Safe to run training/train.py.")
    else:
        print("FAIL - fix the above before spending compute on a real run.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
