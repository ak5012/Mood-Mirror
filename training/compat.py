"""Framework-compatibility contract for this project. Import this FIRST, always.

The browser loads the model with `tf.loadLayersModel(...)`, which reads the
TF.js *Layers* format. That format is produced only by the tensorflowjs
converter's Keras path, and that path is built on `tf_keras` — the Keras **2**
compatibility package. Keras 3 (the default in TensorFlow >= 2.16) writes a
`.keras` v3 archive that the converter cannot read at all.

So the whole chain is pinned by its weakest link:

    tf_keras (Keras 2)  ->  model.h5  ->  tensorflowjs_converter  ->  tf.loadLayersModel

Training against `tensorflow.keras` (Keras 3) produces a model that trains fine
and then cannot be exported. That failure surfaces *after* the training run,
which is the expensive place to find it. This module makes it surface at import.

Three separate incompatibilities are checked here:

  1. Python 3.14 has no TensorFlow wheels (cp313 is the newest TF publishes).
  2. Keras 3 output cannot be converted to the TF.js Layers format.
  3. `pip install tensorflowjs` fails on Windows, because it depends on
     tensorflow-decision-forests, which ships Linux/macOS wheels only. Install
     it with `--no-deps` plus the handful of deps the Keras path actually uses.
"""
import os
import sys

# Must be set before TensorFlow is imported anywhere in the process, otherwise
# `tf.keras` binds to Keras 3 and the export path is silently lost.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quiet the startup spam

MAX_SUPPORTED_PY = (3, 13)


def _fail(msg: str) -> "NoReturn":
    print("\n" + "=" * 72, file=sys.stderr)
    print("INCOMPATIBLE ENVIRONMENT", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(msg.strip(), file=sys.stderr)
    print("=" * 72 + "\n", file=sys.stderr)
    sys.exit(1)


if sys.version_info[:2] > MAX_SUPPORTED_PY:
    _fail(rf"""
Python {sys.version_info.major}.{sys.version_info.minor} is running, but TensorFlow
publishes no wheels past cp{MAX_SUPPORTED_PY[0]}{MAX_SUPPORTED_PY[1]}. `pip install tensorflow` cannot
succeed here — there is no wheel to install and no amount of pinning fixes it.

Use the project virtualenv, which is built on Python 3.13:

    .venv\Scripts\python.exe training\train.py
""")

try:
    import tensorflow as tf
except ModuleNotFoundError:
    _fail(r"""
TensorFlow is not installed in this interpreter. You are probably running the
system Python instead of the project virtualenv. Use:

    .venv\Scripts\python.exe training\train.py
""")

try:
    import tf_keras as keras
except ModuleNotFoundError:
    _fail(r"""
`tf_keras` is missing. It is the Keras 2 package, and it is what the
tensorflowjs converter builds on — Keras 3 models cannot be exported to the
TF.js Layers format that this project's browser code loads.

    .venv\Scripts\python.exe -m pip install "tf-keras==2.20.1"
""")

# Guard against the subtle version-skew failure: tf_keras must be built for the
# TensorFlow it is running on, or layer deserialization breaks in ways that only
# appear at load/convert time.
_tf_mm = tf.__version__.split(".")[:2]
_k_mm = keras.__version__.split(".")[:2]
if _tf_mm != _k_mm:
    _fail(rf"""
Version skew: tensorflow {tf.__version__} with tf_keras {keras.__version__}.
These must share a major.minor version, or saved models fail to deserialize
during conversion. Reinstall a matched pair, e.g.:

    .venv\Scripts\python.exe -m pip install "tensorflow==2.20.0" "tf-keras==2.20.1"
""")

# The model format the converter accepts. Keras 2 HDF5 — deliberately not
# `.keras`, which is the Keras 3 archive format the converter rejects.
MODEL_EXT = ".h5"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def describe() -> str:
    return (
        f"python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} | "
        f"tensorflow {tf.__version__} | tf_keras {keras.__version__} (Keras 2) | "
        f"export target: TF.js Layers"
    )


__all__ = ["tf", "keras", "EMOTIONS", "MODEL_EXT", "describe"]
