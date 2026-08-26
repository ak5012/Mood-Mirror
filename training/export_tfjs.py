"""Export a trained Keras 2 model to the TF.js Layers format the browser loads.

Run:  .venv\\Scripts\\python.exe training\\export_tfjs.py

Two Windows-specific obstacles are handled here.

1. `pip install tensorflowjs` cannot succeed on Windows. It declares a hard
   dependency on `tensorflow-decision-forests`, which publishes Linux and macOS
   wheels only. So tensorflowjs is installed with `--no-deps`.

2. That leaves `import tensorflowjs` failing, because its package __init__
   eagerly imports every converter it ships - including the SavedModel
   converter (which imports tensorflow_decision_forests and tensorflow_hub at
   module level) and the JAX converter. None of those are on the Keras path we
   use, but Python does not care: the import runs and raises.

   `_install_optional_stubs()` registers inert placeholder modules for those
   three names before tensorflowjs is imported. They are never called - the
   Keras path goes through keras_h5_conversion only - so the stubs simply let
   the package finish importing. If a future change to this file starts using
   `convert_tf_saved_model` or `convert_jax`, the stubs will raise a clear
   error rather than silently misbehave.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from compat import MODEL_EXT, EMOTIONS, describe, keras

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
WEB_MODEL_DIR = ROOT / "web" / "model"


class _UnavailableStub(ModuleType):
    """Placeholder for a converter dependency this project does not use."""

    def __init__(self, name):
        super().__init__(name)
        # Present so the import machinery treats this as an importable package
        # rather than asking __getattr__ for it and getting a RuntimeError.
        self.__path__ = []

    def __getattr__(self, name):
        # Dunders belong to the interpreter, not to us. Raising AttributeError
        # lets normal protocol lookups fall through to their defaults; only a
        # genuine attribute access from converter code reaches the RuntimeError.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            f"{self.__name__}.{name} was accessed, but {self.__name__} is a stub "
            f"installed by export_tfjs.py (it has no Windows wheel). Only the "
            f"Keras -> TF.js Layers path is supported here."
        )


def _install_optional_stubs():
    for name in ("tensorflow_decision_forests", "tensorflow_hub"):
        sys.modules.setdefault(name, _UnavailableStub(name))

    if "jax" not in sys.modules:
        jax = _UnavailableStub("jax")
        experimental = _UnavailableStub("jax.experimental")
        jax2tf = _UnavailableStub("jax.experimental.jax2tf")
        # `from jax.experimental import jax2tf` resolves via the attribute, so
        # bind it explicitly instead of letting __getattr__ see it.
        experimental.jax2tf = jax2tf
        jax.experimental = experimental
        sys.modules["jax"] = jax
        sys.modules["jax.experimental"] = experimental
        sys.modules["jax.experimental.jax2tf"] = jax2tf


def export_model(model_path, output_name="cnn"):
    """Convert a Keras 2 .h5 model into the formats this project needs.

    Emits TWO artifacts, because the browser and the Python-side verification
    have different constraints:

    * ``web/model/<name>/`` - a **graph model**, for the browser.
    * ``artifacts/tfjs_layers/<name>/`` - a **layers model**, for parity_check.

    The browser needs a graph model specifically. The page runs tfjs as bundled
    inside @vladmandic/human, and that bundle ships tfjs-core plus tfjs-converter
    but NOT tfjs-layers - Human loads its own models via `tf.loadGraphModel`, so
    it never needed the Layers runtime. `tf.loadLayersModel` is therefore simply
    not a function on `human.tf`, and a layers-format export cannot be loaded no
    matter how correctly it was converted.

    The layers copy is kept because `tensorflowjs.converters.load_keras_model`
    can read it back into Python, which is what lets parity_check.py prove the
    conversion was lossless. Nothing in the browser reads it.
    """
    _install_optional_stubs()
    from tensorflowjs.converters import save_keras_model, convert_tf_saved_model

    model_path = Path(model_path)
    if model_path.suffix == ".keras":
        raise ValueError(
            f"{model_path.name} is a Keras 3 archive. The tensorflowjs converter "
            f"cannot read that format - retrain and save as {MODEL_EXT}. See compat.py."
        )

    print(describe())
    print(f"\nLoading {model_path} ...")
    model = keras.models.load_model(model_path)
    model.summary()

    out_dir = WEB_MODEL_DIR / output_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- graph model: what the browser loads --------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        saved_model = Path(tmp) / "saved_model"
        model.save(str(saved_model), save_format="tf")
        print(f"\nConverting to TF.js Graph format -> {out_dir} ...")
        convert_tf_saved_model(str(saved_model), str(out_dir))

    # --- layers model: Python-side verification only -------------------------
    layers_dir = ARTIFACTS_DIR / "tfjs_layers" / output_name
    if layers_dir.exists():
        shutil.rmtree(layers_dir)
    layers_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting to TF.js Layers format -> {layers_dir} (verification only) ...")
    save_keras_model(model, str(layers_dir))

    # Sidecar metadata. Deliberately a separate file: model.json is the
    # converter's own schema and merging extra keys into it risks tripping the
    # TF.js loader's validation.
    meta = {
        "modelName": output_name,
        "format": "tfjs_graph_model",
        "emotions": EMOTIONS,
        "inputShape": [1, 48, 48, 1],
        "preprocessing": {
            "method": "per_image_standardization",
            "note": (
                "Grayscale face crop resized to 48x48, then (x - mean) / (std + 1e-6) "
                "computed over that single image. Must match standardize() in "
                "training/data.py and standardize() in web/index.html."
            ),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    produced = sorted(p.name for p in out_dir.iterdir())
    total_kb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1024
    print(f"\nWrote {len(produced)} files ({total_kb:.0f} KB): {produced}")
    print(f"Browser loads it with: await tf.loadGraphModel('model/{output_name}/model.json')")
    return out_dir


if __name__ == "__main__":
    path = ARTIFACTS_DIR / f"cnn{MODEL_EXT}"
    if not path.exists():
        print(f"ERROR: {path} not found. Run training/train.py first.", file=sys.stderr)
        sys.exit(1)
    export_model(path, "cnn")
