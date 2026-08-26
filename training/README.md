# Training Pipeline

Trains the emotion CNN on FER2013 and exports it to the TF.js Layers format the
browser loads.

## The compatibility contract (read this first)

The browser calls `tf.loadLayersModel(...)`. That reads the **TF.js Layers**
format, which is produced only by the tensorflowjs converter's Keras path,
which is built on **Keras 2** (`tf_keras`). The whole chain is pinned by that:

```
tf_keras (Keras 2)  ->  cnn.h5  ->  tensorflowjs converter  ->  tf.loadLayersModel
```

Three ways this breaks, all of which this pipeline now guards against:

| Trap | Symptom | Handled by |
|---|---|---|
| **Python 3.14** | `pip install tensorflow` finds no wheel — TF publishes nothing past cp313 | `.venv` is built on **Python 3.13**; `compat.py` refuses to run on 3.14+ |
| **Keras 3** | Trains fine, then the converter cannot read the `.keras` file | `compat.py` sets `TF_USE_LEGACY_KERAS=1` and everything imports `keras` from it; models save as `.h5` |
| **Windows + tensorflowjs** | `pip install tensorflowjs` fails on `tensorflow-decision-forests` (Linux/macOS wheels only) | installed with `--no-deps`; `export_tfjs.py` stubs the unused converters |

`compat.py` fails loudly at import if any of these is violated, so the problem
surfaces in one second instead of after a training run.

## Setup

Already done in this checkout, but to rebuild from scratch:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install "tensorflow==2.20.0" "tf-keras==2.20.1" "numpy<2.2"
.venv\Scripts\python.exe -m pip install --no-deps "tensorflowjs==4.22.0"
.venv\Scripts\python.exe -m pip install importlib_resources six scikit-learn matplotlib pillow kaggle
```

TensorFlow and tf-keras must share a major.minor version or saved models fail to
deserialize during conversion. `compat.py` checks this.

> Windows note: TensorFlow dropped native GPU support after 2.10, so training
> here is CPU-only regardless of your GPU.

## Workflow

```powershell
# 0. Prove the export chain works — seconds, run before any long training
.venv\Scripts\python.exe training\smoke_test_export.py

# 1. Data (needs ~/.kaggle/kaggle.json)
.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.dataset_download_files('msambare/fer2013', path='data', unzip=True)"
.venv\Scripts\python.exe training\data.py          # sanity-check shapes and class counts

# 2. Train  (--quick runs 3 epochs to validate the loop)
.venv\Scripts\python.exe training\train.py

# 3. Export to the browser
.venv\Scripts\python.exe training\export_tfjs.py

# 4. Verify Python and TF.js agree
.venv\Scripts\python.exe training\parity_check.py

# 5. Try it live
cd web; python -m http.server 8000
#   -> click "Load custom CNN", then in the console:
#      await window.moodMirror.verifyParity()
```

## Preprocessing parity

This is the quiet killer. Whatever happens to a training image must happen
identically to a webcam crop, or test accuracy tells you nothing about real
accuracy. The contract:

```
grayscale  ->  48x48  ->  (x - mean) / (std + 1e-6)     [per image]
```

Implemented in three places that must stay in sync:

- `data.py` → `standardize()`
- `train.py` → `ImageDataGenerator(samplewise_center, samplewise_std_normalization)`
  (asserted equal to `standardize()` at startup)
- `web/index.html` → `standardize()` / `standardizeCrop()`

Per-image normalization — rather than a plain `/255` — is what makes the model
tolerate a dim or washed-out webcam: each frame's own brightness and contrast
are divided out before the CNN sees it.

The earlier pipeline applied CLAHE in Python with **no browser counterpart**, so
every model trained under it was scored on differently-distributed data at
inference time. CLAHE was removed rather than reimplemented in JS, because
per-image standardization achieves the same lighting invariance in five lines
that are trivially identical in both languages.

## Label order

The canonical order is alphabetical, matching the image-folder release:

```
angry, disgust, fear, happy, neutral, sad, surprise
```

The original **CSV** release numbers them differently (`4=sad, 5=surprise,
6=neutral`). `data.py::_CSV_TO_CANONICAL` remaps it. Loading the CSV without
that remap trains a model that is confidently wrong on three of seven classes,
and the loss curve looks completely normal while it happens.

## Files

| File | Role |
|---|---|
| `compat.py` | Framework contract + fail-fast environment guard. Import first. |
| `data.py` | FER2013 loading (both releases), label remap, augmentation, `standardize()` |
| `models.py` | ANN baseline and the 3-block CNN |
| `train.py` | Training loop, class weighting, callbacks, evaluation, plots |
| `export_tfjs.py` | Keras 2 `.h5` → TF.js Layers, with the Windows stubs |
| `parity_check.py` | Python vs converted model, plus the browser fixture |
| `smoke_test_export.py` | Full export round-trip in seconds — run before long jobs |

## Expected results

FER2013 is genuinely hard; human agreement is roughly 65%, so that is the
realistic ceiling, not 90%+.

| Metric | Expected |
|---|---|
| Majority-class baseline | ~25% |
| ANN baseline | ~48-55% |
| CNN | ~60-66% |

`disgust` has only 436 training images against `happy`'s 7,215. Inverse-frequency
class weights (in `train.py::class_weights_for`) keep it from being silently
never predicted — watch its per-class recall in the metrics JSON, not just the
overall accuracy.
