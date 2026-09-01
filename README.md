# Mood Mirror

Real-time facial emotion detection in the browser. Webcam → face detection →
CNN classification → emoji overlay, with the model trained from scratch on FER2013.

> **Build status:** Phase 1 complete. The self-trained CNN is trained, exported
> to TF.js, and running in the browser. Every number below comes from an actual
> run — see `artifacts/` for the metrics JSON and confusion matrices.

## Running the shell

The browser blocks webcam access on `file://`, so this must be served over
`localhost`:

```bash
cd web
python -m http.server 8000    # or: npx serve .
```

Then open <http://localhost:8000>.

## Current state

| Component | Status |
|---|---|
| Webcam capture + face detection | done (Human library, detector only) |
| Emotion classification | **self-trained CNN** (pretrained still toggleable for comparison) |
| Emoji overlay + bounding box | done |
| Smoothing (8-frame majority vote) | done |
| Latency / stability instrumentation | done |
| Webcam-condition robustness | done — see robustness table |
| Face alignment to FER2013 geometry | done |
| Confidence gating + probability smoothing | done |
| Self-trained CNN | done — 63.6% test accuracy, macro F1 0.599 |
| Backend + history | not started |
| Deployment | not started |

## Results

Held-out test set: 7,178 FER2013 images, never seen during training or
validation. Trained on CPU (Python 3.13 / TF 2.20 / Keras 2).

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline (always `happy`) | 24.7% | — |
| ANN baseline (25 epochs) | 40.3% | — |
| **CNN (self-trained, 70 epochs)** | **63.6%** | **0.599** |

The CNN beats the dense baseline by **+23.3 points** on identical data and
preprocessing, which is the point of keeping the baseline around.

Per-class, the CNN (precision / recall / F1):

| Class | Precision | Recall | F1 | Note |
|---|---|---|---|---|
| happy | 0.846 | 0.859 | 0.852 | strongest — most training data |
| surprise | 0.711 | 0.816 | 0.760 | visually distinctive |
| neutral | 0.553 | 0.665 | 0.604 | |
| angry | 0.528 | 0.601 | 0.562 | |
| disgust | 0.425 | 0.667 | 0.519 | only 393 training images |
| sad | 0.557 | 0.428 | 0.484 | |
| fear | 0.497 | 0.349 | 0.410 | weakest — confused with sad/surprise |

### What moved the per-class numbers

An earlier model used plain inverse-frequency class weighting and selected its
best epoch on `val_accuracy`. That combination broke three of seven classes
while overall accuracy still looked reasonable: `disgust` was predicted far too
often (precision 0.269) and `angry`/`fear` were barely predicted at all.

Four changes, applied together:

- **`sqrt` class weighting** — `happy` has 6,494 training images against
  `disgust`'s 393, so inverse weighting makes one disgust example count 16.5x a
  happy one. `1/sqrt(freq)` cuts that spread to 4.1x.
- **Checkpoint on macro F1, not accuracy** — selecting on accuracy rewards
  ignoring rare classes outright. Macro F1 weights every class equally.
- **Label smoothing (0.05)** — FER2013's labels are noisy enough that a hard
  1.0 target trains the model to be confident about noise.
- **Cosine LR decay** replacing `ReduceLROnPlateau`.

Every class improved and none regressed:

| Class | F1 before | F1 after | Change |
|---|---|---|---|
| disgust | 0.394 | 0.519 | **+0.125** |
| sad | 0.441 | 0.484 | +0.043 |
| fear | 0.374 | 0.410 | +0.036 |
| angry | 0.539 | 0.562 | +0.023 |
| neutral | 0.593 | 0.604 | +0.011 |
| surprise | 0.756 | 0.760 | +0.004 |
| happy | 0.852 | 0.852 | +0.000 |
| **Macro F1** | 0.564 | **0.599** | **+0.035** |
| **Accuracy** | 61.7% | **63.6%** | **+1.9** |

`fear` and `sad` traded a little precision for substantially more recall — 148
more faces correctly identified across the two. Both remain the weakest classes.

### On the accuracy ceiling

Human agreement on FER2013 is roughly 65%, but that is **not** a cap on model
accuracy: published work reaches 73.28% (Khaireddin & Chen, 2021, VGGNet) and
the 2013 challenge winner scored 71.16%. Models learn annotator consistencies
that individual humans do not. So 63.6% leaves real headroom on this dataset —
reaching it needs a much larger network plus heavier augmentation, which in
turn needs a GPU. 90%+ remains out of reach on FER2013.

### Robustness to webcam conditions

Clean test accuracy does not predict webcam accuracy - FER2013 is evenly lit
and sharp, a real webcam feed is neither. `training/evaluate_robustness.py`
measures that gap with fixed, seeded corruptions.

| Condition | Before | After | Change |
|---|---|---|---|
| clean | 61.67% | 63.56% | +1.9 |
| dim_only (linear darkening) | 61.67% | 63.56% | +1.9 |
| dim_gamma | 60.63% | 63.32% | +2.7 |
| jpeg (q25) | 51.70% | 61.74% | **+10.0** |
| noisy (σ10) | 42.11% | 61.99% | **+19.9** |
| blurry (r1.2) | 41.88% | 59.61% | **+17.7** |
| low_res (24px) | 41.04% | 59.19% | **+18.2** |
| dim + noisy | 32.07% | 59.70% | **+27.6** |
| **webcam_hard** (combined) | **31.65%** | **57.27%** | **+25.6** |

Clean-to-worst-case spread went from **30.0 points to 6.3**. Under realistic
dim-and-noisy conditions the model went from near-useless to usable.

Two things follow from this.

**Low light by itself is not the problem.** `dim_only` moves the number by
exactly 0.00%, because per-image standardization is linear and removes
brightness completely. That is also a clean empirical confirmation that the
Python/browser parity contract holds.

**Noise, blur and resolution loss are the problem** - which is what a cheap
sensor produces *in* low light. Under realistic combined conditions the model
loses roughly half its accuracy.

`degrade_image()` in `data.py` trains against exactly these corruptions
(gamma, black crush, blur, downscale, sensor noise, JPEG), applied per batch
with 35% of images left clean so clean-image accuracy is not traded away.

### Export verification

The Keras → TF.js chain is verified end to end, not assumed:

| Check | Result |
|---|---|
| `smoke_test_export.py` round-trip | max delta `0.00e+00` |
| Python vs converted model (`parity_check.py`) | max delta `0.000e+00` |
| Real TF.js 4.22 runtime vs Python | max delta `2.98e-7` (float32 rounding) |
| Exported format | `layers-model`, keras v2.20.1, converter v4.22.0 |
| Model size in browser | 1.25 MB (24 layers, 324,071 params) |

| Runtime metric | Value | Test device |
|---|---|---|
| Inference latency (avg / p95) | — | — |
| Effective FPS | — | — |
| Flicker reduction from smoothing | — | — |
| Guided-prompt real-world accuracy | — | — |

## Known limitations

- FER2013 is noisy; human agreement on it is roughly 65%, so that is the
  realistic accuracy ceiling, not 90%+.
- `disgust` is severely under-represented (~550 examples vs ~9,000 for `happy`)
  and needs class weighting to avoid being silently never predicted.
- Browser inference speed varies by device; all latency figures are reported
  with the test device named.

## Privacy

No video frames or images ever leave the browser. Only the resulting label,
confidence, and timestamp are sent to the backend.
