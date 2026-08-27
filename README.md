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
| Face alignment to FER2013 geometry | done |
| Confidence gating + probability smoothing | done |
| Self-trained CNN | done — 61.7% test accuracy |
| Backend + history | not started |
| Deployment | not started |

## Results

Held-out test set: 7,178 FER2013 images, never seen during training or
validation. Trained on CPU (Python 3.13 / TF 2.20 / Keras 2).

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline (always `happy`) | 24.7% | — |
| ANN baseline (30 epochs) | 29.0% | 0.242 |
| **CNN (self-trained, 60 epochs)** | **61.7%** | **0.564** |

The CNN beats the dense baseline by **+32.6 points** on identical data and
preprocessing, which is the point of keeping the baseline around.

Per-class, the CNN (precision / recall):

| Class | Precision | Recall | Note |
|---|---|---|---|
| happy | 0.880 | 0.825 | strongest — most training data |
| surprise | 0.696 | 0.827 | visually distinctive |
| neutral | 0.496 | 0.736 | over-predicted; absorbs fear/sad |
| angry | 0.515 | 0.567 | |
| sad | 0.571 | 0.359 | under-predicted |
| fear | 0.532 | 0.288 | weakest recall; confused with sad/surprise |
| disgust | 0.269 | 0.739 | over-predicted — class weights too aggressive |

**Known issue:** macro F1 (0.564) sits well below accuracy (0.617), the
signature of the inverse-frequency class weighting over-correcting. `disgust`
is predicted far too often (precision 0.269 on just 111 test images) while
`fear` and `sad` stay too conservative. Softening the weights to
`sqrt(inverse-frequency)` is the standard fix and is the next thing to try.

### Robustness to webcam conditions

Clean test accuracy does not predict webcam accuracy - FER2013 is evenly lit
and sharp, a real webcam feed is neither. `training/evaluate_robustness.py`
measures that gap with fixed, seeded corruptions.

Baseline (model trained **without** webcam-condition augmentation):

| Condition | Accuracy | vs clean |
|---|---|---|
| clean | 61.67% | — |
| dim_only (linear darkening) | 61.67% | **+0.00%** |
| dim_gamma | 60.63% | −1.04% |
| jpeg (q25) | 51.70% | −9.97% |
| noisy (σ10) | 42.11% | −19.56% |
| blurry (r1.2) | 41.88% | −19.80% |
| low_res (24px) | 41.04% | −20.63% |
| dim + noisy | 32.07% | −29.60% |
| **webcam_hard** (combined) | **31.65%** | **−30.02%** |

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
