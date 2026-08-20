# Mood Mirror

Real-time facial emotion detection in the browser. Webcam → face detection →
CNN classification → emoji overlay, with the model trained from scratch on FER2013.

> **Build status:** Step 0 complete (pipeline shell, pretrained model).
> The self-trained CNN is not built yet. Every metrics table below is
> intentionally empty — no number goes in here until an actual run produces it.

## Running the shell

The browser blocks webcam access on `file://`, so this must be served over
`localhost`:

```bash
cd web
python -m http.server 8000    # or: npx serve .
```

Then open <http://localhost:8000/> .

## Current state

| Component | Status |
|---|---|
| Webcam capture + face detection | done (Human library, detector only) |
| Emotion classification | **pretrained placeholder** — to be replaced |
| Emoji overlay + bounding box | done |
| Smoothing (8-frame majority vote) | done |
| Latency / stability instrumentation | done, collecting real numbers |
| Self-trained CNN | not started |
| Backend + history | not started |
| Deployment | not started |

## Results

Empty until Phase 1 runs.

| Model | Test accuracy |
|---|---|
| Majority-class baseline | — |
| ANN baseline | — |
| CNN (self-trained) | — |

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
