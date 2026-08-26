"""Build a reference sheet of real FER2013 faces for framing calibration.

Run:  .venv\\Scripts\\python.exe training\\make_reference_sheet.py

Why this exists: the model was trained on FER2013's particular crop convention
(the original ICML-2013 challenge used a Viola-Jones frontal-face box scaled to
48x48). The browser crops from a BlazeFace detector box, which frames faces
differently - looser, and tilted whenever the head is tilted. That mismatch
costs real-world accuracy no matter how good the model is, and it is invisible
from test metrics because the test set has FER2013 framing by construction.

There is no published spec for FER2013's exact geometry, so rather than guess,
this writes out a grid of genuine training faces. The live parity crop is shown
against it in the browser, and the crop parameters are tunable, so framing can
be matched by eye instead of by folklore.

Also emits the mean face per class, which makes the target framing much easier
to see than any single noisy example.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compat import EMOTIONS  # noqa: E402
from data import load_fer2013  # noqa: E402

WEB_MODEL_DIR = Path(__file__).resolve().parent.parent / "web" / "model"
COLS = 10
CELL = 48


def main():
    (Xtr, ytr), _, _ = load_fer2013()
    rng = np.random.default_rng(7)

    rows = len(EMOTIONS)
    # +1 column on the left for the per-class mean face.
    sheet = Image.new("L", (CELL * (COLS + 1), CELL * rows), color=0)

    for r, emo in enumerate(EMOTIONS):
        idx = np.flatnonzero(ytr == r)

        mean_face = Xtr[idx].mean(axis=0).astype(np.uint8)
        sheet.paste(Image.fromarray(mean_face, "L"), (0, r * CELL))

        pick = rng.choice(idx, size=min(COLS, len(idx)), replace=False)
        for c, i in enumerate(pick):
            sheet.paste(Image.fromarray(Xtr[i], "L"), ((c + 1) * CELL, r * CELL))

    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = WEB_MODEL_DIR / "fer2013_reference.png"
    sheet.save(out)

    print(f"Wrote {out}  ({sheet.size[0]}x{sheet.size[1]}, {out.stat().st_size / 1024:.0f} KB)")
    print(f"Rows top-to-bottom: {', '.join(EMOTIONS)}")
    print("Column 0 is the per-class mean face - that is the framing to match.")

    # The mean face over the whole training set is the single best picture of
    # FER2013's framing convention: eye line, face width, and headroom all show
    # up as a sharp average where the dataset is consistent.
    overall = Xtr.mean(axis=0).astype(np.uint8)
    big = Image.fromarray(overall, "L").resize((192, 192), Image.NEAREST)
    big.save(WEB_MODEL_DIR / "fer2013_mean_face.png")

    ys, xs = np.mgrid[0:CELL, 0:CELL]
    m = overall.astype(float)
    print(
        f"\nMean-face centroid: x={(xs * m).sum() / m.sum():.1f} "
        f"y={(ys * m).sum() / m.sum():.1f} (of {CELL})"
    )


if __name__ == "__main__":
    main()
