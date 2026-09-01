"""Load and preprocess FER2013.

Two things in here are correctness-critical and easy to get wrong silently:

**Label order.** The canonical class list used everywhere in this project is
alphabetical (`EMOTIONS` in compat.py), which is the order the image-folder
release of FER2013 produces. The original CSV release uses a *different*
numbering: 0=angry 1=disgust 2=fear 3=happy 4=sad 5=surprise 6=neutral - note
that sad, surprise and neutral are shifted. Loading the CSV without remapping
trains a model that is confidently wrong on three of seven classes, and nothing
in the loss curve reveals it. `_CSV_TO_CANONICAL` does the remap.

**Preprocessing parity.** Whatever happens to a training image must happen
identically to a webcam crop in the browser, or test accuracy will not predict
real accuracy. The contract is: grayscale -> 48x48 -> per-image
standardization. `standardize()` below is mirrored line-for-line by
`standardize()` in web/index.html. Do not change one without the other.

Per-image standardization (rather than a plain /255) is what makes the model
survive a dim or washed-out webcam: it removes each frame's own brightness and
contrast before the CNN ever sees the pixels. The previous CLAHE step was
dropped precisely because it had no browser-side counterpart, so every model
trained with it was being fed different-looking data at inference time.
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from compat import EMOTIONS

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# Original FER2013 CSV numbering -> index into EMOTIONS (alphabetical).
_CSV_TO_CANONICAL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 4}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_SIZE = 48
EPS = 1e-6


# ---------------------------------------------------------------------------
# Preprocessing - must stay in lockstep with web/index.html
# ---------------------------------------------------------------------------
def standardize(imgs: np.ndarray) -> np.ndarray:
    """Per-image mean/std normalization. uint8 (N,48,48) -> float32 (N,48,48).

    Mirrored exactly in JS. Each image is normalized against its own statistics,
    so overall scene brightness and contrast cancel out.
    """
    x = imgs.astype(np.float32)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    return (x - mean) / (std + EPS)


def preprocess_batch(imgs: np.ndarray, augment: bool = False) -> np.ndarray:
    """Full inference-time preprocessing, plus optional training augmentation."""
    if augment:
        imgs = np.stack([augment_image(im) for im in imgs])
    return standardize(imgs)


# ---------------------------------------------------------------------------
# Augmentation (training only - never applied at inference)
# ---------------------------------------------------------------------------
def augment_image(img, rng=None):
    """Light augmentation: rotate +/-15 deg, zoom [0.9, 1.1], horizontal flip."""
    rng = rng if rng is not None else np.random.default_rng()
    pil = Image.fromarray(img, mode="L")

    if rng.random() > 0.5:
        pil = pil.rotate(float(rng.uniform(-15, 15)), resample=Image.BILINEAR)

    if rng.random() > 0.5:
        scale = float(rng.uniform(0.9, 1.1))
        new = max(1, int(IMG_SIZE * scale))
        pil = pil.resize((new, new), Image.BILINEAR)
        if new >= IMG_SIZE:
            off = (new - IMG_SIZE) // 2
            pil = pil.crop((off, off, off + IMG_SIZE, off + IMG_SIZE))
        else:
            # NOTE: Image.new takes `color`; there is no `fill` argument. The
            # original code passed fill=128 and raised TypeError on this branch.
            canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), color=128)
            off = (IMG_SIZE - new) // 2
            canvas.paste(pil, (off, off))
            pil = canvas

    if rng.random() > 0.5:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)

    return np.asarray(pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Loading - handles both FER2013 releases
# ---------------------------------------------------------------------------
def _find_csv(data_dir):
    for name in ("fer2013.csv", "icml_face_data.csv", "train.csv"):
        hits = list(data_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def _load_csv(csv_path):
    """Original CSV release: columns emotion, pixels, Usage."""
    splits = {"Training": ([], []), "PublicTest": ([], []), "PrivateTest": ([], [])}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip(): c for c in (reader.fieldnames or [])}
        e_col = cols.get("emotion")
        p_col = cols.get("pixels")
        u_col = cols.get("Usage") or cols.get("usage")
        for row in reader:
            usage = (row.get(u_col) or "Training").strip() if u_col else "Training"
            if usage not in splits:
                continue
            px = np.array(row[p_col].split(), dtype=np.uint8)
            if px.size != IMG_SIZE * IMG_SIZE:
                continue
            splits[usage][0].append(px.reshape(IMG_SIZE, IMG_SIZE))
            splits[usage][1].append(_CSV_TO_CANONICAL[int(row[e_col])])

    def pack(k):
        imgs, labels = splits[k]
        return np.asarray(imgs, dtype=np.uint8), np.asarray(labels, dtype=np.int64)

    # PublicTest is the validation split; PrivateTest is the held-out test set.
    return pack("Training"), pack("PrivateTest"), pack("PublicTest")


def _load_folders(root):
    """Image-folder release (kaggle msambare/fer2013): train/<class>/*.jpg."""

    def read_split(split_dir):
        imgs, labels = [], []
        for cls in EMOTIONS:
            cdir = split_dir / cls
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.iterdir()):
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                im = Image.open(p).convert("L")
                if im.size != (IMG_SIZE, IMG_SIZE):
                    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                imgs.append(np.asarray(im, dtype=np.uint8))
                labels.append(EMOTION_TO_IDX[cls])
        return np.asarray(imgs, dtype=np.uint8), np.asarray(labels, dtype=np.int64)

    train_dir = next((d for d in (root / "train", root) if (d / "happy").is_dir()), None)
    test_dir = next(
        (d for d in (root / "test", root / "validation") if (d / "happy").is_dir()), None
    )
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(f"No FER2013 class folders found under {root}")

    X, y = read_split(train_dir)
    Xt, yt = read_split(test_dir)

    # The folder release ships no validation split; carve a stratified 10% off
    # train with a fixed seed so runs stay comparable.
    rng = np.random.default_rng(1337)
    val_idx = []
    for c in range(len(EMOTIONS)):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val_idx.append(idx[: max(1, int(0.1 * len(idx)))])
    val_idx = np.concatenate(val_idx)
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return (X[mask], y[mask]), (Xt, yt), (X[val_idx], y[val_idx])


def load_fer2013(data_dir=DATA_DIR, use_cache=True):
    """Returns (X_train,y_train), (X_test,y_test), (X_val,y_val).

    Accepts either FER2013 release; labels are normalized to `EMOTIONS` order.
    Decoding ~36k JPEGs takes about 40s, so the packed arrays are cached to a
    single .npz. Delete it (or pass use_cache=False) to force a re-read.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Download FER2013 first - see training/README.md."
        )

    cache = root / "fer2013_cache.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return (
            (z["Xtr"], z["ytr"]),
            (z["Xte"], z["yte"]),
            (z["Xva"], z["yva"]),
        )

    csv_path = _find_csv(root)
    splits = _load_csv(csv_path) if csv_path is not None else _load_folders(root)

    if use_cache:
        (Xtr, ytr), (Xte, yte), (Xva, yva) = splits
        np.savez_compressed(
            cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xva=Xva, yva=yva
        )
    return splits


if __name__ == "__main__":
    (Xtr, ytr), (Xte, yte), (Xva, yva) = load_fer2013()
    print(f"train {Xtr.shape}  test {Xte.shape}  val {Xva.shape}")
    print("classes:", EMOTIONS)
    for name, y in (("train", ytr), ("test", yte), ("val", yva)):
        counts = np.bincount(y, minlength=len(EMOTIONS))
        print(f"  {name:5}", {e: int(c) for e, c in zip(EMOTIONS, counts)})
    s = preprocess_batch(Xtr[:8], augment=True)
    print(f"preprocessed {s.shape} {s.dtype} mean={s.mean():.3f} std={s.std():.3f}")


# ---------------------------------------------------------------------------
# Sensor degradation (training only)
#
# FER2013 is clean, evenly lit, and reasonably sharp. A real webcam in a dim
# room is none of those. That distribution gap costs real-world accuracy even
# when test accuracy looks fine, so the fix is to train on images that have
# been pushed toward webcam conditions.
#
# What matters here is which degradations SURVIVE standardize(). Per-image
# mean/std normalization is linear, so it removes brightness and contrast
# entirely - training on merely darker images teaches the model nothing,
# because standardize() undoes it before the CNN sees anything. What survives
# is everything nonlinear or information-destroying:
#
#   gamma        - tone curve, not a scale factor
#   noise        - added after the signal, scales up as the signal is scaled
#   blur         - removes high frequencies permanently
#   downscale    - removes detail permanently
#   jpeg         - blocking and ringing artifacts
#   black crush  - clipping; those pixels are simply gone
#
# Each is applied independently with its own probability, so a good fraction of
# every batch stays clean or near-clean. That matters: degrading everything
# would trade clean-image accuracy for robustness instead of adding robustness.
# ---------------------------------------------------------------------------
from io import BytesIO  # noqa: E402

# Probability that a given training image gets degraded at all.
DEGRADE_P = 0.65


def degrade_image(img, rng=None, strength=1.0):
    """Push a clean 48x48 image toward webcam conditions.

    Accepts uint8 or float (0-255) and returns float32 in 0-255, so it can be
    used directly as a Keras `preprocessing_function` - which runs after the
    geometric augmentation and immediately before samplewise normalization.
    """
    rng = rng if rng is not None else np.random.default_rng()
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3:            # (H, W, 1) from the Keras generator
        x = x[..., 0]
        had_channel = True
    else:
        had_channel = False

    if rng.random() < DEGRADE_P * strength:
        # --- low light: gamma darkening plus a gain drop ---------------------
        # Gamma is the part standardize() cannot undo; the gain mostly cancels,
        # but it is what makes the noise and quantization below realistic.
        if rng.random() < 0.7:
            gamma = float(rng.uniform(1.0, 2.4))
            gain = float(rng.uniform(0.35, 1.0))
            x = 255.0 * gain * np.power(np.clip(x, 0, 255) / 255.0, gamma)

            # Crushed blacks: dim sensors clip the bottom of the range, and
            # that information is genuinely unrecoverable.
            if rng.random() < 0.4:
                x = np.clip(x - float(rng.uniform(0, 12)), 0, 255)

        # --- blur: defocus or motion ----------------------------------------
        if rng.random() < 0.45:
            radius = float(rng.uniform(0.3, 1.5))
            pil = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), mode="L")
            x = np.asarray(
                pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
            )

        # --- resolution loss: a soft or upscaled webcam ----------------------
        if rng.random() < 0.35:
            small = int(rng.integers(20, 40))
            pil = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), mode="L")
            pil = pil.resize((small, small), Image.BILINEAR)
            x = np.asarray(
                pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32
            )

        # --- sensor noise ----------------------------------------------------
        # Noise is added AFTER darkening, which is the right order: a dim sensor
        # is noisy relative to its signal, and that ratio is what standardize()
        # preserves and the model must learn to see through.
        if rng.random() < 0.6:
            sigma = float(rng.uniform(2.0, 14.0))
            x = x + rng.normal(0.0, sigma, x.shape).astype(np.float32)

        # --- compression artifacts -------------------------------------------
        if rng.random() < 0.25:
            buf = BytesIO()
            Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), mode="L").save(
                buf, format="JPEG", quality=int(rng.integers(18, 60))
            )
            buf.seek(0)
            x = np.asarray(Image.open(buf).convert("L"), dtype=np.float32)

    x = np.clip(x, 0, 255)
    return x[..., np.newaxis] if had_channel else x


# ---------------------------------------------------------------------------
# Random erasing (Zhong et al., 2020)
#
# Blanks a random rectangle of the face. It forces the model to spread evidence
# across the whole face instead of leaning on one region - which matters here
# because several FER2013 classes hinge on a single feature (surprise on the
# eyes, happy on the mouth). A model that has only ever seen complete faces
# degrades badly when a hand, hair, or the frame edge covers part of one.
#
# Published FER2013 results in the 70%+ range use this alongside heavy
# geometric augmentation, which is what makes it worth the cost here.
# ---------------------------------------------------------------------------
ERASE_P = 0.35
ERASE_AREA = (0.02, 0.18)     # fraction of the 48x48 image
ERASE_ASPECT = (0.4, 2.5)


def random_erase(img, rng=None):
    """Zero out one random rectangle. Input/output float or uint8 (H,W[,1])."""
    rng = rng if rng is not None else np.random.default_rng()
    x = np.asarray(img, dtype=np.float32)
    squeeze = x.ndim == 3
    if squeeze:
        x = x[..., 0]

    if rng.random() < ERASE_P:
        h, w = x.shape
        for _ in range(10):     # retry until a box fits
            area = h * w * float(rng.uniform(*ERASE_AREA))
            aspect = float(rng.uniform(*ERASE_ASPECT))
            eh, ew = int(round(np.sqrt(area * aspect))), int(round(np.sqrt(area / aspect)))
            if eh < h and ew < w and eh > 0 and ew > 0:
                top = int(rng.integers(0, h - eh))
                left = int(rng.integers(0, w - ew))
                # Fill with the image's own mean rather than 0: a black box is a
                # strong edge the conv filters would latch onto as a feature.
                x[top:top + eh, left:left + ew] = float(x.mean())
                break

    return x[..., np.newaxis] if squeeze else x


def train_preprocess(img):
    """Full training-time pixel pipeline: sensor degradation, then erasing.

    Used as the Keras `preprocessing_function`, so it runs after the geometric
    transforms and immediately before samplewise normalization.
    """
    return random_erase(degrade_image(img))


# ---------------------------------------------------------------------------
# Ten-crop test-time augmentation
#
# The standard TTA used by the published 70%+ FER2013 results: upscale slightly,
# take 48x48 crops at the four corners and centre, mirror each, and average the
# ten predictions. It buys roughly a point for ten forward passes and no
# retraining, because averaging over crops cancels the model's sensitivity to
# exactly where the face sits in the frame.
#
# Ten passes is too expensive for the browser at 30fps - the page uses the
# two-crop version (image + mirror). This is for offline evaluation, where it
# gives the fairest picture of what the weights are actually capable of.
# ---------------------------------------------------------------------------
TEN_CROP_UPSCALE = 54


def ten_crop(imgs, upscale=TEN_CROP_UPSCALE):
    """uint8 (N,48,48) -> float32 (10,N,48,48): 5 crops x {identity, mirror}."""
    n = len(imgs)
    big = np.empty((n, upscale, upscale), dtype=np.uint8)
    for i in range(n):
        pil = Image.fromarray(imgs[i].astype(np.uint8), "L")
        big[i] = np.asarray(pil.resize((upscale, upscale), Image.BILINEAR))

    d = upscale - IMG_SIZE
    c = d // 2
    offsets = [(0, 0), (0, d), (d, 0), (d, d), (c, c)]   # corners + centre

    out = []
    for top, left in offsets:
        crop = big[:, top:top + IMG_SIZE, left:left + IMG_SIZE]
        out.append(crop)
        out.append(crop[:, :, ::-1])                      # mirror
    return np.stack(out).astype(np.float32)


def predict_ten_crop(model, imgs, batch_size=256):
    """Average softmax over the ten crops. Returns (N, n_classes)."""
    crops = ten_crop(imgs)
    total = None
    for view in crops:
        p = model.predict(standardize(view)[..., np.newaxis],
                          batch_size=batch_size, verbose=0)
        total = p if total is None else total + p
    return total / len(crops)
