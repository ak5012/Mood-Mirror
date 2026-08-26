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
from PIL import Image

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
