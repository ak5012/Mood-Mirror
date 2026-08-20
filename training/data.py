"""Load and preprocess FER2013 for training."""
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import io


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def load_fer2013(data_dir: str = "data"):
    """Load FER2013 CSV, parse pixels, return X_train, y_train, X_test, y_test.

    The CSV has columns: emotion, pixels, Usage.
    pixels: space-separated 48x48 grayscale values (0-255).
    Usage: 'Training', 'PrivateTest', or 'PublicTest'.

    Returns: (X_train, y_train), (X_test, y_test), (X_val, y_val) as uint8 numpy arrays.
    """
    csv_path = Path(data_dir) / "fer2013.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run: kaggle datasets download -d msambare/fer2013 -p {data_dir}")

    train_imgs, train_labels = [], []
    val_imgs, val_labels = [], []
    test_imgs, test_labels = [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = int(row['emotion'])
            usage = row['Usage']
            pixels = np.array([int(x) for x in row['pixels'].split()], dtype=np.uint8).reshape(48, 48)

            if usage == 'Training':
                train_imgs.append(pixels)
                train_labels.append(emotion)
            elif usage == 'PrivateTest':
                val_imgs.append(pixels)
                val_labels.append(emotion)
            elif usage == 'PublicTest':
                test_imgs.append(pixels)
                test_labels.append(emotion)

    return (
        (np.array(train_imgs), np.array(train_labels)),
        (np.array(test_imgs), np.array(test_labels)),
        (np.array(val_imgs), np.array(val_labels)),
    )


def augment_image(img: np.ndarray) -> np.ndarray:
    """Light augmentation: rotate ±15°, slight zoom, horizontal flip.

    Augmentation happens in PIL (smooth resample) then back to numpy.
    """
    pil_img = Image.fromarray(img, mode='L')

    # Random rotation ±15°
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR)

    # Random zoom [0.9, 1.1]
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.9, 1.1)
        new_size = int(48 * scale)
        pil_img = pil_img.resize((new_size, new_size), Image.BILINEAR)
        # Pad or crop back to 48x48
        if new_size > 48:
            left = (new_size - 48) // 2
            pil_img = pil_img.crop((left, left, left + 48, left + 48))
        else:
            padded = Image.new('L', (48, 48), fill=128)
            offset = (48 - new_size) // 2
            padded.paste(pil_img, (offset, offset))
            pil_img = padded

    # Random horizontal flip
    if np.random.rand() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    return np.array(pil_img, dtype=np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Applied to both training and inference to handle uneven lighting.
    This is the standard preprocessing for FER2013.
    """
    pil_img = Image.fromarray(img, mode='L')
    # PIL has no built-in CLAHE; use skimage if available, else simple equalization
    try:
        from skimage import exposure
        img_arr = np.array(pil_img)
        # skimage CLAHE is not in core; use basic equalization as fallback
        return (exposure.equalize_adapthist(img_arr) * 255).astype(np.uint8)
    except ImportError:
        # Fallback: PIL's autocontrast as a simple enhancement
        enhancer = ImageEnhance.Contrast(pil_img)
        return np.array(enhancer.enhance(1.5), dtype=np.uint8)


def preprocess_batch(imgs: np.ndarray, augment: bool = False) -> np.ndarray:
    """Apply CLAHE + optional augmentation, normalize to [0, 1]."""
    processed = []
    for img in imgs:
        if augment:
            img = augment_image(img)
        img = apply_clahe(img)
        processed.append(img.astype(np.float32) / 255.0)
    return np.array(processed)


if __name__ == "__main__":
    # Quick sanity check
    print("Testing data pipeline...")
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_fer2013()
    print(f"Train: {X_train.shape} {y_train.shape}")
    print(f"Test:  {X_test.shape} {y_test.shape}")
    print(f"Val:   {X_val.shape} {y_val.shape}")
    print(f"Classes: {EMOTIONS}")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # Test augmentation
    sample = preprocess_batch(X_train[:5], augment=True)
    print(f"Preprocessed sample: {sample.shape} dtype={sample.dtype} range=[{sample.min():.2f}, {sample.max():.2f}]")
