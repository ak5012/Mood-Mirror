"""Train the ANN baseline and the CNN on FER2013, then report honest metrics.

Run:  .venv\\Scripts\\python.exe training\\train.py
      .venv\\Scripts\\python.exe training\\train.py --quick     (3 epochs, smoke run)

Models are saved as Keras 2 .h5 - the only format the tensorflowjs converter
accepts. See compat.py for why. Run smoke_test_export.py before trusting a long
run; it proves the export path works in a few seconds.

Augmentation note: the original version augmented the training set once, up
front, and then trained on that frozen copy for every epoch. That gives the
model a single fixed variant of each image, which is barely better than no
augmentation. Here augmentation is re-rolled per batch per epoch via
ImageDataGenerator, so the model sees a different rotation/zoom/flip of each
face on every pass.

Normalization note: `samplewise_center` + `samplewise_std_normalization` is
exactly `(x - mean) / (std + 1e-6)` computed per image, which is what
data.standardize() does and what the browser does. Parity is asserted at
startup rather than assumed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from compat import MODEL_EXT, EMOTIONS, describe, keras
from data import load_fer2013, standardize, degrade_image
from models import build_ann_baseline, build_cnn

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


def _slug(name):
    return name.lower().replace(" ", "_")


def assert_normalization_parity(X):
    """ImageDataGenerator's samplewise normalization must equal standardize()."""
    gen = keras.preprocessing.image.ImageDataGenerator(
        samplewise_center=True, samplewise_std_normalization=True
    )
    sample = X[:16, ..., np.newaxis].astype(np.float32)
    via_gen = np.stack([gen.standardize(x.copy()) for x in sample])
    via_ours = standardize(X[:16])[..., np.newaxis]
    delta = np.abs(via_gen - via_ours).max()
    if delta > 1e-4:
        raise SystemExit(
            f"Normalization parity broken: max delta {delta:.2e} between "
            f"ImageDataGenerator and data.standardize(). Training and inference "
            f"would see different data."
        )
    print(f"normalization parity: OK (max delta {delta:.2e})")


def class_weights_for(y):
    """Inverse-frequency weights. Without these, disgust (~1.5% of the data)
    is never predicted and the model still looks fine on overall accuracy."""
    counts = np.bincount(y, minlength=len(EMOTIONS))
    total = len(y)
    n_present = int((counts > 0).sum())
    return {
        i: (total / (n_present * c)) if c > 0 else 0.0
        for i, c in enumerate(counts)
    }


def train_model(model, Xtr, ytr, Xva, yva, name, epochs, batch_size, degrade=True):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # preprocessing_function runs at the START of standardize(), i.e. after the
    # geometric transforms and immediately before samplewise normalization -
    # exactly where sensor degradation belongs. Validation never sees it.
    aug = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=(degrade_image if degrade else None),
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    train_flow = aug.flow(
        Xtr[..., np.newaxis].astype(np.float32), ytr, batch_size=batch_size, shuffle=True
    )

    # Validation gets normalization only - never augmentation.
    val = (standardize(Xva)[..., np.newaxis], yva)

    ckpt = OUTPUT_DIR / f"{_slug(name)}_best{MODEL_EXT}"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_accuracy", save_best_only=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        train_flow,
        validation_data=val,
        epochs=epochs,
        class_weight=class_weights_for(ytr),
        callbacks=callbacks,
        verbose=2,
    )
    return history


def evaluate_model(model, Xte, yte, name):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X = standardize(Xte)[..., np.newaxis]
    y_pred = np.argmax(model.predict(X, batch_size=256, verbose=0), axis=1)
    acc = accuracy_score(yte, y_pred)

    print(f"\n{'=' * 60}\n{name} test accuracy: {acc:.4f}\n{'=' * 60}")
    print(classification_report(yte, y_pred, target_names=EMOTIONS, digits=3, zero_division=0))

    report = classification_report(
        yte, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(yte, y_pred)
    _plot_confusion(cm, acc, name)

    metrics = {
        "model": name,
        "test_accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class_precision": {e: float(report[e]["precision"]) for e in EMOTIONS},
        "per_class_recall": {e: float(report[e]["recall"]) for e in EMOTIONS},
        "per_class_f1": {e: float(report[e]["f1-score"]) for e in EMOTIONS},
        "confusion_matrix": cm.tolist(),
        "class_order": EMOTIONS,
    }
    (OUTPUT_DIR / f"{_slug(name)}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return acc, metrics


def _plot_confusion(cm, acc, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} - row-normalized confusion\ntest accuracy {acc:.4f}")
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(
                j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{_slug(name)}_confusion_matrix.png", dpi=120)
    plt.close(fig)


def _plot_history(history, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.history
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(h["loss"], label="train")
    a1.plot(h["val_loss"], label="val")
    a1.set(xlabel="epoch", ylabel="loss", title=f"{name} loss")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(h["accuracy"], label="train")
    a2.plot(h["val_accuracy"], label="val")
    a2.set(xlabel="epoch", ylabel="accuracy", title=f"{name} accuracy")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{_slug(name)}_training.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-cnn", type=int, default=60)
    ap.add_argument("--epochs-ann", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--quick", action="store_true", help="3 epochs each, for a smoke run")
    ap.add_argument("--no-degrade", action="store_true",
                    help="disable webcam-condition augmentation (noise/blur/gamma/jpeg)")
    args = ap.parse_args()
    if args.quick:
        args.epochs_cnn = args.epochs_ann = 3

    print(describe(), "\n")
    print("Loading FER2013 ...")
    (Xtr, ytr), (Xte, yte), (Xva, yva) = load_fer2013()
    print(f"  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    assert_normalization_parity(Xtr)

    results = {}

    print("\n" + "=" * 60 + "\nANN baseline\n" + "=" * 60)
    ann = build_ann_baseline()
    h = train_model(ann, Xtr, ytr, Xva, yva, "ANN Baseline", args.epochs_ann,
                    args.batch_size, degrade=not args.no_degrade)
    _plot_history(h, "ANN Baseline")
    ann_acc, _ = evaluate_model(ann, Xte, yte, "ANN Baseline")
    ann.save(OUTPUT_DIR / f"ann_baseline{MODEL_EXT}")
    results["ann"] = ann_acc

    print("\n" + "=" * 60 + "\nCNN\n" + "=" * 60)
    cnn = build_cnn()
    h = train_model(cnn, Xtr, ytr, Xva, yva, "CNN", args.epochs_cnn,
                    args.batch_size, degrade=not args.no_degrade)
    _plot_history(h, "CNN")
    cnn_acc, cnn_metrics = evaluate_model(cnn, Xte, yte, "CNN")
    cnn.save(OUTPUT_DIR / f"cnn{MODEL_EXT}")
    results["cnn"] = cnn_acc

    counts = np.bincount(yte, minlength=len(EMOTIONS))
    majority = counts.max() / counts.sum()

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print(f"  Majority-class baseline ({EMOTIONS[counts.argmax()]}): {majority:.4f}")
    print(f"  ANN baseline test accuracy:              {ann_acc:.4f}")
    print(f"  CNN test accuracy:                       {cnn_acc:.4f}")
    print(f"  CNN improvement over ANN:                {cnn_acc - ann_acc:+.4f}")
    print(f"  CNN macro F1:                            {cnn_metrics['macro_f1']:.4f}")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "majority_class_baseline": float(majority),
                "ann_test_accuracy": float(ann_acc),
                "cnn_test_accuracy": float(cnn_acc),
                "cnn_macro_f1": cnn_metrics["macro_f1"],
                "epochs": {"ann": args.epochs_ann, "cnn": args.epochs_cnn},
                "environment": describe(),
            },
            indent=2,
        )
    )
    print(f"\nArtifacts -> {OUTPUT_DIR}")
    print("Next: .venv\\Scripts\\python.exe training\\export_tfjs.py")


if __name__ == "__main__":
    main()
