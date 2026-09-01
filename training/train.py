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
from data import load_fer2013, standardize, train_preprocess, predict_ten_crop
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


def class_weights_for(y, mode="sqrt"):
    """Per-class loss weights, normalized to mean 1.

    `inverse` is the textbook 1/frequency weighting. It is too aggressive here:
    happy has 6,494 training images and disgust has 393, a 16.5x ratio, so
    inverse weighting tells the model a disgust example matters 16.5x more than
    a happy one. The measured result was disgust predicted far too often
    (precision 0.269) while angry and fear were barely predicted at all
    (recall 0.05 and 0.045). Overall accuracy looked acceptable while three of
    seven classes were effectively broken.

    `sqrt` uses 1/sqrt(frequency), which still lifts the rare classes but with
    a 4.1x spread instead of 16.5x. That is the standard remedy when inverse
    weighting over-corrects, and it is the default here.

    `effective` implements the class-balanced weighting of Cui et al. (2019),
    which weights by the "effective number" of samples rather than the raw
    count - a principled interpolation between no weighting and inverse.
    """
    counts = np.bincount(y, minlength=len(EMOTIONS)).astype(np.float64)
    present = counts > 0

    if mode == "none":
        w = np.ones_like(counts)
    elif mode == "inverse":
        w = np.where(present, 1.0 / np.maximum(counts, 1), 0.0)
    elif mode == "sqrt":
        w = np.where(present, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    elif mode == "effective":
        beta = 0.999
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = np.where(present, 1.0 / np.maximum(eff, 1e-8), 0.0)
    else:
        raise ValueError(f"unknown class-weight mode: {mode}")

    # Normalize to mean 1 over present classes so the effective learning rate
    # does not change when the weighting scheme changes.
    w = w / w[present].mean()
    return {i: float(w[i]) for i in range(len(EMOTIONS))}


class MacroF1(keras.callbacks.Callback):
    """Log val_macro_f1 each epoch so checkpointing can select on it.

    Selecting the best epoch by val_accuracy quietly favours the majority
    classes: a model that predicts happy and neutral well and ignores disgust
    and fear can win on accuracy while being useless on four of seven classes.
    Macro F1 averages per-class F1 with equal weight, so improving a rare class
    counts as much as improving a common one. That is the metric that matches
    "accuracy across all emotions".
    """

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import f1_score

        logs = logs if logs is not None else {}
        pred = self.model.predict(self.X_val, batch_size=256, verbose=0).argmax(1)
        logs["val_macro_f1"] = float(
            f1_score(self.y_val, pred, average="macro", zero_division=0)
        )


def train_model(model, Xtr, ytr, Xva, yva, name, epochs, batch_size, degrade=True,
                weight_mode="sqrt", label_smoothing=0.05):
    n_classes = len(EMOTIONS)
    steps = int(np.ceil(len(Xtr) / batch_size))

    # Cosine decay instead of ReduceLROnPlateau. The plateau callback only cuts
    # the rate after damage is already visible in val_loss, and each cut is a
    # discontinuity the optimizer has to recover from. A cosine schedule anneals
    # smoothly to near zero over the whole run, which reliably buys a point or
    # two on a fixed epoch budget.
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=epochs * steps, alpha=1e-2
    )

    # Label smoothing needs one-hot targets, so class weighting has to move from
    # `class_weight` (integer labels only) to per-sample weights.
    cw = class_weights_for(ytr, weight_mode)
    sample_w = np.asarray([cw[int(c)] for c in ytr], dtype=np.float32)
    ytr_oh = keras.utils.to_categorical(ytr, n_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=schedule),
        # FER2013's labels are crowd-sourced and roughly 35% disputable, so a
        # hard 1.0 target trains the model to be confident about noise. Smoothing
        # caps that confidence and consistently helps on noisy-label datasets.
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    print(f"  class weights ({weight_mode}): "
          + ", ".join(f"{e}={cw[i]:.2f}" for i, e in enumerate(EMOTIONS)))

    # preprocessing_function runs at the START of standardize(), i.e. after the
    # geometric transforms and immediately before samplewise normalization -
    # exactly where sensor degradation belongs. Validation never sees it.
    aug = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=(train_preprocess if degrade else None),
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    train_flow = aug.flow(
        Xtr[..., np.newaxis].astype(np.float32), ytr_oh,
        sample_weight=sample_w, batch_size=batch_size, shuffle=True,
    )

    # Validation gets normalization only - never augmentation or degradation.
    Xva_p = standardize(Xva)[..., np.newaxis]
    val = (Xva_p, keras.utils.to_categorical(yva, n_classes))

    ckpt = OUTPUT_DIR / f"{_slug(name)}_best{MODEL_EXT}"
    callbacks = [
        # Must run first so val_macro_f1 is in `logs` before the callbacks below
        # read it.
        MacroF1(Xva_p, yva),
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_macro_f1", mode="max",
            save_best_only=True, verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=15,
            restore_best_weights=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_flow,
        validation_data=val,
        epochs=epochs,
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
    ap.add_argument("--weights", default="sqrt",
                    choices=["sqrt", "inverse", "effective", "none"],
                    help="class-weighting scheme (default sqrt)")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
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
                    args.batch_size, degrade=not args.no_degrade,
                    weight_mode=args.weights, label_smoothing=args.label_smoothing)
    _plot_history(h, "ANN Baseline")
    ann_acc, _ = evaluate_model(ann, Xte, yte, "ANN Baseline")
    ann.save(OUTPUT_DIR / f"ann_baseline{MODEL_EXT}")
    results["ann"] = ann_acc

    print("\n" + "=" * 60 + "\nCNN\n" + "=" * 60)
    cnn = build_cnn()
    h = train_model(cnn, Xtr, ytr, Xva, yva, "CNN", args.epochs_cnn,
                    args.batch_size, degrade=not args.no_degrade,
                    weight_mode=args.weights, label_smoothing=args.label_smoothing)
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
