#!/usr/bin/env python3
"""
Paired preprocessing ablation for hair-disorder classifiers.

Compares:
  A) repo:      rescale pixels to [0, 1]
  B) imagenet:  model-specific Keras ImageNet preprocess_input

All models use the same fixed stratified folds, seed, epochs, batch size,
architecture, optimizer, and augmentation settings across preprocessing arms.
The original test folder is scanned for exact duplicate decoded images but is
never used for training, model selection, or CV evaluation.

Example:
  DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib .venv/bin/python \
    preprocessing_ablation.py \
    --data-root "/Users/nguyenbaochau/Downloads/Hair Diseases - Final" \
    --output preprocessing_ablation_smoke \
    --epochs 1 --n-splits 5 --batch-size 32
"""
import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODEL_NAMES = ["MobileNetV2", "InceptionV3", "DenseNet169"]
PREPROCESSING_ARMS = ["repo", "imagenet"]


def scan_images(root: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        directory = root / split
        if not directory.is_dir():
            if split != "test":
                raise ValueError(f"Missing directory: {directory}")
            continue
        for category in sorted(directory.iterdir()):
            if not category.is_dir():
                continue
            for path in sorted(category.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    pixel_hash = hashlib.sha256(
                        str(im.size).encode() + im.tobytes()
                    ).hexdigest()
                rows.append({
                    "path": str(path.resolve()),
                    "label": category.name,
                    "split": split,
                    "pixel_hash": pixel_hash,
                })
    if not rows:
        raise ValueError("No images found.")
    return pd.DataFrame(rows)


def set_seed(tf, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_model(tf, name: str, class_count: int):
    layers = tf.keras.layers
    backbone = getattr(tf.keras.applications, name)(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    backbone.trainable = False
    x = backbone.output
    if name == "DenseNet169":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.Flatten()(x)
    if name != "MobileNetV2":
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
    output = layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(backbone.input, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_generator(tf, frame, classes, model_name, preprocessing, batch_size,
                   training, seed):
    if preprocessing == "repo":
        options = {"rescale": 1.0 / 255}
    else:
        modules = {
            "MobileNetV2": "mobilenet_v2",
            "InceptionV3": "inception_v3",
            "DenseNet169": "densenet",
        }
        module = getattr(tf.keras.applications, modules[model_name])
        options = {"preprocessing_function": module.preprocess_input}

    if training:
        options.update(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
        )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**options)
    generator = datagen.flow_from_dataframe(
        frame,
        x_col="path",
        y_col="label",
        classes=classes,
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=training,
        seed=seed,
        validate_filenames=True,
    )
    if generator.samples != len(frame):
        raise ValueError(
            f"Generator skipped images: expected {len(frame)}, got {generator.samples}"
        )
    return generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--arms", nargs="+", choices=PREPROCESSING_ARMS,
                        default=PREPROCESSING_ARMS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.n_splits < 2 or args.epochs < 1 or args.batch_size < 1:
        parser.error("n-splits >= 2, epochs >= 1, batch-size >= 1 required.")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "data_root": str(args.data_root.resolve()),
        "output": str(args.output.resolve()),
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "models": args.models,
        "arms": args.arms,
        "folds_shared_across_arms": True,
        "test_used_for_training_or_selection": False,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2))

    all_images = scan_images(args.data_root)
    all_images.to_csv(args.output / "image_manifest.csv", index=False)
    duplicates = all_images[all_images.duplicated("pixel_hash", keep=False)]
    if len(duplicates):
        duplicates.to_csv(args.output / "exact_duplicates.csv", index=False)
        raise ValueError(
            f"Found {len(duplicates)} rows with exact duplicate decoded pixels. "
            "Review exact_duplicates.csv before running CV."
        )

    frame = all_images[all_images["split"].isin(["train", "val"])].reset_index(drop=True)
    classes = sorted(frame["label"].unique())
    if len(classes) < 2 or frame["label"].value_counts().min() < args.n_splits:
        raise ValueError("Need at least n-splits images for every class.")
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    labels = frame["label"].map(label_to_id).to_numpy()

    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(np.zeros(len(labels)), labels))
    fold_ids = np.full(len(frame), -1, dtype=int)
    for fold, (_, eval_idx) in enumerate(splits, 1):
        fold_ids[eval_idx] = fold
    frame.assign(evaluation_fold=fold_ids).to_csv(
        args.output / "fold_assignments.csv", index=False
    )
    (args.output / "classes.json").write_text(json.dumps(classes, indent=2))
    print(pd.crosstab(frame["label"], fold_ids))
    print(
        f"CV images: {len(frame)}; held-out test images: "
        f"{int((all_images['split'] == 'test').sum())}"
    )
    print(f"Shared fold assignments saved to: {args.output / 'fold_assignments.csv'}")
    if args.dry_run:
        print("Dry run complete; no TensorFlow training was started.")
        return

    import tensorflow as tf
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    fold_rows = []
    summaries = []
    total_start = time.time()

    # Use the same fold indices and reset the same seed for paired arms.
    # The only intended experimental difference is preprocessing.
    for model_name in args.models:
        for arm in args.arms:
            print(f"\n===== MODEL={model_name} | PREPROCESSING={arm} =====")
            oof = np.full((len(frame), len(classes)), np.nan, dtype=np.float32)

            for fold, (train_idx, eval_idx) in enumerate(splits, 1):
                fold_start = time.time()
                tf.keras.backend.clear_session()
                gc.collect()
                fold_seed = args.seed + fold
                set_seed(tf, fold_seed)

                fold_dir = args.output / model_name / arm / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=False)

                train_gen = make_generator(
                    tf, frame.iloc[train_idx], classes, model_name, arm,
                    args.batch_size, True, fold_seed
                )
                eval_gen = make_generator(
                    tf, frame.iloc[eval_idx], classes, model_name, arm,
                    args.batch_size, False, fold_seed
                )
                model = build_model(tf, model_name, len(classes))
                callbacks = [tf.keras.callbacks.CSVLogger(str(fold_dir / "history.csv"))]
                if model_name != "MobileNetV2":
                    callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                        lambda epoch: 0.001 if epoch <= 10 else 0.0001
                    ))

                model.fit(
                    train_gen,
                    epochs=args.epochs,
                    callbacks=callbacks,
                    verbose=2,
                )
                probabilities = model.predict(eval_gen, verbose=1)
                if len(probabilities) != len(eval_idx):
                    raise ValueError("Prediction count does not match evaluation fold.")
                oof[eval_idx] = probabilities
                truth = labels[eval_idx]
                prediction = probabilities.argmax(axis=1)

                row = {
                    "model": model_name,
                    "preprocessing": arm,
                    "fold": fold,
                    "seed": fold_seed,
                    "train_n": len(train_idx),
                    "eval_n": len(eval_idx),
                    "accuracy": float(accuracy_score(truth, prediction)),
                    "macro_f1": float(f1_score(
                        truth, prediction, average="macro", zero_division=0
                    )),
                    "elapsed_seconds": round(time.time() - fold_start, 2),
                }
                fold_rows.append(row)
                pd.DataFrame(fold_rows).to_csv(
                    args.output / "fold_metrics.csv", index=False
                )
                (fold_dir / "classification_report.json").write_text(json.dumps(
                    classification_report(
                        truth, prediction, labels=list(range(len(classes))),
                        target_names=classes, output_dict=True, zero_division=0
                    ),
                    indent=2,
                ))
                pd.DataFrame(
                    confusion_matrix(truth, prediction, labels=list(range(len(classes)))),
                    index=classes, columns=classes
                ).to_csv(fold_dir / "confusion_matrix.csv")
                print(row)

                del model, train_gen, eval_gen, probabilities
                tf.keras.backend.clear_session()
                gc.collect()

            if np.isnan(oof).any():
                raise ValueError(f"Some OOF predictions are missing for {model_name}/{arm}.")
            metrics = pd.DataFrame([
                r for r in fold_rows
                if r["model"] == model_name and r["preprocessing"] == arm
            ])
            summaries.append({
                "model": model_name,
                "preprocessing": arm,
                "accuracy_mean": float(metrics["accuracy"].mean()),
                "accuracy_sd": float(metrics["accuracy"].std(ddof=1)),
                "macro_f1_mean": float(metrics["macro_f1"].mean()),
                "macro_f1_sd": float(metrics["macro_f1"].std(ddof=1)),
                "oof_accuracy": float(accuracy_score(labels, oof.argmax(axis=1))),
                "oof_macro_f1": float(f1_score(
                    labels, oof.argmax(axis=1), average="macro", zero_division=0
                )),
            })
            pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
            print("\nCurrent summary:")
            print(pd.DataFrame(summaries).to_string(index=False))

    summary = pd.DataFrame(summaries)
    pivot = summary.pivot(index="model", columns="preprocessing",
                          values=["accuracy_mean", "macro_f1_mean", "oof_accuracy"])
    pivot.to_csv(args.output / "paired_preprocessing_comparison.csv")
    (args.output / "run_info.json").write_text(json.dumps({
        "finished": True,
        "total_elapsed_seconds": round(time.time() - total_start, 2),
        "note": "Held-out test images were not used in CV training/evaluation.",
    }, indent=2))
    print("\nFinal paired comparison:")
    print(pivot.to_string())
    print(f"\nResults saved in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
