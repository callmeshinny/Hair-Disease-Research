"""5-fold image-level CV for the three hair-disorder classifiers.

The train and validation folders are pooled for cross-validation. The test
folder is inspected for duplicate images but remains held out from CV.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    pixel_hash = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()
                    ).hexdigest()
                rows.append({
                    "path": str(path.resolve()), "label": category.name,
                    "split": split, "pixel_hash": pixel_hash,
                })
    if not rows:
        raise ValueError("No images found.")
    return pd.DataFrame(rows)


def make_folds(labels, splitter_name: str, n_splits: int, seed: int):
    splitter_type = StratifiedKFold if splitter_name == "stratified" else KFold
    splitter = splitter_type(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


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
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def make_generator(tf, frame, classes, model_name, args, training, seed):
    options = {"rescale": 1.0 / 255}
    if args.preprocessing == "imagenet":
        modules = {"MobileNetV2": "mobilenet_v2", "InceptionV3": "inception_v3", "DenseNet169": "densenet"}
        options = {"preprocessing_function": getattr(tf.keras.applications, modules[model_name]).preprocess_input}
    if training:
        options.update(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                       shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
                       vertical_flip=True, fill_mode="nearest")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**options)
    result = datagen.flow_from_dataframe(
        frame, x_col="path", y_col="label", classes=classes, target_size=(224, 224),
        color_mode="rgb", class_mode="categorical", batch_size=args.batch_size,
        shuffle=training, seed=seed, validate_filenames=True,
    )
    if result.samples != len(frame):
        raise ValueError("Generator skipped images; fix the dataset before training.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cv_results"))
    parser.add_argument("--splitter", choices=["stratified", "kfold"], default="stratified")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing", choices=["repo", "imagenet"], default="repo")
    parser.add_argument("--models", nargs="+", default=["MobileNetV2", "InceptionV3", "DenseNet169"],
                        choices=["MobileNetV2", "InceptionV3", "DenseNet169"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n_splits < 2 or args.epochs < 1 or args.batch_size < 1:
        parser.error("n-splits >= 2, epochs >= 1, batch-size >= 1 required.")

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    all_images = scan_images(args.data_root)
    all_images.to_csv(args.output / "image_manifest.csv", index=False)
    duplicates = all_images[all_images.duplicated("pixel_hash", keep=False)]
    if len(duplicates):
        duplicates.to_csv(args.output / "exact_duplicates.csv", index=False)
        raise ValueError("Identical decoded images found. Review exact_duplicates.csv before CV.")

    frame = all_images[all_images["split"].isin(["train", "val"])].reset_index(drop=True)
    classes = sorted(frame.label.unique())
    if len(classes) < 2 or frame.label.value_counts().min() < args.n_splits:
        raise ValueError("Need at least n-splits images for every class.")
    labels = frame.label.map({label: i for i, label in enumerate(classes)}).to_numpy()
    splits = make_folds(labels, args.splitter, args.n_splits, args.seed)
    fold_ids = np.full(len(frame), -1, dtype=int)
    for fold, (train_index, eval_index) in enumerate(splits, 1):
        fold_ids[eval_index] = fold
        if set(labels[train_index]) != set(range(len(classes))):
            raise ValueError(f"Fold {fold} training split lacks a class.")
    frame.assign(evaluation_fold=fold_ids).to_csv(args.output / "fold_assignments.csv", index=False)
    (args.output / "classes.json").write_text(json.dumps(classes, indent=2))
    print(pd.crosstab(frame.label, fold_ids))
    print(f"CV images: {len(frame)}; held-out test images: {(all_images['split'] == 'test').sum()}")
    if args.dry_run:
        return

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    results, summaries = [], []
    for name in args.models:
        oof = np.full((len(frame), len(classes)), np.nan, dtype=np.float32)
        for fold, (train_index, eval_index) in enumerate(splits, 1):
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.utils.set_random_seed(args.seed + fold)
            folder = args.output / name / f"fold_{fold}"
            folder.mkdir(parents=True)
            train = make_generator(tf, frame.iloc[train_index], classes, name, args, True, args.seed + fold)
            evaluation = make_generator(tf, frame.iloc[eval_index], classes, name, args, False, args.seed + fold)
            model = build_model(tf, name, len(classes))
            callbacks = [tf.keras.callbacks.CSVLogger(str(folder / "history.csv"))]
            if name != "MobileNetV2":
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: 0.001 if epoch <= 10 else 0.0001
                ))
            model.fit(train, epochs=args.epochs, callbacks=callbacks, verbose=2)
            probabilities = model.predict(evaluation, verbose=1)
            oof[eval_index] = probabilities
            truth, prediction = labels[eval_index], probabilities.argmax(axis=1)
            row = {"model": name, "fold": fold, "train_n": len(train_index), "eval_n": len(eval_index),
                   "accuracy": accuracy_score(truth, prediction),
                   "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0)}
            results.append(row)
            pd.DataFrame(results).to_csv(args.output / "fold_metrics.csv", index=False)
            (folder / "classification_report.json").write_text(json.dumps(
                classification_report(truth, prediction, target_names=classes, output_dict=True, zero_division=0), indent=2
            ))
            pd.DataFrame(confusion_matrix(truth, prediction), index=classes, columns=classes).to_csv(
                folder / "confusion_matrix.csv"
            )
            print(row)
            del model, train, evaluation
        metrics = pd.DataFrame([row for row in results if row["model"] == name])
        summaries.append({"model": name, "accuracy_mean": metrics.accuracy.mean(),
                          "accuracy_sd": metrics.accuracy.std(ddof=1),
                          "macro_f1_mean": metrics.macro_f1.mean(),
                          "macro_f1_sd": metrics.macro_f1.std(ddof=1),
                          "oof_accuracy": accuracy_score(labels, oof.argmax(axis=1))})
        pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__ == "__main__":
    main()
