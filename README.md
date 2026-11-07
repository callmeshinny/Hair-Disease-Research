# Hair Disease Research

Reproducible TensorFlow/Keras experiments for classifying 10 hair-disease
categories. All reported runs use seed `42`, the same v3 split, and separate
train/validation/test data. The repository contains the notebooks and
lightweight evaluation artifacts; image datasets and model binaries are kept
outside GitHub.

## Results at a glance

The table reports the final single-model metric from each completed run.
Accuracy and Macro-F1 are calculated on the held-out test split. The completed
cross-validation runs are listed separately below.

| Model | Dataset | Train | Validation | Test | Evaluation | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---|---:|---:|
| ResNet50V2 | fixed 5% subset | 410 | 88 | 88 | single model | 69.32% | 68.41% |
| InceptionV3 | full v3 dataset | 8,185 | 1,755 | 1,753 | single model | 99.20% | 99.19% |
| DenseNet169 | full v3 dataset | 8,185 | 1,755 | 1,753 | single model | 98.46% | 98.45% |
| MobileNetV2 | full v3 dataset | 8,185 | 1,755 | 1,753 | single model | 98.86% | 98.85% |
| VGG16 | full v3 dataset | 8,185 | 1,755 | 1,753 | single model | **88.82%** | **88.53%** |

The full dataset contains `11,693` images in total. The fixed 5% subset
contains `586` images in total and is generated deterministically from the
same v3 source split. Results trained on 5% and 100% must not be interpreted
as a perfectly controlled model comparison; dataset size is a major
experimental difference.

## Completed runs

Only runs with final metrics, predictions, and evaluation outputs are included.
Interrupted runs, smoke tests, incomplete configurations, and large model
checkpoints are excluded.

- [ResNet50V2 5% single model](results/resnet50v2_seed42_sub5pct_20261027_090212/)
- [ResNet50V2 5% five-fold CV](results/resnet50v2_cv5_seed42_sub5pct_20261027_090425/)
- [InceptionV3 full-data single model](results/inceptionv3_seed42_20261023_161640/)
- [DenseNet169 full-data single model](results/densenet169_seed42_20261023_165946/)
- [MobileNetV2 full-data single model](results/mobilenetv2_seed42_20261021_042340/)
- [VGG16 full-data single model](results/vgg16_seed42_20261107_211549/)

### VGG16 full-data details

The completed VGG16 run used all `11,693` images from the v3 split:

- Train: `8,185`; validation: `1,755`; test: `1,753`.
- Image size: `224 x 224`; batch size: `8`; seed: `42`.
- Backbone: ImageNet-pretrained VGG16 with VGG16 `preprocess_input`.
- Phase 1 trained the classification head; phase 2 fine-tuned `block5_conv1`.
- Selected checkpoint: phase 2, using validation loss only.
- Test accuracy: **`88.82%`**.
- Test macro-F1: **`88.53%`**; balanced accuracy: `88.67%`.
- Bootstrap 95% CI for accuracy: `87.28%`–`90.25%`.

The result is a single fixed-split evaluation on the independent test set.

Each result directory includes the applicable configuration, test metrics,
classification report, predictions, confusion matrices, training history,
and error-analysis files. Cross-validation directories additionally include
fold metrics, out-of-fold predictions, and ensemble evaluation files.

## Notebooks

- [ResNet50V2 local notebook](notebooks/ResNet50V2_HD_local.ipynb)
- [VGG16 local notebook](notebooks/VGG16_HD_local.ipynb)
- [InceptionV3 notebook](notebooks/InceptionV3_HD.ipynb)
- [DenseNet169 notebook](notebooks/DenseNet169_HD.ipynb)
- [MobileNetV2 notebook](notebooks/MobileNetV2_HD.ipynb)

The VGG16 local notebook is configured for the full v3 dataset
(`DATA_MODE = "full"`), trains with two transfer-learning phases, saves
weights-only checkpoints, and skips cross-validation during **Run All**.
Cross-validation is kept in the dedicated CV notebook.

## Reproducibility and repository scope

The 5% subset is selected with `SEED = 42` and
`SUBSET_FRACTION = 0.05`. Full-data runs use
`data_processed/dataset_224_v3`. Set the dataset path in the notebook before
running. Dataset archives, image files, `.keras`/`.h5` checkpoints, caches,
and editor files are intentionally ignored because of size and reproducibility
storage constraints.
