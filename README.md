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
| VGG16 | fixed 5% subset | 410 | 88 | 88 | single model | 48.86% | 44.99% |
| VGG16 | fixed 5% subset | 498 development / 88 test | — | 88 | 5-fold test ensemble | **76.14%** | **75.95%** |
| InceptionV3 | full v3 dataset | 8,185 | 1,755 | 1,753 | single model | 99.20% | 99.19% |
| DenseNet169 | full v3 dataset | 8,185 | 1,755 | 1,753 | single model | 98.46% | 98.45% |
| MobileNetV2 | full v3 dataset | 8,185 | 1,755 | 1,753 | single model | 98.86% | 98.85% |

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
- [VGG16 5% single model](results/vgg16_seed42_sub5pct_20261027_095513/)
- [VGG16 5% five-fold CV, final run](results/vgg16_cv5_seed42_sub5pct_20261027_122209/)
- [InceptionV3 full-data single model](results/inceptionv3_seed42_20261023_161640/)
- [DenseNet169 full-data single model](results/densenet169_seed42_20261023_165946/)
- [MobileNetV2 full-data single model](results/mobilenetv2_seed42_20261021_042340/)

Each result directory includes the applicable configuration, test metrics,
classification report, predictions, confusion matrices, training history,
and error-analysis files. Cross-validation directories additionally include
fold metrics, out-of-fold predictions, and ensemble evaluation files.

### VGG16 five-fold CV details

The final VGG16 CV run used the deterministic 5% subset (`586` images total):

- Development data for cross-validation: `498` images.
- Test data: `88` images.
- Folds: `5`.
- Batch size: `8`.
- Phase 1: up to `3` epochs, early stopping patience `1`.
- Phase 2: up to `5` epochs, early stopping patience `2`.
- Backbone: ImageNet-pretrained VGG16 with VGG16
  `preprocess_input`.
- Pooled out-of-fold accuracy: `61.24%`.
- Mean test accuracy across the five fold models: `62.50%`.
- Final test ensemble accuracy: **`76.14%`** (`67/88` correct).
- Final test ensemble macro-F1: **`75.95%`**.
- Final test ensemble weighted-F1: `76.37%`.
- Final test ensemble macro-AUC: `95.71%`.

The `76.14%` value is the final reported VGG16 CV result because it comes
from averaging the predictions of all five fold models on the untouched test
set. It should not be compared directly with the single-model result or with
the full-data models without accounting for the different training data.

## Notebooks

- [ResNet50V2 local notebook](notebooks/ResNet50V2_HD_local.ipynb)
- [VGG16 local notebook](notebooks/VGG16_HD_local.ipynb)
- [InceptionV3 notebook](notebooks/InceptionV3_HD.ipynb)
- [DenseNet169 notebook](notebooks/DenseNet169_HD.ipynb)
- [MobileNetV2 notebook](notebooks/MobileNetV2_HD.ipynb)

The notebooks document preprocessing, augmentation, two-phase transfer
learning, early stopping, checkpoint selection, and evaluation.

## Reproducibility and repository scope

The 5% subset is selected with `SEED = 42` and
`SUBSET_FRACTION = 0.05`. Full-data runs use
`data_processed/dataset_224_v3`. Set the dataset path in the notebook before
running. Dataset archives, image files, `.keras`/`.h5` checkpoints, caches,
and editor files are intentionally ignored because of size and reproducibility
storage constraints.
