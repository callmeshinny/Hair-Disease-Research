# Hair Disease Research

Reproducible TensorFlow/Keras experiments for classifying 10 hair-disease
categories. The repository stores notebooks and lightweight evaluation
artifacts; image datasets and large model files are kept outside GitHub.

## Dataset protocol

The source dataset v3 contains **11,693 images**:

| Split | Images |
|---|---:|
| Train | 8,185 |
| Validation | 1,755 |
| Test | 1,753 |
| **Total** | **11,693** |

The fixed 5% stratified subset contains **586 images**:
**410 train, 88 validation, and 88 test**. A run must not be compared with
another run without checking this split and dataset-size difference.

## Recorded model results

Accuracy and macro-F1 below are from the saved test metrics for the linked
run. Percentages are rounded to two decimals.

| Model | Dataset used | Train | Val | Test | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| ResNet50V2 | Fixed 5% subset | 410 | 88 | 88 | 69.32% | 68.41% |
| VGG16 | Full dataset v3 | 8,185 | 1,755 | 1,753 | 98.97% | 98.97% |
| InceptionV3 | Full dataset v3 | 8,185 | 1,755 | 1,753 | 99.20% | 99.19% |
| DenseNet169 | Full dataset v3 | 8,185 | 1,755 | 1,753 | 98.46% | 98.45% |
| MobileNetV2 | Full dataset v3 | 8,185 | 1,755 | 1,753 | 98.86% | 98.85% |

**Important:** this is not a controlled leaderboard because ResNet50V2 was
evaluated on 5% while the other listed single-model runs used the full
dataset. For a fair model comparison, train every architecture on the same
fixed subset or every architecture on the full dataset.

## Notebooks

- [`ResNet50V2_HD_local.ipynb`](notebooks/ResNet50V2_HD_local.ipynb) -
  local Mac/M2 notebook using the fixed 5% subset.
- [`VGG16_HD_local.ipynb`](notebooks/VGG16_HD_local.ipynb) - VGG16 local
  training and evaluation pipeline.
- [`DenseNet169_HD.ipynb`](notebooks/DenseNet169_HD.ipynb) - DenseNet169
  training and evaluation pipeline.
- [`MobileNetV2_HD.ipynb`](notebooks/MobileNetV2_HD.ipynb) - MobileNetV2
  training and evaluation pipeline.

Each notebook records preprocessing, augmentation, training phases, model
selection, metrics, confidence intervals, confusion matrices, and error
analysis.

## Results

- [`ResNet50V2 5% single model`](results/resnet50v2_seed42_sub5pct_20261027_090212/)
- [`ResNet50V2 5% five-fold CV`](results/resnet50v2_cv5_seed42_sub5pct_20261027_090425/)
- [`DenseNet169 full-dataset run`](results/densenet169_seed42_20261023_165946/)
- [`MobileNetV2 full-dataset run`](results/mobilenetv2_seed42_20261021_042340/)
- Existing VGG16 artifacts are under `results/`; the linked configuration
  identifies the dataset used by each run.

Each result folder includes configuration, class counts, test metrics,
classification reports, predictions, training history, confusion matrices,
and error-analysis files where available. `.keras`, `.h5`, and other large
model/checkpoint binaries are intentionally excluded.
