# Hair Disease Research

This repository contains the reproducible notebooks and exported evaluation
artifacts for the hair-disease classification experiments.

## Included experiments

- `ResNet50V2` on the fixed 3% stratified subset:
  - single-model evaluation
  - five-fold cross-validation
- `VGG16` evaluation and five-fold cross-validation on the fixed 5% subset

The notebooks document the dataset configuration, preprocessing, training
phases, metrics, confidence intervals, confusion matrices, and error analysis.

## Repository scope

The repository includes notebooks, configuration files, metrics, predictions,
plots, and cross-validation summaries. Dataset images and large `.keras`
model/checkpoint files are intentionally excluded. Keep those artifacts in
Drive or another approved artifact store and use the paths/configuration
recorded in each run's `config.json` or `cv_config.json`.

The subset folders are generated deterministically from the source dataset
using the recorded seed and fraction:

- ResNet50V2: `SUBSET_FRACTION = 0.03`
- VGG16: the recorded 5% run configuration

## Notebooks

- `notebooks/ResNet50V2_HD_local.ipynb`
- `notebooks/VGG16_HD_local.ipynb`
