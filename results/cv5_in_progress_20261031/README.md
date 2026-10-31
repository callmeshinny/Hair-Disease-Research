# In-progress 5-fold cross-validation

This directory is an interim snapshot of a run that was still active when this
branch was created. It is not a completed experiment and must not be used for
model comparison or cited as a final result.

Configuration:

- Models requested: MobileNetV2, InceptionV3, DenseNet169.
- Splitter: stratified 5-fold image-level cross-validation.
- Development images: 10,800 (`train` and `val` pooled).
- Evaluation images per fold: 2,160.
- Training images per fold: 8,640.
- Epochs requested: 20.
- The held-out 1,200-image test set is excluded from this cross-validation.

The snapshot contains the run configuration, class list, deterministic fold
assignments, and the completed training-history file for MobileNetV2 fold 1.
It deliberately excludes the local image manifest because it contains
machine-specific absolute file paths. Once all five folds for all models have
completed, final metrics should be uploaded in a separate completed-results
directory.
