# Archived result files

| File | Description |
|---|---|
| `base_classifier_hyperparameters.csv` | Selected classifier configuration for each seed, dataset, sampling fraction, and model. |
| `base_classifier_performance.csv` | Held-out base-task metrics for the selected configurations. |
| `dataset_metafeatures.csv` | Dataset descriptions computed for each seed, dataset, and sampling fraction. |
| `meta_dataset.csv` | Final join used by the paper: identifiers, 12 dataset meta-features, seven algorithm descriptors, and MCC. |
| `joint_random_forest_shap.csv` | Mean absolute KernelSHAP values from the joint random-forest analysis. |

These files are archived outputs, not synthetic examples. The original raw
datasets are not included. See [`../docs/reproducibility.md`](../docs/reproducibility.md)
for row counts, filtering, split details, and validity limitations.
