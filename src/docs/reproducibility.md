# Reproducibility and validity notes

## Archived artifacts

The repository includes the complete derived meta-dataset used by the
manuscript, the preceding base-task and dataset-description tables, the final
symbolic-regression log, the dataset-only reference-regressor log, the joint
random-forest/KernelSHAP log, the SHAP relevance values, and the manuscript
figures.

Raw source datasets are not redistributed because they originate from multiple
third-party providers. Source links are retained in the per-dataset README
files.

## Row counts

| Artifact | Rows |
|---|---:|
| Base-classifier hyperparameters | 25,836 |
| Base-classifier performance | 25,836 |
| Dataset meta-features | 1,615 |
| Complete meta-dataset | 25,836 |
| Retained meta-regression rows (`MCC > 0`) | 23,323 |
| Meta-training partition | 18,658 |
| Meta-test partition | 4,665 |

In the complete meta-dataset, 1,960 rows have MCC equal to zero and 553 have a
negative MCC. These are valid recorded outcomes. Their exclusion is an explicit
analytical choice in the reported regression analyses, not a data-cleaning
operation.

## Base-task preprocessing

The code reproduces the historical sequence:

1. cap a dataset at one million rows through class-wise sampling;
2. draw the requested class-wise sampling fraction;
3. replace infinities, ordinal-encode non-numeric columns, and mean-impute;
4. create a stratified 80/20 train/test split;
5. tune on the training partition with four-fold cross-validation;
6. refit the selected configuration and evaluate on the held-out partition.

Because preprocessing precedes the train/test split, its fitted values can use
information from the held-out portion. This is documented to preserve
reproducibility. A future replication intended to estimate leakage-free
generalization should place encoding and imputation inside a fitted pipeline
within every training fold.

## Meta-level split

The regression target is divided into histogram bins using
`numpy.histogram_bin_edges(..., bins="auto")`. Rows are split 80/20 with
stratification over those bins and random state 42.

The split is row-wise. Dataset identities, classifier identities, subsamples,
and seed repetitions can appear in both partitions. Scores therefore describe
in-corpus interpolation. Dataset-grouped and joint dataset–algorithm holdouts
are required before making claims about unseen tasks or algorithm families.

## Algorithm descriptors

The seven algorithm descriptors are compact heuristics. In particular,
`Processing Units Number` is the natural logarithm of a model-specific
structural-size proxy. It is not measured hardware utilization, latency, FLOPs,
or a comparable parameter count. The full formulas and encodings are listed in
[`model-descriptors.md`](model-descriptors.md).

## Maintenance corrections

The publication version makes deterministic maintenance corrections while
preserving every archived result:

- paths are repository-relative and work from the repository root;
- the evaluation-stage size limiter receives the required random seed;
- the configured name `HistGradientBoosting` is used consistently;
- comments match the implemented preprocessing, cross-validation, descriptor
  formulas, and HROCH batch semantics;
- joint and dataset-only reference analyses are separate command-line modes;
- withdrawn min–max scripts, logs, and plots are excluded;
- script and result names are descriptive and documented.

These corrections do not retroactively alter the archived CSVs or reported
metrics.
