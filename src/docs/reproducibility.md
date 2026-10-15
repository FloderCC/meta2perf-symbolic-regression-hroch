# Reproducibility and validity notes

## Archived artifacts

The repository includes the complete derived meta-dataset used by the
manuscript, the preceding base-task and dataset-description tables, the final
symbolic-regression log, the dataset-only reference-regressor log, the joint
random-forest/KernelSHAP log, the SHAP relevance values, and the manuscript
figures.

The completed final evaluation is in
[`../outputs/model_evaluation/final/`](../outputs/model_evaluation/final/),
including final equation coefficients, row-wise
and LODO metrics, predictions, correlations and TreeSHAP values. Earlier
experiment outputs retain their original contents in the archive directories.
The prepared input is the shared
[meta-dataset](../outputs/metadataset/archived/meta_dataset.csv).

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

Our unit of analysis is the preprocessed dataset, and our objective is to
examine associations between its descriptors and the MCC values obtained under
a common experimental protocol. Encoding and imputation are performed before
descriptor extraction, and the resulting representation is subsequently held
fixed during classifier training and evaluation. This ordering ensures that
the descriptors characterize the data actually received by the classifiers.
Additional transformations after descriptor extraction could alter those
properties and introduce an unmodeled source of variation into the
descriptor–MCC relationship.

Accordingly, preprocessing is treated as a fixed component of the experimental
protocol, with the comparison of alternative preprocessing strategies outside
the scope of this study. Our findings are interpreted as associations
conditional on this shared data preparation procedure.

Encoding and imputation use the complete subsample, including the portion later
held out for classifier evaluation. The reported scores therefore do not
estimate an end-to-end raw-data preparation and training procedure fitted
solely within the training partitions.

## Current environment and preparation integrity

The pinned maintenance dependencies require Python 3.12 or newer; the local
environment is CPython 3.12.10. These pins describe the current environment,
not a reconstruction of unrecorded historical dependency versions.

Preparation readers select a single input bank: an explicit build, the archive,
or the automatic choice documented in the README. Missing files in a selected
build cause an error rather than per-table fallback. Every new table has a
completion companion that records content checksums, input lineage, raw-data
checksums, preparation code and package versions. Missing metadata, partial
tables, duplicate keys, non-finite required values and incompatible contexts
are rejected. Existing historical tables remain readable without retroactive
metadata changes.

Tuning checks all configured source CSV locations before starting, records
failed tasks and infeasible sampling fractions, and requires every feasible
task to return all configured classifiers. Any failed task prevents publication
of a consolidated tuning table. Evaluation and descriptor extraction must
cover its keys completely; the final join checks both coverages and enforces
many-to-one descriptor matching. Partial outputs are retained and cannot be
mistaken for completed tables or silently reused in another preparation run.

## GP selection and fixed-equation analyses

The GP sequence is screening (up to 300 configurations, 100 generations each),
top-10 refinement (500 generations) and independent long runs (5000 generations
for each of three manually specified configurations). Configuration ranking
uses training Huber loss and test R-squared only as a secondary tie-breaker;
the test partition therefore participates in that tie-breaking rule. Final
configuration scripts do not automatically import the refinement winners.

New GP artifacts are tied to the data checksum, ordered descriptors, engine
fingerprint and package versions. All GP stages use the same engine definition
so compatible screening checkpoints remain resumable in refinement. Budget,
logging and resume controls are excluded from that fingerprint. Legacy caches
and checkpoints are preserved but not reused without compatibility metadata.
Legacy screening summaries remain usable to select historical hyperparameters
only when the input matches the original data checksum pinned in
`src/utils/gp_integrity.py`.

Prediction plots use the final formulas and full-precision constants in
`src/equations/`; they do not automatically evaluate the result of a new
discovery run. Those same modules supply the formulas for coefficient-only LODO
recalibration. LODO removes all seeds, sample sizes and classifiers belonging
to the held-out dataset, and uses the other datasets to optimize coefficients.
The equation form and starting coefficients were selected on the dataset collection,
including the datasets later held out. LODO therefore evaluates conditional
coefficient recalibration, not nested validation of equation discovery.

`src.analysis.plot_symbolic_comparison` uses these canonical equations by
default. Its optional `--deap-summary PATH` argument evaluates a selected prefix
tree from a GP summary. Reproducing that summary's scores also requires the
matching input data and descriptor definitions.

## Meta-model evaluation

Run `src.pipeline.evaluate_meta_models` after preparing the meta-dataset, or
directly with the saved input tables. The selected meta-dataset already contains
the corrected complexity and regularization descriptors. This stage evaluates
the supplied final equations and reference regressors; it does not repair
descriptors, train base classifiers, or search for new symbolic structures.

Choose a fresh output directory, then validate and run:

```bash
python -m src.pipeline.evaluate_meta_models --dry-run --output-dir src/outputs/model_evaluation/reproduced_01 --methods DE EGWO --n-jobs 16
python -m src.pipeline.evaluate_meta_models --output-dir src/outputs/model_evaluation/reproduced_01 --methods DE EGWO --n-jobs 16
```

Without `--output-dir`, a timestamped run is created under
`src/outputs/model_evaluation/`. `--data PATH` selects another prepared input.
Resume only a run started with the same source version, inputs and settings:
repeat its command with the same output directory and add `--resume`.

- Retain the 23,323 observations with `MCC > 0` and the seed-42 row-wise
  partition of 18,658 training and 4,665 test observations.
- Evaluate the HROCH and grouped 19-coefficient DEAP equations directly;
  row-wise coefficients remain fixed and predictions are raw.
- Select reference configurations through
  `src.experiments.baselines.select_configurations`: reuse documented settings
  where available and select missing joint settings by four-fold training-only
  negative-MSE grid search. Keep dataset-only and joint settings separate.
- Hold out every classifier, subsample and seed of one dataset in each of 65
  LODO folds. Fit symbolic constants using DE and EGWO on the other datasets:
  population 30, 100 iterations, seed 42, and sign-preserving bounds of 0.1 to
  10 times the final reference constants. Fit reference regressors afresh with
  their fixed joint configurations.
- Optimize raw training MSE and report LODO predictions both raw and clipped to
  the training-fold target range. Estimate uncertainty with 2,000 whole-dataset
  bootstrap resamples.
- Use `src.analysis.model_diagnostics` for correlations, HROCH derivative signs
  and interventional TreeSHAP. TreeSHAP uses 100
  training rows as background and all 4,665 test rows of the joint Random Forest.

The run writes `positive/rowwise/metrics.csv` and
`positive/lodo/comparison.csv`, together with detailed
`positive/rowwise/`, `positive/lodo/`, and `positive/diagnostics/` artifacts.
New completed runs record `complete: true` and `evaluation_ready: true` in
`completion.json`. Check that both symbolic forms and all five reference
regressors cover the 65 held-out datasets with finite predictions and metrics.

The stored `model_evaluation/final/` artifacts contain the completed results.
Check their `complete` status, fold counts and finite-result indicators. Run
new evaluations in a fresh directory; resume is supported only when the inputs,
source code, versions and settings match the interrupted execution.

## Meta-level split

The regression target is divided into histogram bins using
`numpy.histogram_bin_edges(..., bins="auto")`. Rows are split 80/20 with
stratification over those bins and random state 42.

The split is row-wise. Dataset identities, classifier identities, subsamples,
and seed repetitions can appear in both partitions. Scores therefore describe
in-corpus interpolation. The complementary LODO analysis groups datasets while
remaining conditional on the supplied forms and configurations. Fully nested
dataset-grouped and joint dataset–algorithm holdouts assess stronger claims
about independently unseen tasks or algorithm families.

## Algorithm descriptors

The seven algorithm descriptors are compact heuristics. In particular,
`Processing Units Number` is the natural logarithm of a model-specific
structural-size proxy. It is not measured hardware utilization, latency, FLOPs,
or a comparable parameter count. The full formulas and encodings are listed in
[`model-descriptors.md`](model-descriptors.md).
