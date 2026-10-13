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
only when the supplied metadataset matches the archived metadataset.

Prediction plots evaluate the fixed formulas already written in their source;
they do not automatically evaluate the result of a new discovery run. The
fixed-equation modules supply the formulas for coefficient-only LODO
recalibration. LODO removes all seeds, sample sizes and classifiers belonging
to the held-out dataset, and uses the other datasets to optimize coefficients.
The equation form and starting coefficients were selected on the full corpus,
including the datasets later held out. LODO therefore evaluates conditional
coefficient recalibration, not nested validation of equation discovery.

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

## Repository reorganization

Source entry points now live in `src/pipeline/`, `src/experiments/` and
`src/analysis/`; fixed equations live in `src/equations/`. Outputs are grouped
under `src/outputs/` by experiment, with existing files in `archived/` folders.
The input tables, numerical results, fold checkpoints, logs and figures were
moved without rewriting their contents. Paths recorded inside those historical
files still describe the original layout.

The [organization guide](repository-organization.md) lists module names and
run conventions. The [move inventory](reorganization_manifest.csv) records
every pre-existing file in `src`, its new location, original size and SHA-256
where recorded. Raw source CSV files stayed in place and were inventoried by
path and size without hashing their contents.

This reorganization was reviewed through file inspection only. No Python code,
training, tests, compilation, imports or validation scripts were executed.
