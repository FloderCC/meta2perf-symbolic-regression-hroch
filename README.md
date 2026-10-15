# Meta2Perf: tabular-classifier performance modeling

Code and results for the manuscript **“Exploring Tabular Classifier
Performance from Dataset and Algorithm Descriptors with Symbolic Regression.”**

The project builds a meta-dataset linking 12 dataset meta-features, seven
heuristic algorithm descriptors, and the held-out Matthews correlation
coefficient (MCC) of tuned classifiers. It then analyzes feature associations,
fits symbolic expressions with HROCH and DEAP, evaluates five reference
meta-regressors, and studies fixed-equation coefficient recalibration through
leave-one-dataset-out (LODO) experiments.

## Current evaluation results

The completed analysis is stored in
[`src/outputs/model_evaluation/final/`](src/outputs/model_evaluation/final/).
Its [row-wise metrics](src/outputs/model_evaluation/final/positive/rowwise/metrics.csv)
and [LODO metrics](src/outputs/model_evaluation/final/positive/lodo/comparison.csv) use
the corrected meta-dataset and final equations. Row-wise predictions are raw;
primary LODO results use training-fold clipping. The tables below retain the
earlier experimental results as historical records.

## Main archived results

All values below come from existing archived logs or summary tables and are
rounded to three decimal places. The complete meta-dataset contains 25,836
rows; the reported meta-regression analyses retain the 23,323 rows with
`MCC > 0`.

These tables document the historical experiments. The selected meta-dataset
and equation modules now contain the corrected descriptors and final constants;
the commands below reproduce the current analyses from those inputs.

### Joint descriptors: row-wise test partition

These analyses use 12 dataset and seven algorithm descriptors, with 18,658
training rows and 4,665 test rows.

| Analysis and archived source | Test \(R^2\) | Adjusted \(R^2\) | sMAPE | MAE |
|---|---:|---:|---:|---:|
| [HROCH symbolic regression](src/outputs/hroch/archived/run.log) | 0.607 | 0.606 | 0.284 | 0.137 |
| [DEAP GP, best final equation](src/outputs/genetic_programming/configuration_02/archived/summary.csv) | 0.720 | 0.719 | 0.253 | 0.115 |
| [Random forest and KernelSHAP analysis](src/outputs/random_forest_shap/archived/run.log) | 0.923 | 0.923 | 0.131 | 0.046 |

The reported DEAP equation corresponds to configuration 02, trained with seed
42 and a 5,000-generation budget. Its archived individual has 400 nodes. The
archive also includes
the [300-configuration screening summary](src/outputs/genetic_programming/hyperparameter_search/archived/summary.csv)
and [top-10 refinement summary](src/outputs/genetic_programming/top_configuration_refinement/archived/summary.csv).

### Dataset-only reference regressors

These marginal baselines use the 12 dataset descriptors and the same row-wise
split. Their source is the [archived benchmark log](src/outputs/baselines/dataset_only/archived/run.log).

| Regressor | Test \(R^2\) | Adjusted \(R^2\) | sMAPE | MAE |
|---|---:|---:|---:|---:|
| Linear regression | 0.029 | 0.026 | 0.436 | 0.244 |
| Decision tree | 0.455 | 0.453 | 0.317 | 0.158 |
| Random forest | 0.468 | 0.467 | 0.310 | 0.156 |
| Gradient boosting | 0.470 | 0.468 | 0.312 | 0.158 |
| Multilayer perceptron | 0.193 | 0.191 | 0.393 | 0.211 |

sMAPE in the two tables above is a ratio: 0.284 corresponds to 28.4%. The code
also supports benchmarking all five regressors with joint descriptors; the
existing archive contains their dataset-only benchmark and the separate joint
random-forest/KernelSHAP run.

### Fixed-equation LODO recalibration

Both archived experiments completed all 65 dataset folds and cover 23,323
rows, with no invalid predictions. In each fold, all rows of one dataset are
held out and coefficients are recalibrated on the other datasets. DE denotes
differential evolution; EGWO denotes enhanced grey wolf optimization.
`Original constants` evaluates the supplied equation without recalibration,
and `Training mean` predicts the mean MCC of that fold's training rows.

The following scores pool the held-out predictions across all folds; they are
not averages of the 65 fold scores. LODO sMAPE is stored as a **percentage**.

| Equation | Method | Pooled \(R^2\) | MAE | RMSE | sMAPE (%) |
|---|---|---:|---:|---:|---:|
| Extended | Original constants | 0.734 | 0.113 | 0.150 | 24.770 |
| Extended | DE | 0.723 | 0.116 | 0.153 | 25.269 |
| Extended | EGWO | 0.730 | 0.114 | 0.151 | 24.895 |
| HROCH | Original constants | 0.636 | 0.133 | 0.175 | 27.789 |
| HROCH | DE | 0.625 | 0.135 | 0.178 | 28.180 |
| HROCH | EGWO | 0.626 | 0.135 | 0.178 | 28.176 |
| Both | Training mean | -0.015 | 0.253 | 0.293 | 44.587 |

Sources: [extended-equation summary](src/outputs/recalibration/extended_equation/archived/summary.csv)
and [HROCH-equation summary](src/outputs/recalibration/hroch_equation/archived/summary.csv),
using `prediction_mode=raw`. These summaries also retain Spearman correlation,
macro fold MAE, worst fold MAE and median fold `R^2`.

<details>
<summary>LODO results with prediction clipping</summary>

These are the archived `prediction_mode=clipped` rows. Predictions are clipped
to the target range of the corresponding training fold for reporting;
coefficient optimization still minimizes raw training MSE.

| Equation | Method | Pooled \(R^2\) | MAE | RMSE | sMAPE (%) |
|---|---|---:|---:|---:|---:|
| Extended | Original constants | 0.736 | 0.111 | 0.150 | 24.603 |
| Extended | DE | 0.726 | 0.113 | 0.152 | 25.035 |
| Extended | EGWO | 0.731 | 0.112 | 0.151 | 24.716 |
| HROCH | Original constants | 0.639 | 0.131 | 0.175 | 27.592 |
| HROCH | DE | 0.628 | 0.133 | 0.177 | 27.985 |
| HROCH | EGWO | 0.628 | 0.133 | 0.177 | 27.982 |
| Both | Training mean | -0.015 | 0.253 | 0.293 | 44.587 |

</details>

In these archived runs, neither recalibration method improves pooled raw
`R^2` or MAE over the original constants for either equation. The extended
equation is a separately written simplification with rounded constants derived
from the reported DEAP equation; its predictions are not interchangeable with the
full GP individual in the first table.

The equation structures and original coefficients were selected on the full
corpus. These LODO scores therefore evaluate coefficient recalibration
conditional on supplied equations, rather than nested validation of equation
discovery. They use a different evaluation protocol from the row-wise test
scores above.

## Repository layout

```text
.
├── CITATION.cff
├── README.md
├── requirements.txt
├── tools/
│   └── validate_repository.py
└── src/
    ├── experiment_config.py
    ├── paths.py
    ├── pipeline/       # preparation, final join and meta-model evaluation
    ├── experiments/
    │   ├── hroch/
    │   ├── baselines/
    │   ├── genetic_programming/
    │   └── recalibration/
    ├── analysis/       # explanations and figures
    ├── equations/      # extended and HROCH fixed equations
    ├── datasets/       # source links; raw CSV files are not redistributed
    ├── docs/           # tables and reproducibility details
    ├── outputs/        # tables, logs, figures and checkpoints grouped by run
    ├── cache/          # reusable generated GP populations
    └── utils/
```

All commands below are run from the repository root with `python -m ...`.
Paths are resolved relative to the repository, so it is not necessary to
change the working directory to `src/`.

Current modules and output conventions are documented in
[Repository organization](src/docs/repository-organization.md).

## Installation

Python 3.12 or newer is required by the pinned dependencies. The local
maintenance environment uses CPython 3.12.10.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original experiment snapshot did not record exact dependency versions.
The requirements file pins the current maintenance environment; it cannot
reconstruct the historical environment bit-for-bit.

## Reproduce the current analyses

The saved meta-dataset already contains the corrected complexity and
regularization descriptors. The runner evaluates the final equations, fits the
reference regressors, performs 65-fold LODO, and generates the correlation,
directional and TreeSHAP analyses. It retains `MCC > 0` and the shared row-wise
split. Symbolic coefficients are fitted only within LODO training folds.
Choose an output directory that does not already contain results:

```bash
python -m src.pipeline.evaluate_meta_models --dry-run --output-dir src/outputs/model_evaluation/reproduced_01 --methods DE EGWO --n-jobs 16
python -m src.pipeline.evaluate_meta_models --output-dir src/outputs/model_evaluation/reproduced_01 --methods DE EGWO --n-jobs 16
```

The protocol and output files are described in
[Reproducing the analyses](src/docs/reproducibility.md#meta-model-evaluation).

## Use the archived results

The repository includes the paper's meta-dataset, logs, SHAP values, and
figures. To select historical preparation tables explicitly in PowerShell:

```powershell
$env:META2PERF_INPUT_SOURCE = "archived"
```

These lightweight commands regenerate prediction and diagnostic figures:

```bash
python -m src.analysis.plot_hroch_predictions
python -m src.analysis.plot_symbolic_comparison
python -m src.analysis.plot_extended_equation_predictions
python -m src.analysis.plot_feature_correlations
python -m src.analysis.plot_shap_relevance
```

Generated figures are written to `src/outputs/figures/<analysis>/<run_id>/`.
Existing figures are preserved in the corresponding `archived/` directories.

The three equation-plot commands use the final HROCH and grouped DEAP equations
in `src/equations/` and the normal selected meta-dataset input bank. Model
evaluation is not required before plotting. The comparison uses the shared row-wise
split with seed 42 and writes `Fig5_symbolic_comparison.pdf`, a PNG copy and
`symbolic_comparison_metrics.csv` below
`src/outputs/figures/symbolic_comparison/<run_id>/`. Both X axes end at 1;
predicted MCC values remain unclipped. Its inputs are contained in the
repository. Optional `--data` and `--output-dir` arguments select an alternative
meta-dataset or a fresh output directory. Passing `--deap-summary` with
`src/outputs/genetic_programming/configuration_02/archived/summary.csv` explicitly
uses the historical DEAP prefix tree instead of the final grouped equation.
Reproducing its historical scores also requires its original input descriptors;
the source and checksum are documented in the
[reproducibility notes](src/docs/reproducibility.md#gp-selection-and-fixed-equation-analyses).
The correlation and SHAP plotting commands continue to use the selected input
bank and saved relevance tables, respectively.

The reference meta-regressors can be rerun with either descriptor set:

```bash
# Primary joint analysis: 12 dataset + 7 algorithm descriptors
python -m src.experiments.baselines.benchmark --feature-set joint

# Marginal dataset-only baseline
python -m src.experiments.baselines.benchmark --feature-set dataset-only
```

To refit the joint random forest and regenerate its KernelSHAP relevance
table, run the explanation stage before plotting the new relevance values:

```bash
python -m src.analysis.explain_random_forest
python -m src.analysis.plot_shap_relevance
```

The plotting command reads the latest generated SHAP table, falling back to
the archive when no new table exists. KernelSHAP explanation involves model
fitting and additional computation; it is separate from plotting saved values.

Fitting the HROCH symbolic regressor can take up to the configured 72-hour limit:

```bash
python -m src.experiments.hroch.train
```

## Rebuild the meta-dataset from raw datasets

Raw third-party datasets are not redistributed. Download them from the links
under `src/datasets/<dataset name>/README.md`, and place exactly one CSV file in
each corresponding directory. Select a fresh build name in PowerShell before
running the four stages:

```powershell
$env:META2PERF_INPUT_SOURCE = "build"
$env:META2PERF_DATA_RUN = "rebuild_01"
```

On other shells, export these same two environment variables. Then run:

```bash
python -m src.pipeline.tune_base_classifiers
python -m src.pipeline.evaluate_base_classifiers
python -m src.pipeline.extract_dataset_metafeatures
python -m src.pipeline.build_meta_dataset
```

This pipeline is computationally expensive: it spans 65 datasets, five
sampling fractions, five seeds, 16 classifiers, and hyperparameter searches.
The archived CSVs make rerunning it unnecessary for reproducing the
meta-analysis.

Evaluation and descriptor extraction both depend on tuning and may run in
either order. The final join needs both completed outputs. Keep the same
`META2PERF_DATA_RUN` for all four stages; its default is `current`.

One input bank is selected for the whole process. `META2PERF_INPUT_SOURCE=build`
requires the selected build, and `archived` selects the historical tables.
The default `auto` uses a build when `META2PERF_DATA_RUN` is explicitly set or
its directory is nonempty; otherwise it uses the archive. A missing build
table never falls back individually to a historical table.

Each new preparation table has a `.complete.json` companion recording its
checksum, input tables, raw-dataset checksums, code and package versions.
Readers reject incomplete or incompatible builds. Tuning also retains
`tuning_status.json`, including failures and fractions omitted because their
stratified split is infeasible. Only partial files produced by that invocation
are consolidated, and every feasible task must cover all configured classifiers.
Existing outputs are preserved: a failed preparation stage requires a fresh
build name rather than overwriting its partial files. Legacy tables without
completion metadata remain available in the archive.

## Additional experiments

The DEAP genetic-programming screening, refinement and long training stages
are separate modules:

```bash
python -m src.experiments.genetic_programming.search_hyperparameters
python -m src.experiments.genetic_programming.refine_top_configurations
python -m src.experiments.genetic_programming.train_configuration_02
```

Each stage writes below `src/outputs/genetic_programming/<stage>/<run_id>/`.
Refinement reads the latest screening summary, falling back to its archive,
and validates its compatibility before selecting configurations. Screening
evaluates up to 300 configurations for 100 generations each; refinement takes
the top 10 to 500 generations. Ranking uses training Huber loss, with test
R-squared as a secondary tie-breaker. The reported final equation uses the
manually specified configuration 02 for 5000 generations with seed 42. Its
training script does not automatically read the refinement summary.

GP caches, checkpoints, run manifests and new summaries record the metadataset
checksum, feature order, engine fingerprint and package versions. Compatible
screening checkpoints can continue in refinement. Generation budgets, logging
and resume controls are excluded from the engine fingerprint, so increasing
a run's budget can continue its checkpoint. Historical summaries can supply
hyperparameters only for their original input checksum; legacy caches and checkpoints
without compatibility metadata are preserved and are not reused.

The HROCH and extended-equation prediction plots evaluate formulas fixed in
their source files. They do not automatically load equations from new HROCH or
GP training runs. Transferring a newly selected equation is a separate manual
step.

The two fixed-equation LODO experiments are available as:

```bash
python -m src.experiments.recalibration.recalibrate_extended_equation_lodo
python -m src.experiments.recalibration.recalibrate_hroch_equation_lodo
```

They read the same selected meta-dataset and write below
`src/outputs/recalibration/<equation>/<run_id>/`. Their `--data`, `--output-dir`
and `--resume` options remain available. For a resumed LODO run, pass its
existing output directory explicitly with `--output-dir` and `--resume`.

LODO leaves out all rows of one dataset, recalibrates coefficients on the
remaining datasets with DE and EGWO, and repeats this for each dataset. The
equation structure and original coefficients were selected on the full corpus.
This evaluates recalibration conditional on a supplied fixed equation; it is
not nested validation of the full equation-discovery procedure. The two
equations are evaluated independently.

Experiment and figure run ids use UTC timestamps. `META2PERF_RUN_ID` can select
a specific run directory; use a new name for a fresh run. GP checkpoint resume
also requires its existing `RESUME_FROM_CHECKPOINT` setting to be enabled.

## Experimental protocol

- Each base-task subsample is capped at one million rows and sampled by class.
- Preprocessing converts infinities to missing values, ordinal-encodes
  non-numeric columns, and mean-imputes numeric missing values.
- To reproduce the historical protocol, preprocessing is applied to the whole
  subsample before descriptor extraction and the stratified 80/20 base-task
  split. The resulting representation remains fixed during classifier training
  and evaluation, so the descriptors characterize the data received by the
  classifiers.
- Hyperparameters are selected on the 80% base-task training partition using
  four-fold cross-validation and MCC.
- Search spaces with at most 20 Cartesian combinations use exhaustive
  `GridSearchCV`; larger spaces use `GASearchCV` with population 20 and at most
  ten generations.
- The selected configuration is fitted to the complete training partition and
  evaluated once on the held-out 20%.

The unit of analysis is the preprocessed dataset. Findings describe associations
between its descriptors and MCC conditional on this shared preparation
procedure. Comparing alternative preprocessing strategies is outside the study's
scope. Because encoding and imputation use the complete subsample, these scores
do not evaluate an end-to-end procedure fitted solely on raw training data.
At the meta-level, the 80/20 split is row-wise and describes interpolation
within the corpus; dataset identities can occur in both partitions.

The symbolic-regression run uses HROCH with population 1,000, tournament 500,
30 neighbors, constants in `[-1, 5]`, `alpha=0.01`, `beta=0.9`,
`pretest_size=2`, `sample_size=200`, MSE, double precision, random state 42,
and a 259,200-second limit. Under HROCH's 64-row batch semantics, the last two
settings correspond to 128 and 12,800 rows. These settings were selected
exploratorily to balance predictive fit and equation complexity; they were not
the result of an exhaustive search.

## Documentation

- [Meta-model evaluation protocol and outputs](src/docs/reproducibility.md#meta-model-evaluation)
- [Datasets and targets](src/docs/datasets.md)
- [Base-classifier hyperparameter spaces](src/docs/ml-hyperparams.md)
- [Algorithm descriptor definitions](src/docs/model-descriptors.md)
- [Reference meta-regressor grids](src/docs/reference-meta-regressor-hyperparams.md)
- [Reproducibility and validity notes](src/docs/reproducibility.md)
- [Repository organization and entry points](src/docs/repository-organization.md)
- [Archived tables and experiment outputs](src/outputs/README.md)
- [Archived experiment logs](src/outputs/archived_logs.md)

## Validation

Run the dependency-free structural checks with:

```bash
python tools/validate_repository.py
python -m compileall -q src tools
```

These checks validate source syntax, archived CSV schemas and row counts,
and documented dataset coverage.
They do not rerun model training.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff). GitHub exposes it through
the repository's **Cite this repository** menu.

## Data and licensing

The repository contains derived experimental outputs, but not the raw
third-party datasets. Each source dataset remains subject to its provider's
terms. No software license was present in the original project snapshot, so
none has been invented for this release; absent a license, normal copyright
restrictions apply.
