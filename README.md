# Meta2Perf: tabular-classifier performance modeling

Code and archived results for the manuscript **“Exploring Tabular Classifier
Performance from Dataset and Algorithm Descriptors with Symbolic Regression.”**

The project builds a meta-dataset linking 12 dataset meta-features, seven
heuristic algorithm descriptors, and the held-out Matthews correlation
coefficient (MCC) of tuned classifiers. It then analyzes feature associations,
fits an explicit symbolic-regression expression, and evaluates five reference
meta-regressors.

## Main archived results

| Analysis | Test \(R^2\) | Adjusted \(R^2\) | sMAPE | MAE |
|---|---:|---:|---:|---:|
| Symbolic regression, joint descriptors | 0.607 | 0.606 | 0.284 | 0.137 |
| Random forest, joint descriptors | 0.923 | 0.922 | 0.132 | 0.047 |
| Random forest, dataset-only baseline | 0.468 | 0.467 | 0.310 | 0.156 |

sMAPE is stored as a ratio rather than a percentage; 0.284 corresponds to
28.4%.

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
    ├── tune_base_classifiers.py
    ├── evaluate_base_classifiers.py
    ├── extract_dataset_metafeatures.py
    ├── build_meta_dataset.py
    ├── fit_symbolic_regression.py
    ├── benchmark_meta_regressors.py
    ├── explain_random_forest.py
    ├── plot_feature_correlations.py
    ├── plot_shap_relevance.py
    ├── plot_symbolic_regression.py
    ├── datasets/       # source links; raw CSV files are not redistributed
    ├── docs/           # tables and reproducibility details
    ├── logs/           # archived experiment logs
    ├── plots/          # archived manuscript figures
    ├── results/        # archived CSV outputs
    └── utils/
```

All commands below are run from the repository root with `python -m ...`.
Paths are resolved relative to the repository, so it is not necessary to
change the working directory to `src/`.

## Installation

Python 3.10 or newer is recommended.

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
The requirements file declares the necessary packages but cannot reconstruct
the historical environment bit-for-bit.

## Use the archived results

The repository includes the paper's meta-dataset, logs, SHAP values, and
figures. These lightweight commands evaluate the archived symbolic expression
and regenerate the three figures:

```bash
python -m src.plot_symbolic_regression
python -m src.plot_feature_correlations
python -m src.plot_shap_relevance
```

Generated files are written to `src/plots/pdf/` and `src/plots/png/`.

The reference meta-regressors can be rerun with either descriptor set:

```bash
# Primary joint analysis: 12 dataset + 7 algorithm descriptors
python -m src.benchmark_meta_regressors --feature-set joint

# Marginal dataset-only baseline
python -m src.benchmark_meta_regressors --feature-set dataset-only
```

Fitting the symbolic regressor can take up to the configured 72-hour limit:

```bash
python -m src.fit_symbolic_regression
```

## Rebuild the meta-dataset from raw datasets

Raw third-party datasets are not redistributed. Download them from the links
under `src/datasets/<dataset name>/README.md`, and place exactly one CSV file in
each corresponding directory. Then run:

```bash
python -m src.tune_base_classifiers
python -m src.evaluate_base_classifiers
python -m src.extract_dataset_metafeatures
python -m src.build_meta_dataset
```

This pipeline is computationally expensive: it spans 65 datasets, five
sampling fractions, five seeds, 16 classifiers, and hyperparameter searches.
The archived CSVs make rerunning it unnecessary for reproducing the
meta-analysis.

## Experimental protocol

- Each base-task subsample is capped at one million rows and sampled by class.
- Preprocessing converts infinities to missing values, ordinal-encodes
  non-numeric columns, and mean-imputes numeric missing values.
- To reproduce the historical protocol, preprocessing is applied to the whole
  subsample before its stratified 80/20 base-task split. This can leak
  distributional information. A new leakage-free study should fit all
  preprocessing operations inside the training folds.
- Hyperparameters are selected on the 80% base-task training partition using
  four-fold cross-validation and MCC.
- Search spaces with at most 20 Cartesian combinations use exhaustive
  `GridSearchCV`; larger spaces use `GASearchCV` with population 20 and at most
  ten generations.
- The selected configuration is fitted to the complete training partition and
  evaluated once on the held-out 20%.

The symbolic-regression run uses HROCH with population 1,000, tournament 500,
30 neighbors, constants in `[-1, 5]`, `alpha=0.01`, `beta=0.9`,
`pretest_size=2`, `sample_size=200`, MSE, double precision, random state 42,
and a 259,200-second limit. Under HROCH's 64-row batch semantics, the last two
settings correspond to 128 and 12,800 rows. These settings were selected
exploratorily to balance predictive fit and equation complexity; they were not
the result of an exhaustive search. The historical log contains its completion
timestamp but not a start timestamp, so the exact elapsed runtime cannot be
recovered.

## Documentation

- [Datasets and targets](src/docs/datasets.md)
- [Base-classifier hyperparameter spaces](src/docs/ml-hyperparams.md)
- [Algorithm descriptor definitions](src/docs/model-descriptors.md)
- [Reference meta-regressor grids](src/docs/reference-meta-regressor-hyperparams.md)
- [Reproducibility and validity notes](src/docs/reproducibility.md)

## Validation

Run the dependency-free structural checks with:

```bash
python tools/validate_repository.py
python -m compileall -q src tools
```

These checks validate source syntax, archived CSV schemas and row counts,
documented dataset coverage, and the absence of the withdrawn min–max analysis.
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