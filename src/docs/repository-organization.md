# Repository organization

Source modules, datasets, caches and generated artifacts are organized under
`src`. Repository checks are in `tools/`.

## Code directories

| Directory | Responsibility |
|---|---|
| `pipeline/` | Base-classifier tuning, held-out evaluation, dataset meta-features, meta-dataset construction and final meta-model evaluation. |
| `experiments/hroch/` | HROCH symbolic-regression training. |
| `experiments/baselines/` | Training-only reference-configuration selection and meta-regressor comparison. |
| `experiments/genetic_programming/` | GP hyperparameter screening, refinement and three configured long runs. |
| `experiments/recalibration/` | Fixed-equation leave-one-dataset-out coefficient recalibration. |
| `analysis/` | Model diagnostics, directional checks, random-forest explanations and prediction, correlation and SHAP figures. |
| `equations/` | Extended equation with 19 constants and HROCH equation with six constants. |
| `utils/` | Shared data processing, base-classifier construction, regression metrics and GP operations. |
| `datasets/` | Existing source CSV files and per-dataset documentation; their locations are unchanged. |
| `docs/` | Scientific definitions, execution protocols and repository guides. |

`paths.py` centralizes all locations. `experiment_config.py` retains the dataset,
classifier, seed, fraction and split settings. Package `__init__.py` files allow
entry points to be invoked with `python -m` from the repository root.

## Current entry points

Paths below are relative to `src`; invoke modules with `python -m` from the
repository root.

| Module | Purpose |
|---|---|
| `pipeline/tune_base_classifiers.py` | Tune base classifiers on each task. |
| `pipeline/evaluate_base_classifiers.py` | Evaluate the selected base classifiers. |
| `pipeline/extract_dataset_metafeatures.py` | Extract dataset descriptors. |
| `pipeline/build_meta_dataset.py` | Join dataset descriptors, model descriptors and MCC targets. |
| `pipeline/evaluate_meta_models.py` | Evaluate symbolic and reference predictors, LODO and diagnostics. |
| `experiments/hroch/train.py` | Search for a HROCH symbolic expression. |
| `experiments/baselines/benchmark.py` | Compare reference meta-regressors using joint or dataset-only descriptors. |
| `experiments/baselines/select_configurations.py` | Select reference configurations using the training partition. |
| `experiments/baselines/benchmark_lodo.py` | Fit and evaluate fixed reference configurations across dataset folds. |
| `experiments/genetic_programming/search_hyperparameters.py` | Screen DEAP configurations. |
| `experiments/genetic_programming/refine_top_configurations.py` | Refine the selected DEAP configurations. |
| `experiments/genetic_programming/train_configuration_01.py` | Run the first configured DEAP search. |
| `experiments/genetic_programming/train_configuration_02.py` | Run the second configured DEAP search. |
| `experiments/genetic_programming/train_configuration_03.py` | Run the third configured DEAP search. |
| `experiments/recalibration/recalibrate_extended_equation_lodo.py` | Fit grouped DEAP coefficients within dataset folds. |
| `experiments/recalibration/recalibrate_hroch_equation_lodo.py` | Fit HROCH coefficients within dataset folds. |
| `analysis/model_diagnostics.py` | Produce correlations, derivative checks and TreeSHAP. |
| `analysis/explain_random_forest.py` | Fit the joint Random Forest and calculate KernelSHAP values. |
| `analysis/plot_feature_correlations.py` | Plot descriptor correlations. |
| `analysis/plot_shap_relevance.py` | Plot saved SHAP relevance values. |
| `analysis/plot_hroch_predictions.py` | Plot the final HROCH predictions. |
| `analysis/plot_extended_equation_predictions.py` | Plot the final grouped DEAP predictions. |
| `analysis/plot_symbolic_comparison.py` | Plot both symbolic predictors on the shared test partition. |

The final evaluation stage is `pipeline/evaluate_meta_models.py`, invoked after
`pipeline/build_meta_dataset.py` or directly with the prepared input tables.
It calls `experiments/baselines/select_configurations.py` for reference settings,
the existing LODO experiments for grouped fitting, and
`analysis/model_diagnostics.py` for correlations, derivative checks and TreeSHAP.
The command and protocol are described in
[Reproducibility](reproducibility.md#meta-model-evaluation).

`analysis/plot_symbolic_comparison.py` independently generates the two-panel
HROCH/DEAP prediction figure from the final canonical equations in `equations/`
and the selected prepared meta-dataset. Its PDF, PNG and train/test metric table
are written below `outputs/figures/symbolic_comparison/<run_id>/`. Both X axes
end at 1. `--deap-summary` explicitly selects a historical prefix tree instead;
its source checksum is recorded in the
[reproducibility notes](reproducibility.md#gp-selection-and-fixed-equation-analyses).

## Output locations

Existing artifacts are stored under these paths, relative to `src/outputs`:

| Output | Location |
|---|---|
| Four preparation/meta-dataset tables | `metadataset/archived/` |
| Completed final meta-model evaluation | `model_evaluation/final/` |
| HROCH log | `hroch/archived/run.log` |
| Dataset-only reference-regressor log | `baselines/dataset_only/archived/run.log` |
| Random-forest log and SHAP table | `random_forest_shap/archived/` |
| GP screening summary | `genetic_programming/hyperparameter_search/archived/summary.csv` |
| GP refinement summary | `genetic_programming/top_configuration_refinement/archived/summary.csv` |
| Three GP long-run summaries | `genetic_programming/configuration_01/archived/`, `configuration_02/archived/`, `configuration_03/archived/` |
| Extended-equation LODO outputs | `recalibration/extended_equation/archived/` |
| HROCH-equation LODO outputs | `recalibration/hroch_equation/archived/` |
| Correlation figures | `figures/feature_correlations/archived/` |
| SHAP figures | `figures/shap_relevance/archived/` |
| HROCH prediction figures | `figures/hroch_predictions/archived/` |
| Extended-equation prediction figures and CSVs | `figures/extended_equation_predictions/archived/` |
| Preprocessing audit reports | `preprocessing_audit/archived/<original_timestamp>/` |

The [output index](../outputs/README.md) lists result files and their contents.
The [log index](../outputs/archived_logs.md) links the available experiment logs.

New experiments and figures use `src/outputs/<family>/<run_id>/`, with a UTC
timestamp run id. For example:

```text
src/outputs/genetic_programming/configuration_01/<run_id>/
    summary.csv
    run.log
    hall_of_fame.csv
    final_population.csv       # only when enabled
    checkpoints/
```

Screening and refinement produce one log and candidate table per evaluated
configuration. LODO runs also retain their `run_config.json`, `folds/`,
predictions, metrics, coefficients and convergence. Other experiments retain
the outputs they already produced.

Meta-model evaluation uses `src/outputs/model_evaluation/<run_id>/`, with the
completed final results in `model_evaluation/final/`. Each run records its input
checksums, configuration and completion status. Start a fresh directory for a
new execution. Fold and model checkpoints are local execution state; summaries,
predictions and settings are retained as outputs.

GP initial-population caches use `src/cache/gp_initial_populations/`.

## Input selection and run ids

- `META2PERF_DATA_RUN` selects the shared preparation build below
  `src/outputs/metadataset/`; its default is `current`. All four preparation
  stages must use the same value. Choose a fresh name to preserve earlier builds.
- `META2PERF_INPUT_SOURCE` selects one bank for all preparation inputs: `build`,
  `archived`, or `auto` (the default). Automatic selection uses the build when
  `META2PERF_DATA_RUN` is set or its directory is nonempty; otherwise it uses the
  archive. Missing build tables cause an error and never fall back individually.
- New preparation tables require `.complete.json` companions, recording hashes,
  lineage, raw-data context, code and versions. Tuning also records task outcomes
  in `tuning_status.json`. Partial tables are retained and rejected by readers;
  choose a fresh build name after a failure. Legacy archived tables are read
  without modifying them or inventing retrospective metadata.
- `META2PERF_RUN_ID` selects the experiment/figure run name. If unset, a UTC
  timestamp is generated. It does not select the preparation input build.
- Both run labels must be single directory names and cannot select `archived`.
- SHAP plotting reads the latest generated SHAP table, falling back to the
  archive. GP refinement follows the same rule for the screening summary and
  validates its context before use. Historical screening summaries are usable
  only with the original data checksum pinned in `utils/gp_integrity.py`.
  Checkpoints require matching compatibility metadata before reuse.
- To resume a GP run, select its existing `META2PERF_RUN_ID` and enable its
  `RESUME_FROM_CHECKPOINT` setting. Its `run_context.json`, checkpoints, cache
  metadata and summaries must match the data, feature order, engine and package
  versions. Compatible screening checkpoints can continue in refinement.
  Generation, logging and resume controls are excluded from the engine hash.
  To resume LODO, pass its existing
  `--output-dir` together with `--resume`; its data/code/settings checks remain.

The structural validator is `tools/validate_repository.py`. Preprocessing-audit
reports are available under `outputs/preprocessing_audit/archived/`. The README
lists the current module commands.
