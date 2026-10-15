# Experiment outputs

All generated tables, logs, figures and checkpoints are grouped here by
experiment and run. Existing outputs are preserved in `archived/` directories.
New experiment and figure runs use UTC timestamp ids by default.

## Final meta-model evaluation

[`model_evaluation/final/`](model_evaluation/final/) contains the completed
evaluation of the corrected meta-dataset and final equations:

| Artifact | Contents |
|---|---|
| [`positive/rowwise/metrics.csv`](model_evaluation/final/positive/rowwise/metrics.csv) | Train/test metrics for the two symbolic equations and both reference-regressor feature sets. |
| [`positive/lodo/comparison.csv`](model_evaluation/final/positive/lodo/comparison.csv) | Dataset-grouped metrics, bootstrap intervals, and raw/clipped sensitivity. |
| [`metadataset/archived/meta_dataset.csv`](metadataset/archived/meta_dataset.csv) | Shared prepared input, including nonpositive MCC rows excluded from the analyses. |
| [`positive/rowwise/coefficients.json`](model_evaluation/final/positive/rowwise/coefficients.json) | Supplied final HROCH and grouped DEAP coefficients. |
| `positive/rowwise/` | The shared split, reference configurations, predictions and symbolic comparison figure. |
| `positive/lodo/` | Per-fold metrics, predictions, coefficients and configurations for both equations and five reference regressors. |
| `positive/diagnostics/` | Correlations, HROCH derivatives and interventional TreeSHAP results. |
| [`completion.json`](model_evaluation/final/completion.json) | Recorded completion status and input/code provenance. |

The analyses retain 23,323 observations with `MCC > 0`. Row-wise metrics use
raw predictions; the primary LODO comparison clips predictions to each
training-fold target range. Confirm the recorded `complete` status, coverage of
all 65 held-out datasets, and finite predictions and metrics.

Read the saved final results directly, or use a fresh output directory to run
the evaluation. New runs from `src.pipeline.evaluate_meta_models` are written
under `model_evaluation/<run_id>/` and record their `evaluation_ready` status.

Resume checkpoints in `folds/` and `checkpoints/` remain local, recoverable
execution state; final metrics, predictions and settings are retained alongside
the results. The [evaluation protocol](../docs/reproducibility.md#meta-model-evaluation)
documents the command and interpretation.

## Earlier experiment outputs

| File | Description |
|---|---|
| `metadataset/archived/base_classifier_hyperparameters.csv` | Selected classifier configuration for each seed, dataset, sampling fraction, and model. |
| `metadataset/archived/base_classifier_performance.csv` | Held-out base-task metrics for the selected configurations. |
| `metadataset/archived/dataset_metafeatures.csv` | Dataset descriptions computed for each seed, dataset, and sampling fraction. |
| `metadataset/archived/meta_dataset.csv` | Final join used by the paper: identifiers, 12 dataset meta-features, seven algorithm descriptors, and MCC. |
| `random_forest_shap/archived/joint_random_forest_shap.csv` | Mean absolute KernelSHAP values from the joint random-forest analysis. |
| [`hroch/archived/run.log`](hroch/archived/run.log) | HROCH training/test metrics and full symbolic expression. |
| [`random_forest_shap/archived/run.log`](random_forest_shap/archived/run.log) | Joint random-forest training/test metrics and selected parameters. |
| [`baselines/dataset_only/archived/run.log`](baselines/dataset_only/archived/run.log) | Training/test metrics and selected parameters for all five dataset-only reference regressors. |
| [`genetic_programming/hyperparameter_search/archived/summary.csv`](genetic_programming/hyperparameter_search/archived/summary.csv) | DEAP screening results for 300 configurations, including one without a safe hall-of-fame candidate. |
| [`genetic_programming/top_configuration_refinement/archived/summary.csv`](genetic_programming/top_configuration_refinement/archived/summary.csv) | DEAP refinement results for the top 10 configurations. |
| [`genetic_programming/configuration_01/archived/summary.csv`](genetic_programming/configuration_01/archived/summary.csv) | First DEAP long-run configuration: metrics, expression and complexity. |
| [`genetic_programming/configuration_02/archived/summary.csv`](genetic_programming/configuration_02/archived/summary.csv) | Second DEAP long-run configuration: metrics, expression and complexity. |
| [`genetic_programming/configuration_03/archived/summary.csv`](genetic_programming/configuration_03/archived/summary.csv) | Third DEAP long-run configuration: metrics, expression and complexity. |
| [`recalibration/extended_equation/archived/summary.csv`](recalibration/extended_equation/archived/summary.csv) | Pooled fixed-equation LODO results for original constants, training mean, DE and EGWO; raw and clipped predictions. |
| [`recalibration/hroch_equation/archived/summary.csv`](recalibration/hroch_equation/archived/summary.csv) | Corresponding LODO results for the HROCH equation. |

The [repository README](../../README.md#main-archived-results) summarizes these
archived metrics. GP summaries store sMAPE as a ratio; LODO summaries use
`smape_pct` in percentage units. LODO summaries pool held-out predictions over
all 65 datasets and also retain fold-level aggregates; the detailed
`fold_metrics.csv`, `oof_predictions.csv`, `coefficients.csv`,
`convergence.csv` and `run_config.json` files remain beside each summary.

Other output families:

| Directory | Contents |
|---|---|
| `model_evaluation/` | Complete meta-model evaluation runs: row-wise, grouped LODO and diagnostics. |
| `hroch/` | Symbolic-regression training logs and equation. |
| `baselines/joint/`, `baselines/dataset_only/` | Reference meta-regressor logs for each feature set. |
| `random_forest_shap/` | Random-forest logs and SHAP relevance tables. |
| `genetic_programming/hyperparameter_search/` | Screening summary, configuration logs, candidate tables and checkpoints. |
| `genetic_programming/top_configuration_refinement/` | Refinement summary, configuration logs, candidate tables and checkpoints. |
| `genetic_programming/configuration_01/`, `configuration_02/`, `configuration_03/` | Independent long runs of the three selected configurations. |
| `recalibration/extended_equation/`, `recalibration/hroch_equation/` | LODO configuration, fold checkpoints, metrics, predictions, coefficients and convergence. |
| `figures/` | PDF/PNG figures and prediction tables grouped by analysis. |
| `figures/symbolic_comparison/<run_id>/` | Two-panel full-precision HROCH/DEAP comparison: `Fig5_symbolic_comparison.pdf`, its PNG copy and `symbolic_comparison_metrics.csv`; generated by `src.analysis.plot_symbolic_comparison`. |
| `preprocessing_audit/` | Audit reports grouped by execution. |

The preparation pipeline writes into `metadataset/current/`, or a separate
build selected with `META2PERF_DATA_RUN`. New preparation tables have
`.complete.json` companions with checksums, input lineage, raw-data context,
code and versions. `tuning_status.json` records task outcomes and infeasible
sampling fractions. Partial outputs remain on disk but are not accepted as
completed inputs. Input selection uses one whole build or the archive, never
an individual historical fallback for a missing build table.

New GP runs retain `run_context.json`; their summaries, checkpoints and cache
metadata identify the metadataset checksum, feature order, engine and versions.
Reusable GP population caches belong in `src/cache/gp_initial_populations/`,
outside the result families. Reuse requires matching compatibility metadata.

These files are archived outputs, not synthetic examples. The original raw
datasets are not included in the publication repository. See
[reproducibility notes](../docs/reproducibility.md) for row counts, filtering,
split details, and validity limitations.
