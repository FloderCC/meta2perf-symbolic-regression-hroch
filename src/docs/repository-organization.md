# Repository organization

All source files, datasets, caches and generated artifacts remain inside `src`.
The reorganization moves or renames existing files and updates their references;
it does not discard scripts, datasets, reports, logs, figures or checkpoints.

## Code directories

| Directory | Responsibility |
|---|---|
| `pipeline/` | Base-classifier tuning, held-out evaluation, dataset meta-features and meta-dataset construction. |
| `experiments/hroch/` | HROCH symbolic-regression training. |
| `experiments/baselines/` | Reference meta-regressor comparison. |
| `experiments/genetic_programming/` | GP hyperparameter screening, refinement and three configured long runs. |
| `experiments/recalibration/` | Fixed-equation leave-one-dataset-out coefficient recalibration. |
| `analysis/` | Random-forest explanations and prediction, correlation and SHAP figures. |
| `equations/` | Extended equation with 19 constants and HROCH equation with six constants. |
| `utils/` | Shared data processing, base-classifier construction, regression metrics and GP operations. |
| `datasets/` | Existing source CSV files and per-dataset documentation; their locations are unchanged. |
| `docs/` | Scientific documentation, this guide and the move inventory. |

`paths.py` centralizes all locations. `experiment_config.py` retains the dataset,
classifier, seed, fraction and split settings. Package `__init__.py` files allow
entry points to be invoked with `python -m` from the repository root.

## Script mapping

Paths in this table are relative to `src`.

| Original path | Current path |
|---|---|
| `tune_base_classifiers.py` | `pipeline/tune_base_classifiers.py` |
| `evaluate_base_classifiers.py` | `pipeline/evaluate_base_classifiers.py` |
| `extract_dataset_metafeatures.py` | `pipeline/extract_dataset_metafeatures.py` |
| `build_meta_dataset.py` | `pipeline/build_meta_dataset.py` |
| `fit_symbolic_regression.py` | `experiments/hroch/train.py` |
| `benchmark_meta_regressors.py` | `experiments/baselines/benchmark.py` |
| `exp_perf_beam_grid_search.py` | `experiments/genetic_programming/search_hyperparameters.py` |
| `exp_perf_beam_grid_search_top10.py` | `experiments/genetic_programming/refine_top_configurations.py` |
| `exp_perf_beam_grid_search_top10_s1.py` | `experiments/genetic_programming/train_configuration_01.py` |
| `exp_perf_beam_grid_search_top10_s2.py` | `experiments/genetic_programming/train_configuration_02.py` |
| `exp_perf_beam_grid_search_top10_s3.py` | `experiments/genetic_programming/train_configuration_03.py` |
| `exp_perf_lodo_recalibration.py` | `experiments/recalibration/recalibrate_extended_equation_lodo.py` |
| `exp_perf_lodo_recalibration_eq2.py` | `experiments/recalibration/recalibrate_hroch_equation_lodo.py` |
| `explain_random_forest.py` | `analysis/explain_random_forest.py` |
| `plot_feature_correlations.py` | `analysis/plot_feature_correlations.py` |
| `plot_shap_relevance.py` | `analysis/plot_shap_relevance.py` |
| `plot_symbolic_regression.py` | `analysis/plot_hroch_predictions.py` |
| `plot_reg_model.py` | `analysis/plot_extended_equation_predictions.py` |
| `meta_analysis.py` | `utils/meta_regression.py` |
| `utils/dataset_utils.py` | `utils/dataset_processing.py` |
| `utils/model_utils.py` | `utils/base_classifiers.py` |
| `utils/ga_initialization.py` | `utils/gp_initialization.py` |
| `utils/gp_custom_operators.py` | `utils/gp_operators.py` |
| `utils/fixed_equation.py` | `equations/extended.py` |
| `utils/fixed_equation_eq2.py` | `equations/hroch.py` |

`paths.py`, `experiment_config.py`, `src/__init__.py` and `utils/__init__.py`
retain their locations. All three GP configuration scripts remain separate;
their numbers identify configurations, and all three retain random seed 42.

`analysis/plot_symbolic_comparison.py` is the standalone generator of the paper's
two-panel HROCH/DEAP prediction figure. It reads the archived full-precision DEAP
configuration 02 expression and the fixed HROCH equation using repository
inputs. Its PDF, PNG and train/test metric table are written below
`outputs/figures/symbolic_comparison/<run_id>/`. Both X axes end at 1. The original
temporary manuscript-integration files are retained under the ignored `tmp/`
directory; the standalone generator does not depend on them.

## Output locations

Existing artifacts are stored under these paths, relative to `src/outputs`:

| Output | Archive location |
|---|---|
| Four preparation/meta-dataset tables | `metadataset/archived/` |
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

The old result index is now `src/outputs/README.md`; the old log index is
`src/outputs/archived_logs.md`. Empty legacy directories may remain on disk:
no directory cleanup or file deletion was performed.

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
the outputs they already produced; no new training or scoring logic is added.

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
  only with the archived metadataset; old checkpoints and caches are preserved
  and are not automatically reused.
- To resume a GP run, select its existing `META2PERF_RUN_ID` and enable its
  `RESUME_FROM_CHECKPOINT` setting. Its `run_context.json`, checkpoints, cache
  metadata and summaries must match the data, feature order, engine and package
  versions. Compatible screening checkpoints can continue in refinement.
  Generation, logging and resume controls are excluded from the engine hash.
  To resume LODO, pass its existing
  `--output-dir` together with `--resume`; its data/code/settings checks remain.

The structural validator remains at `tools/validate_repository.py`. Historical
preprocessing-audit reports remain under `outputs/preprocessing_audit/archived/`;
the current repository has no `audit_preprocessing.py` entry point. The README
lists the current module commands for later use.

## Preservation record

[reorganization_manifest.csv](reorganization_manifest.csv) records all 346
pre-existing files under `src`, with original and current paths, original size,
SHA-256 where recorded, and whether reference edits are permitted. Raw source
CSV files were inventoried by path and size and were not read for hashing.

Archived CSV, JSON, log and figure contents are preserved byte for byte.
Historical paths, code hashes and identifiers embedded inside those artifacts
remain unchanged as provenance; the inventory translates their old locations.
Source code and documentation are edited only to support the new organization
and output conventions. No Python code, imports, compilation, training, tests
or validators were executed for this change; review is static.
