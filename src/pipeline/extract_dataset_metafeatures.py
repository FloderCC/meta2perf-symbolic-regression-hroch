"""Extract dataset meta-features for every seed and sampling fraction."""

from __future__ import annotations

import gc
import os
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from src.experiment_config import dataset_list, test_size
from src.paths import (
    DATASETS_DIR, METADATASET_OUTPUT_DIR, ensure_output_directories,
    metadataset_input_path,
)
from src.utils.dataset_processing import (
    describe_dataset_using_pymfe,
    get_dataset_sample,
    limit_dataset_size,
    preprocess_dataset,
)
from src.utils.pipeline_integrity import (
    TASK_KEYS, dataset_csv_paths, ensure_output_available, mark_complete,
    preparation_context, read_pipeline_table, require_preparation_context,
    validate_frame, validate_key_coverage,
)


TUNING_RESULTS = metadataset_input_path("base_classifier_hyperparameters.csv")
METAFEATURE_RESULTS = METADATASET_OUTPUT_DIR / "dataset_metafeatures.csv"


def _configure_warnings() -> None:
    os.environ["PYTHONWARNINGS"] = "ignore"
    for category in (
        UndefinedMetricWarning,
        ConvergenceWarning,
        FutureWarning,
        UserWarning,
    ):
        warnings.filterwarnings("ignore", category=category)


def _dataset_configuration(dataset_name: str) -> tuple[list[str], str]:
    for configured_name, excluded_columns, target_column in dataset_list:
        if configured_name == dataset_name:
            return excluded_columns, target_column
    raise KeyError(f"Dataset {dataset_name!r} is not present in experiment_config.py")


def main() -> None:
    _configure_warnings()
    ensure_output_available(METAFEATURE_RESULTS)
    raw_inputs = dataset_csv_paths(dataset_list, DATASETS_DIR)
    context = preparation_context(raw_inputs)
    ensure_output_directories(METADATASET_OUTPUT_DIR)

    results = read_pipeline_table(TUNING_RESULTS)
    require_preparation_context(TUNING_RESULTS, context)
    results = results[TASK_KEYS].drop_duplicates()
    expected_keys = results[TASK_KEYS].copy()
    print(f"Number of dataset descriptions: {len(results)}")

    for run_number, (index, row) in enumerate(results.copy().iterrows(), start=1):
        seed = int(row["Seed"])
        dataset_name = row["Dataset"]
        sample_fraction = float(row["Sample Size"])
        excluded_columns, target_column = _dataset_configuration(dataset_name)

        full_dataset = pd.read_csv(raw_inputs[dataset_name])
        if excluded_columns:
            full_dataset.drop(columns=excluded_columns, inplace=True)
        full_dataset = limit_dataset_size(full_dataset, target_column, seed)
        sampled_dataset = get_dataset_sample(
            full_dataset,
            seed,
            sample_fraction,
            target_column,
            test_size,
        )
        if sampled_dataset is None:
            raise ValueError(
                f"The stratified split is infeasible for {dataset_name}, "
                f"fraction={sample_fraction}, seed={seed}."
            )

        print(
            f"[{run_number}/{len(results)}] {dataset_name}, "
            f"fraction={sample_fraction}, seed={seed}"
        )
        sampled_dataset = preprocess_dataset(sampled_dataset)
        features = describe_dataset_using_pymfe(
            sampled_dataset,
            target_column,
        )
        for name, value in features.items():
            results.loc[index, name] = value

        results.to_csv(METAFEATURE_RESULTS, index=False)
        del sampled_dataset, full_dataset
        gc.collect()

    validate_frame(results, METAFEATURE_RESULTS.name)
    validate_key_coverage(results, expected_keys, TASK_KEYS, "Dataset descriptors")
    if preparation_context(raw_inputs) != context:
        raise RuntimeError("Raw datasets changed during descriptor extraction; output remains incomplete.")
    mark_complete(METAFEATURE_RESULTS, results, inputs=(TUNING_RESULTS,), context=context)


if __name__ == "__main__":
    main()
