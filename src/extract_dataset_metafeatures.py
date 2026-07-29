"""Extract dataset meta-features for every seed and sampling fraction."""

from __future__ import annotations

import gc
import os
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from src.experiment_config import dataset_list, test_size
from src.paths import DATASETS_DIR, RESULTS_DIR, ensure_output_directories
from src.utils.dataset_utils import (
    describe_dataset_using_pymfe,
    get_dataset_sample,
    limit_dataset_size,
    preprocess_dataset,
)


TUNING_RESULTS = RESULTS_DIR / "base_classifier_hyperparameters.csv"
METAFEATURE_RESULTS = RESULTS_DIR / "dataset_metafeatures.csv"


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
    ensure_output_directories()

    results = pd.read_csv(TUNING_RESULTS)
    results = results[["Seed", "Dataset", "Sample Size"]].drop_duplicates()
    print(f"Number of dataset descriptions: {len(results)}")

    for run_number, (index, row) in enumerate(results.copy().iterrows(), start=1):
        seed = int(row["Seed"])
        dataset_name = row["Dataset"]
        sample_fraction = float(row["Sample Size"])
        excluded_columns, target_column = _dataset_configuration(dataset_name)

        dataset_dir = DATASETS_DIR / dataset_name
        csv_files = sorted(dataset_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found in {dataset_dir}. See src/docs/datasets.md."
            )

        full_dataset = pd.read_csv(csv_files[0])
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
            seed,
        )
        for name, value in features.items():
            results.loc[index, name] = value

        results.to_csv(METAFEATURE_RESULTS, index=False)
        del sampled_dataset, full_dataset
        gc.collect()


if __name__ == "__main__":
    main()
