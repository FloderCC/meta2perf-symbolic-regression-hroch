"""Train tuned base classifiers and evaluate them on held-out task data."""

from __future__ import annotations

import gc
import os
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.experiment_config import dataset_list, test_size
from src.paths import DATASETS_DIR, RESULTS_DIR, ensure_output_directories
from src.utils.dataset_utils import (
    get_dataset_sample,
    limit_dataset_size,
    preprocess_dataset,
)
from src.utils.model_utils import reset_seed, train_best_model


TUNING_RESULTS = RESULTS_DIR / "base_classifier_hyperparameters.csv"
EVALUATION_RESULTS = RESULTS_DIR / "base_classifier_performance.csv"


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
    print(f"Number of fitted configurations: {len(results)}")

    for run_number, (index, row) in enumerate(results.copy().iterrows(), start=1):
        seed = int(row["Seed"])
        dataset_name = row["Dataset"]
        sample_fraction = float(row["Sample Size"])
        model_name = row["Model"]
        model_parameters = row["Model Parameters"]
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

        # This reproduces the original protocol: preprocessing is applied to
        # the complete subsample before the held-out split.
        sampled_dataset = preprocess_dataset(sampled_dataset)
        X = sampled_dataset.drop(target_column, axis=1)
        y = sampled_dataset[target_column]
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        reset_seed(seed)
        print(
            f"[{run_number}/{len(results)}] {dataset_name}, "
            f"fraction={sample_fraction}, model={model_name}"
        )
        model = train_best_model(
            model_name,
            seed,
            x_train,
            y_train,
            model_parameters,
        )
        predictions = model.predict(x_test)

        results.loc[index, "Accuracy"] = accuracy_score(y_test, predictions)
        results.loc[index, "Precision"] = precision_score(
            y_test, predictions, average="weighted"
        )
        results.loc[index, "Recall"] = recall_score(
            y_test, predictions, average="weighted"
        )
        results.loc[index, "F1"] = f1_score(y_test, predictions, average="weighted")
        results.loc[index, "MCC"] = matthews_corrcoef(y_test, predictions)
        results.to_csv(EVALUATION_RESULTS, index=False)

        del model, predictions, x_train, x_test, y_train, y_test
        del X, y, sampled_dataset, full_dataset
        gc.collect()


if __name__ == "__main__":
    main()
