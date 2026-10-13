"""Tune every configured base task and publish only a complete tuning table."""

import os
import warnings
from multiprocessing import Pool

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.model_selection import train_test_split

from src.experiment_config import (
    dataset_list, dataset_percentage_list, model_list, seed_list, test_size,
)
from src.paths import DATASETS_DIR, METADATASET_OUTPUT_DIR, ensure_output_directories
from src.utils.artifact_integrity import atomic_json
from src.utils.base_classifiers import create_models, reset_seed
from src.utils.dataset_processing import get_dataset_sample, preprocess_dataset, limit_dataset_size
from src.utils.pipeline_integrity import (
    dataset_csv_paths, mark_complete, preparation_context, validate_frame,
)

results_header = ["Seed", "Dataset", "Sample Size", "Model", "Model Parameters"]


def run_exp(setup: list) -> dict:
    setup_name, seed, dataset_setup, csv_path = setup
    dataset_name, excluded_columns, class_name = dataset_setup
    outcome = {
        "setup": setup_name, "seed": seed, "dataset": dataset_name,
        "status": "failed", "valid_fractions": [], "skipped_fractions": [],
        "rows": 0, "partial_path": None,
    }
    results = []
    try:
        os.environ["PYTHONWARNINGS"] = "ignore"
        for category in (UndefinedMetricWarning, ConvergenceWarning, FutureWarning, UserWarning):
            warnings.filterwarnings("ignore", category=category)

        full_df = pd.read_csv(csv_path)
        if excluded_columns:
            full_df.drop(columns=excluded_columns, inplace=True)
        full_df = limit_dataset_size(full_df, class_name, seed)

        for dataset_percentage in dataset_percentage_list:
            df = get_dataset_sample(full_df, seed, dataset_percentage, class_name, test_size)
            if df is None:
                outcome["skipped_fractions"].append({
                    "fraction": dataset_percentage, "reason": "infeasible stratified split",
                })
                continue

            df = preprocess_dataset(df)
            X = df.drop(class_name, axis=1)
            y = df[class_name]
            x_train, _, y_train, _ = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            reset_seed(seed)
            tuned_models = create_models(seed, x_train, y_train)
            observed_models = [entry[0] for entry in tuned_models]
            if len(observed_models) != len(model_list) or set(observed_models) != set(model_list):
                raise ValueError(f"Incomplete classifier coverage: {observed_models}")

            for model_name, model_parameters_desc in tuned_models:
                results.append([seed, dataset_name, dataset_percentage, model_name, model_parameters_desc])
                partial_path = METADATASET_OUTPUT_DIR / f"base_classifier_hyperparameters_{setup_name}.csv"
                pd.DataFrame(results, columns=results_header).to_csv(partial_path, index=False)
                outcome["partial_path"] = str(partial_path)
                outcome["rows"] = len(results)
            outcome["valid_fractions"].append(dataset_percentage)

        outcome["status"] = "completed"
    except Exception as error:
        outcome["error"] = f"{type(error).__name__}: {error}"
        print(f"Error in setup {setup_name}: {error}", flush=True)
    return outcome


def main() -> None:
    if METADATASET_OUTPUT_DIR.exists() and any(METADATASET_OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Build directory is not empty: {METADATASET_OUTPUT_DIR}. "
            "Select a fresh META2PERF_DATA_RUN; existing outputs are preserved."
        )
    raw_inputs = dataset_csv_paths(dataset_list, DATASETS_DIR)
    context = preparation_context(raw_inputs)
    ensure_output_directories(METADATASET_OUTPUT_DIR)
    setups = [
        [f"seed_{seed}_dataset_{number}", seed, configuration, raw_inputs[configuration[0]]]
        for seed in seed_list
        for number, configuration in enumerate(dataset_list, start=1)
    ]
    with Pool(processes=os.cpu_count() or 4) as pool:
        outcomes = pool.map(run_exp, setups)

    failures = [outcome for outcome in outcomes if outcome["status"] != "completed"]
    atomic_json(METADATASET_OUTPUT_DIR / "tuning_status.json", {
        "complete": False, "context": context, "tasks": outcomes,
    })
    if failures:
        raise RuntimeError(
            f"{len(failures)} tuning tasks failed. Partial files and tuning_status.json "
            "are preserved; no consolidated table has been published."
        )
    # Only paths returned by this invocation are consolidated; no directory glob
    # can pull old partial results into a new build.
    partial_paths = [outcome["partial_path"] for outcome in outcomes if outcome["partial_path"]]
    if not partial_paths:
        raise RuntimeError("No feasible tuning tasks produced results; inspect tuning_status.json.")
    all_results = pd.concat([pd.read_csv(path) for path in partial_paths], ignore_index=True)
    expected_rows = sum(len(outcome["valid_fractions"]) * len(model_list) for outcome in outcomes)
    if len(all_results) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} tuning rows, found {len(all_results)}.")
    validate_frame(all_results, "base_classifier_hyperparameters.csv")
    if preparation_context(raw_inputs) != context:
        raise RuntimeError("Raw datasets changed during tuning. Partial results are preserved.")
    target = METADATASET_OUTPUT_DIR / "base_classifier_hyperparameters.csv"
    all_results.to_csv(target, index=False)
    mark_complete(target, all_results, inputs=(), context=context)
    atomic_json(METADATASET_OUTPUT_DIR / "tuning_status.json", {
        "complete": True, "context": context, "tasks": outcomes,
    })


if __name__ == "__main__":
    main()
