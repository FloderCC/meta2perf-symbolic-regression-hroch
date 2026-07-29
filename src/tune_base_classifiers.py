"""Tune the 16 base classifiers for every dataset, fraction, and seed."""

import os
import warnings
from multiprocessing import Pool

from src.experiment_config import (
    dataset_list,
    dataset_percentage_list,
    model_list,
    seed_list,
    test_size,
)
from src.paths import DATASETS_DIR, RESULTS_DIR, ensure_output_directories
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.model_selection import train_test_split

from src.utils.dataset_utils import get_dataset_sample, preprocess_dataset, limit_dataset_size
from src.utils.model_utils import create_models, reset_seed

results_header = ["Seed", "Dataset", "Sample Size", "Model", "Model Parameters"]

def run_exp(setup: list):
    try:
        # Ignore warnings
        os.environ['PYTHONWARNINGS'] = 'ignore'
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        results = []

        setup_name = setup[0]
        seed = setup[1]
        dataset_setup = setup[2]

        dataset_name = dataset_setup[0]
        excluded_columns = dataset_setup[1]
        class_name = dataset_setup[2]

        # Loading the dataset
        dataset_folder = DATASETS_DIR / dataset_name
        csv_files = sorted(dataset_folder.glob("*.csv"))
        if not csv_files:
            return
        full_df = pd.read_csv(csv_files[0])

        if excluded_columns:
            full_df.drop(columns=excluded_columns, inplace=True)

        # Limiting dataset size
        full_df = limit_dataset_size(full_df, class_name, seed)

        for dataset_percentage in dataset_percentage_list:
            df = get_dataset_sample(full_df, seed, dataset_percentage, class_name, test_size)

            if df is None:
                continue

            df = preprocess_dataset(df)
            X = df.drop(class_name, axis=1)
            y = df[class_name]

            x_train, _, y_train, _ = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )

            reset_seed(seed)
            tuned_models = create_models(seed, x_train, y_train)

            for (model_name, model_parameters_desc) in tuned_models:
                results.append([seed, dataset_name, dataset_percentage, model_name, model_parameters_desc])
                results_df = pd.DataFrame(results, columns=results_header)
                partial_path = RESULTS_DIR / f"base_classifier_hyperparameters_{setup_name}.csv"
                results_df.to_csv(partial_path, index=False)

    except Exception as e:
        print(f"Error in setup {setup}: {e}")

def main():
    ensure_output_directories()
    setups_to_parallelize = []
    for seed in seed_list:
        dataset_no = 0
        for dataset_setup in dataset_list:
            dataset_no += 1
            setups_to_parallelize.append([f"seed_{seed}_dataset_{dataset_no}", seed, dataset_setup])

    # Determine the number of processes. It's often beneficial to set this to the number of CPU cores.
    max_workers = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None

    with Pool(processes=max_workers) as pool:
        pool.map(run_exp, setups_to_parallelize)

    partial_files = sorted(RESULTS_DIR.glob("base_classifier_hyperparameters_seed_*.csv"))
    if not partial_files:
        raise RuntimeError(
            "No tuning outputs were produced. Add each source CSV under "
            "src/datasets/<dataset name>/ and verify the dataset configuration."
        )
    all_results = pd.concat([pd.read_csv(path) for path in partial_files], ignore_index=True)
    all_results.to_csv(RESULTS_DIR / "base_classifier_hyperparameters.csv", index=False)

    for path in partial_files:
        path.unlink()

if __name__ == "__main__":
    main()
