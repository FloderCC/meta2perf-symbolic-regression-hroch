"""Evaluate the five reference meta-regressors used in the paper."""

from __future__ import annotations

import argparse
import logging
import random
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils.meta_regression import adjusted_r2, smape, stratified_regression_split
from src.utils.pipeline_integrity import read_meta_dataset
from src.paths import META_DATASET_PATH, ensure_output_directories, output_run_directory


RANDOM_STATE = 42
TARGET = "MCC"
IDENTIFIER_COLUMNS = ["Seed", "Dataset", "Sample Size", "Model"]
ALGORITHM_DESCRIPTOR_COLUMNS = [
    "Processing Units Number",
    "Model Type",
    "Training Complexity",
    "Prediction Complexity",
    "Regularization",
    "Robust to Outliers",
    "Representational Capacity",
]

PARAMETER_GRIDS = {
    "LinearRegression": {},
    "DecisionTreeRegressor": {
        "max_depth": [8, 16, 64, 128],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100],
        "max_depth": [8, 16, 64, 128],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", 0.5],
        "bootstrap": [True, False],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100, 300],
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 5],
        "subsample": [1.0, 0.8],
    },
    "MLPRegressor": {
        "hidden_layer_sizes": [(50,), (100,), (200,)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set",
        choices=("joint", "dataset-only"),
        default="joint",
        help=(
            "Use dataset and algorithm descriptors (joint, the primary analysis) "
            "or only dataset descriptors (the marginal baseline)."
        ),
    )
    return parser.parse_args()


def _configure_logging(feature_set: str) -> None:
    run_directory = output_run_directory("baselines", feature_set.replace("-", "_"))
    ensure_output_directories(run_directory)
    log_path = run_directory / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
        force=True,
    )


def _models() -> list:
    return [
        LinearRegression(),
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        MLPRegressor(random_state=RANDOM_STATE, max_iter=1_000),
    ]


def _log_metrics(label: str, y_true, y_pred, predictor_count: int) -> None:
    r2_value = r2_score(y_true, y_pred)
    logging.info(
        "%s (%d rows): R^2: %.3f, Adjusted R^2: %.3f, "
        "sMAPE: %.3f, MAE: %.3f",
        label,
        len(y_true),
        r2_value,
        adjusted_r2(r2_value, len(y_true), predictor_count),
        smape(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.feature_set)
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    meta_dataset = read_meta_dataset(META_DATASET_PATH)
    meta_dataset = meta_dataset.drop(columns=IDENTIFIER_COLUMNS)
    meta_dataset = meta_dataset[meta_dataset[TARGET] > 0]
    if args.feature_set == "dataset-only":
        meta_dataset = meta_dataset.drop(columns=ALGORITHM_DESCRIPTOR_COLUMNS)

    X = meta_dataset.drop(columns=TARGET)
    y = meta_dataset[TARGET]
    X_train, X_test, y_train, y_test = stratified_regression_split(X, y)

    logging.info("Feature set: %s (%d predictors)", args.feature_set, X.shape[1])
    if args.feature_set == "dataset-only":
        logging.info(
            "Interpretation: this is a marginal baseline. Identical dataset "
            "descriptors can map to outcomes from multiple classifiers."
        )

    for model in _models():
        model_name = model.__class__.__name__
        parameter_grid = PARAMETER_GRIDS[model_name]
        if parameter_grid:
            search = GridSearchCV(
                model,
                parameter_grid,
                scoring="neg_mean_squared_error",
                cv=4,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            fitted_model = search.best_estimator_
            selected_parameters = search.best_params_
        else:
            fitted_model = model.fit(X_train, y_train)
            selected_parameters = "Default parameters (no tuning)"

        logging.info("Results for %s:", model_name)
        _log_metrics(
            "Train dataset",
            y_train,
            fitted_model.predict(X_train),
            X.shape[1],
        )
        _log_metrics(
            "Test dataset",
            y_test,
            fitted_model.predict(X_test),
            X.shape[1],
        )
        logging.info("Selected hyperparameters: %s", selected_parameters)


if __name__ == "__main__":
    main()
