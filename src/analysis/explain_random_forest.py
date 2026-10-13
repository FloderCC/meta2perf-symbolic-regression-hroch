"""Fit and explain the joint random-forest meta-regressor with KernelSHAP."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from shap import KernelExplainer, kmeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from src.utils.meta_regression import adjusted_r2, smape, stratified_regression_split
from src.utils.pipeline_integrity import read_meta_dataset
from src.paths import META_DATASET_PATH, ensure_output_directories, output_run_directory


RANDOM_STATE = 42
RUN_DIR = output_run_directory("random_forest_shap")
TARGET = "MCC"
IDENTIFIER_COLUMNS = ["Seed", "Dataset", "Sample Size", "Model"]
PARAMETER_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [8, 16, 64, 128],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2", 0.5],
    "bootstrap": [True, False],
}


def _configure_logging() -> None:
    ensure_output_directories(RUN_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(RUN_DIR / "run.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def main() -> None:
    _configure_logging()
    warnings.filterwarnings("ignore")

    meta_dataset = read_meta_dataset(META_DATASET_PATH)
    meta_dataset = meta_dataset.drop(columns=IDENTIFIER_COLUMNS)
    meta_dataset = meta_dataset[meta_dataset[TARGET] > 0]
    X = meta_dataset.drop(columns=TARGET)
    y = meta_dataset[TARGET]
    X_train, X_test, y_train, y_test = stratified_regression_split(X, y)

    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        PARAMETER_GRID,
        scoring="neg_mean_squared_error",
        cv=4,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    for label, features, targets in (
        ("Train dataset", X_train, y_train),
        ("Test dataset", X_test, y_test),
    ):
        predictions = model.predict(features)
        r2_value = r2_score(targets, predictions)
        logging.info(
            "%s (%d rows): R^2: %.3f, Adjusted R^2: %.3f, "
            "sMAPE: %.3f, MAE: %.3f",
            label,
            len(targets),
            r2_value,
            adjusted_r2(r2_value, len(targets), X.shape[1]),
            smape(targets, predictions),
            mean_absolute_error(targets, predictions),
        )
    logging.info("Selected hyperparameters: %s", search.best_params_)

    logging.getLogger("shap").setLevel(logging.ERROR)
    background = kmeans(X_train.to_numpy(), 10)
    explainer = KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test.to_numpy(), nsamples=100)
    mean_absolute_shap = np.mean(np.abs(shap_values), axis=0)

    relevance = pd.DataFrame(
        {"feature": X.columns, "mean_absolute_shap": mean_absolute_shap}
    )
    relevance.to_csv(RUN_DIR / "joint_random_forest_shap.csv", index=False)
    logging.info("KernelSHAP explanation completed.")


if __name__ == "__main__":
    main()
