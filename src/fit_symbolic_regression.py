"""Fit the unscaled symbolic-regression model reported in the paper."""

from __future__ import annotations

import logging
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from HROCH import SymbolicRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.meta_analysis import adjusted_r2, smape, stratified_regression_split
from src.paths import LOGS_DIR, RESULTS_DIR, ensure_output_directories


RANDOM_STATE = 42
TARGET = "MCC"
IDENTIFIER_COLUMNS = ["Seed", "Dataset", "Sample Size", "Model"]


class GreenStreamHandler(logging.StreamHandler):
    """Write console log messages in green when ANSI colors are supported."""

    def emit(self, record):
        try:
            message = self.format(record)
            self.stream.write(f"\033[92m{message}\033[0m{self.terminator}")
            self.flush()
        except Exception:
            self.handleError(record)


def _configure_logging() -> None:
    ensure_output_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "symbolic_regression.txt", mode="w"),
            GreenStreamHandler(),
        ],
        force=True,
    )


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
    _configure_logging()
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    meta_dataset = pd.read_csv(RESULTS_DIR / "meta_dataset.csv")
    meta_dataset = meta_dataset.drop(columns=IDENTIFIER_COLUMNS)
    # This analytical choice matches the revised manuscript. Zero and negative
    # MCC values remain available in the archived, unfiltered meta-dataset.
    meta_dataset = meta_dataset[meta_dataset[TARGET] > 0]

    X = meta_dataset.drop(columns=TARGET)
    y = meta_dataset[TARGET]
    X_train, X_test, y_train, y_test = stratified_regression_split(X, y)

    model = SymbolicRegressor(
        num_threads=os.cpu_count(),
        random_state=RANDOM_STATE,
        verbose=1,
        metric="MSE",
        time_limit=259_200,
        iter_limit=0,
        problem="math",
        precision="f64",
        population_settings={"size": 1_000, "tournament": 500},
        init_const_settings={"const_min": -1.0, "const_max": 5.0},
        algo_settings={
            "neighbours_count": 30,
            "alpha": 0.01,
            "beta": 0.9,
            # HROCH represents these values in 64-row batches:
            # pretest_size=2 -> 128 rows; sample_size=200 -> 12,800 rows.
            "pretest_size": 2,
            "sample_size": 200,
        },
    )

    start = time.perf_counter()
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    elapsed = time.perf_counter() - start

    train_predictions = model.predict(X_train.to_numpy())
    test_predictions = model.predict(X_test.to_numpy())
    predictor_count = X_train.shape[1]

    logging.info("Final results:")
    _log_metrics("Train dataset", y_train, train_predictions, predictor_count)
    _log_metrics("Test dataset", y_test, test_predictions, predictor_count)
    logging.info("Elapsed fit time: %.1f seconds", elapsed)
    logging.info("Equation:")
    logging.info("%s", model.equation)


if __name__ == "__main__":
    main()
