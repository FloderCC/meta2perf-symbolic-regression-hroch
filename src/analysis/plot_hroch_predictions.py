"""Evaluate and plot the final HROCH expression on corrected descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.equations.hroch import predict as symbolic_predict
from src.utils.meta_regression import adjusted_r2, smape, stratified_regression_split
from src.utils.pipeline_integrity import read_meta_dataset
from src.paths import (
    META_DATASET_PATH,
    ensure_output_directories,
    output_run_directory,
)


TARGET = "MCC"
IDENTIFIER_COLUMNS = ["Seed", "Dataset", "Sample Size", "Model"]




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=META_DATASET_PATH)
    args = parser.parse_args()
    figures_dir = output_run_directory("figures", "hroch_predictions")
    ensure_output_directories(figures_dir)
    meta_dataset = read_meta_dataset(args.data)
    meta_dataset = meta_dataset.drop(columns=IDENTIFIER_COLUMNS)
    meta_dataset = meta_dataset[meta_dataset[TARGET] > 0]
    X = meta_dataset.drop(columns=TARGET)
    y = meta_dataset[TARGET]
    X_train, X_test, y_train, y_test = stratified_regression_split(X, y)

    for label, features, targets in (
        ("Train", X_train, y_train),
        ("Test", X_test, y_test),
    ):
        predictions = symbolic_predict(features)
        r2_value = r2_score(targets, predictions)
        print(
            f"{label} dataset ({len(targets)} rows): "
            f"R^2={r2_value:.3f}, "
            f"adjusted R^2={adjusted_r2(r2_value, len(targets), X.shape[1]):.3f}, "
            f"sMAPE={smape(targets, predictions):.3f}, "
            f"MAE={mean_absolute_error(targets, predictions):.3f}"
        )

    test_predictions = symbolic_predict(X_test)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        y_test,
        test_predictions,
        alpha=0.5,
        edgecolors="#182e3d",
        label="Predictions",
    )
    lower = min(float(y_test.min()), float(test_predictions.min()))
    upper = max(float(y_test.max()), float(test_predictions.max()))
    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="#c42525",
        label="Ideal fit (y = x)",
    )
    axis.set_xlabel("True MCC")
    axis.set_ylabel("Predicted MCC")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figures_dir / "symbolic_regression_test.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        figures_dir / "symbolic_regression_test.png",
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
