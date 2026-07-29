"""Evaluate and plot the archived symbolic-regression expression."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.meta_analysis import adjusted_r2, smape, stratified_regression_split
from src.paths import (
    PLOTS_PDF_DIR,
    PLOTS_PNG_DIR,
    RESULTS_DIR,
    ensure_output_directories,
)


TARGET = "MCC"
IDENTIFIER_COLUMNS = ["Seed", "Dataset", "Sample Size", "Model"]


def symbolic_predict(features: pd.DataFrame) -> np.ndarray:
    """Evaluate the final expression using the feature order in meta_dataset.csv."""

    x = features.to_numpy()
    a = 3.3237551265443805
    b = 1.4435745948155903
    c = 3.8006530371420686
    d = 12.379278274862575
    e = 2.1575551710877265
    f = -0.011386770840701929

    t1 = (x[:, 13] - x[:, 15]) - a
    t2 = x[:, 11] + b
    t3 = x[:, 11] + x[:, 9] + x[:, 1]
    t4 = ((x[:, 14] + t1) / t1) * x[:, 15] * t2
    t5 = (x[:, 10] * t4) / t3
    t6 = (
        (c + (x[:, 4] - d / x[:, 6]) * (x[:, 13] - x[:, 15]))
        * (x[:, 0] / x[:, 2])
    ) ** 2 + x[:, 4] + t2

    numerator = (x[:, 17] + 1) * t5 - t6 - (x[:, 5] + x[:, 6]) / e + x[:, 1]
    denominator = t3 * f * (t4 + x[:, 9] + x[:, 1]) - t6
    return numerator / denominator


def main() -> None:
    ensure_output_directories()
    meta_dataset = pd.read_csv(RESULTS_DIR / "meta_dataset.csv")
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
        PLOTS_PDF_DIR / "symbolic_regression_test.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        PLOTS_PNG_DIR / "symbolic_regression_test.png",
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
