"""Final HROCH equation for the corrected meta-dataset, with six coefficients."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "class_ent", "eq_num_attr", "gravity", "inst_to_attr", "nr_attr",
    "nr_bin", "nr_class", "nr_cor_attr", "nr_inst", "nr_norm",
    "nr_outliers", "ns_ratio", "Processing Units Number", "Model Type",
    "Training Complexity", "Prediction Complexity", "Regularization",
    "Robust to Outliers", "Representational Capacity",
)
COEFFICIENT_NAMES = ("a", "b", "c", "d", "e", "f")
DEFAULT_COEFFICIENTS = np.array([
    4.865837522725318,
    1.1256530281989316,
    3.5256330733899675,
    12.507643915250597,
    2.132726876109822,
    -0.009773204206524137,
], dtype=float)
DEFAULT_COEFFICIENTS.setflags(write=False)


def predict(features, coefficients=None) -> np.ndarray:
    """Evaluate the supplied arithmetic in the explicit x0..x18 feature order.

    Defaults are the supplied final coefficients. An explicit coefficient vector
    is accepted for LODO fitting. Nonfinite outputs remain visible to callers.
    """
    if isinstance(features, pd.DataFrame):
        x = features.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    else:
        x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected an (n, {len(FEATURE_NAMES)}) feature matrix.")
    coefficients = np.asarray(
        DEFAULT_COEFFICIENTS if coefficients is None else coefficients, dtype=float
    )
    if coefficients.shape != DEFAULT_COEFFICIENTS.shape or not np.isfinite(coefficients).all():
        raise ValueError("Expected six finite coefficients in a, b, c, d, e, f order.")
    a, b, c, d, e, f = coefficients
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
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


symbolic_predict = predict
