"""Final grouped DEAP equation for the corrected meta-dataset.

The 19 supplied final constants are shared by evaluation and LODO fitting.
Repeated constants share a parameter; integer powers stay fixed. Features use
their original units, with no fitted scaling or imputation.
"""

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
COEFFICIENT_NAMES = (
    "C_4_44", "C_20_65", "C_0_29", "C_0_31", "C_16_53", "C_0_62",
    "C_3_92", "C_48_25", "C_49_93", "C_37_97", "C_1_75", "C_0_98",
    "C_31_51", "C_33_56", "C_12_03", "C_9_23", "C_39_37", "C_33_34",
    "C_77_11",
)
DEFAULT_COEFFICIENTS = np.array([
    4.436387158962334, 21.017598763216697, 0.26149710186442016,
    0.4370849413454668, 16.711964883971923, 0.708763384574498,
    3.6748734018085374, 45.52942272372836, 83.39007680021791,
    36.46003772076081, 1.652759610031753, 0.9071889832719622,
    34.92368710274265, 32.38844287648101, 11.662600535000225,
    9.243015416208092, 35.26855405622173, 31.600629130095175,
    76.31302514262624,
], dtype=float)
DEFAULT_COEFFICIENTS.setflags(write=False)


def predict(X, coefficients=None):
    """Evaluate the supplied expression, keeping nonfinite results visible.

    DataFrame columns are selected by name in the explicit x0..x18 order.
    Arrays must already have that order. Overflowing intermediate operations
    are allowed (e.g. a finite number divided by exp(large) gives zero), as in
    the supplied NumPy expression. Only final nonfinite outputs are invalid.
    """
    if isinstance(X, pd.DataFrame):
        X = X.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected an (n, {len(FEATURE_NAMES)}) feature matrix.")
    coefficients = np.asarray(
        DEFAULT_COEFFICIENTS if coefficients is None else coefficients, dtype=float
    )
    if coefficients.shape != DEFAULT_COEFFICIENTS.shape or not np.isfinite(coefficients).all():
        raise ValueError("Expected 19 finite coefficients in COEFFICIENT_NAMES order.")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return np.asarray(_evaluate(X, *coefficients), dtype=float)


def _evaluate(
    X,
    C_4_44=4.436387158962334,
    C_20_65=21.017598763216697,
    C_0_29=0.26149710186442016,
    C_0_31=0.4370849413454668,
    C_16_53=16.711964883971923,
    C_0_62=0.708763384574498,
    C_3_92=3.6748734018085374,
    C_48_25=45.52942272372836,
    C_49_93=83.39007680021791,
    C_37_97=36.46003772076081,
    C_1_75=1.652759610031753,
    C_0_98=0.9071889832719622,
    C_31_51=34.92368710274265,
    C_33_56=32.38844287648101,
    C_12_03=11.662600535000225,
    C_9_23=9.243015416208092,
    C_39_37=35.26855405622173,
    C_33_34=31.600629130095175,
    C_77_11=76.31302514262624
):
    (X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18) = X.T

    # -------------------------
    # T1
    # -------------------------
    T1 = (
        X7
        + (
            X6 / (
                X14
                + X4 * X18
                * (np.sqrt(np.exp(np.exp(X15))) - X6 + X13)
                * np.log(C_4_44)
            )
        )
        + np.sqrt(X11 / X3)
    )

    # -------------------------
    # T2
    # -------------------------
    T2 = (
        X11 / (
            X4
            + np.exp(X9)
            * (
                -C_20_65
                + X4 * np.exp(np.sqrt(X3))
                * (
                    np.exp(X5 / C_0_29) ** 3
                    / (X2 ** 2 - C_0_31)
                )
            )
            * (C_16_53 / (X3 * X8))
        )
        + X11 / (
            X18 * np.log(X3)
            + 2 * (X13 + X14)
        )
    )

    # -------------------------
    # T3
    # -------------------------
    T3 = (
        X11 / (
            X6
            + (
                X1 / X10
                + X18
                * (
                    X16
                    + X18 ** 3
                    * np.exp(np.sqrt(X3))
                    * (C_0_62 / np.exp(np.exp(X18)))
                )
                * (X2 / (np.sqrt(X1) ** 3))
                * (X18 ** 3 / X1)
            ) / X13
        )
        + X1 / (X12 + X10)
        + X15
        * (
            (
                np.sqrt(X2)
                / (
                    (
                        np.sqrt(X3)
                        / (
                            (
                                X0 ** 6
                                + (np.sqrt(np.log(X1) ** 9)) ** 3
                            )
                            / (
                                C_0_29
                                + (
                                    (X4 * np.exp(X18))
                                    / (X2 * X11)
                                ) ** 162
                            )
                            + X17
                        )
                    )
                    / (
                        X16 ** 2
                        + (X0 * X1) ** 3
                        * X17
                        * np.log(X2) ** 6
                        + X16
                    )
                )
            ) ** (1 / 8)
        )
    )

    # -------------------------
    # T4
    # -------------------------
    T4 = (
        np.sqrt(X9 / X11)
        + (
            X11 / (
                np.exp(
                    X5
                    / (
                        (X6 - C_3_92) * np.exp(X15)
                        + X1
                    )
                )
                + X1 / X8
            )
            + C_48_25 * X15
            + X10
        )
        / (
            X4
            + C_49_93 * (X3 / X2)
        )
        + C_3_92 / (np.exp(X6) - X5)
    )

    # -------------------------
    # T5
    # -------------------------
    T5 = (
        X1 / (
            np.exp(
                3 * X5 / (
                    X11 / (
                        np.exp(
                            X11 / (X4 - C_37_97)
                        )
                        + X7
                    )
                    + X9 * X15
                    + X18
                )
            )
            + C_1_75
            * (
                X3
                / (
                    np.exp(3 * X0 ** 3)
                    + np.exp(X5)
                )
            )
        )
        + X1 / (
            np.exp(3 * np.sqrt(X10 / X3))
            + np.exp(3 * X0 ** 3) * X18
        )
        + X7
        + X1 / (
            X5
            + X14
            + C_0_98
            * (
                X11
                + (
                    X12 ** 2 * np.exp(X12)
                ) / (X11 ** 2)
            )
            * (X11 / X8)
        )
        - X4 / C_31_51
    )

    # -------------------------
    # T6
    # -------------------------
    T6 = (
        X11 / (
            X10
            + (
                X2 / X10
                + C_33_56
                * (X2 + np.exp(X12))
                * (C_12_03 / X8)
            )
            / (
                (
                    X10 ** 2
                    * X14
                    * np.exp(X18 ** 3)
                ) / C_37_97
                + X6 ** 3
            )
        )
        + C_77_11 / (
            X4
            + (
                -C_9_23 / X10
                + X6
                * (
                    np.exp(np.exp(X0))
                    + np.log(X2)
                    * np.exp(np.sqrt(X3))
                    * (X2 / X8)
                )
                * ((X2 - C_9_23) / X8)
            )
            / (X3 - X13)
        )
        + (X9 - X4 - C_39_37)
        / (
            X4
            + X17 * (
                (X11 - X5) ** 2 / X0
            )
        )
        + C_33_34
    )

    F = C_33_56 / (T1 + T2 + T3 + T4 + T5 + T6)

    return F
