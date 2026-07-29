"""Shared helpers for the paper's row-wise meta-level analyses."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


META_RANDOM_STATE = 42
META_TEST_SIZE = 0.2


def stratified_regression_split(
    X,
    y,
    *,
    test_size: float = META_TEST_SIZE,
    bins: str | int = "auto",
    random_state: int = META_RANDOM_STATE,
):
    """Split rows after stratifying a continuous target into histogram bins.

    This reproduces the row-wise protocol used for the reported results. It is
    an in-corpus interpolation split: dataset and classifier identities can
    occur in both partitions.
    """

    if isinstance(bins, str):
        bin_edges = np.histogram_bin_edges(y, bins=bins)[:-1]
    elif isinstance(bins, int):
        bin_edges = np.linspace(min(y), max(y), num=bins, endpoint=False)
    else:
        raise TypeError(f"bins must be a string or integer, got {type(bins).__name__}")

    groups = np.digitize(y, bin_edges)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=groups,
        random_state=random_state,
    )


def smape(y_true, y_pred) -> float:
    """Return symmetric mean absolute percentage error.

    The paper analysis filters to MCC > 0, so the denominator is non-zero for
    the retained targets. ``nanmean`` also makes the helper explicit about any
    undefined 0/0 values produced by external inputs.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmean(np.abs(y_pred - y_true) / denominator))


def adjusted_r2(r2_value: float, sample_count: int, predictor_count: int) -> float:
    """Compute adjusted R² for a fitted regression model."""

    return 1 - (1 - r2_value) * (
        (sample_count - 1) / (sample_count - predictor_count - 1)
    )
