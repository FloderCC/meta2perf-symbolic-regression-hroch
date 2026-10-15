"""Meaningful equation and information-flow checks on small synthetic data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analysis.model_diagnostics import (
    directional_report, explain_random_forest, hroch_directional_audit,
    shap_row_selection, split_frames,
)
from src.equations.hroch import DEFAULT_COEFFICIENTS, FEATURE_NAMES, predict


def temporary_output():
    # Explicitly stay within the repository on Windows sandboxed hosts too.
    return tempfile.TemporaryDirectory(prefix="model_diagnostic_test_",
                                       dir=Path(__file__).resolve().parents[1])


def synthetic_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    values = rng.uniform(0.8, 2, (n, len(FEATURE_NAMES)))
    values[:, 4] = rng.uniform(5, 15, n)
    values[:, 6] = rng.uniform(2, 5, n)
    values[:, 13] = 2.0
    values[:, 15] = 1.0
    frame = pd.DataFrame(values, columns=FEATURE_NAMES)
    frame.insert(0, "row_id", np.arange(n) * 3 + 17)
    frame["MCC"] = rng.uniform(-0.2, 1, n)
    frame["Dataset"] = [f"dataset_{i // 4}" for i in range(n)]
    frame["Seed"] = 42
    frame["Sample Size"] = 0.5
    frame["Model"] = ["RF", "ExtraTrees", "MLP", "DT"] * (n // 4)
    return frame


class DirectionalTests(unittest.TestCase):
    def test_all_analytic_partials_match_independent_finite_differences(self):
        frame = synthetic_frame()
        coefficients = DEFAULT_COEFFICIENTS * np.array([1.1, 0.9, 1.2, 1.05, 0.8, 1.1])
        analytic = hroch_directional_audit(frame, coefficients).to_numpy()
        for feature_index, name in enumerate(FEATURE_NAMES):
            plus, minus = frame.copy(), frame.copy()
            step = 1e-5 * np.maximum(1, np.abs(frame[name].to_numpy()))
            plus[name] = frame[name] + step
            minus[name] = frame[name] - step
            numerical = (predict(plus, coefficients) - predict(minus, coefficients)) / (2 * step)
            np.testing.assert_allclose(analytic[:, feature_index], numerical, atol=2e-8, rtol=2e-5,
                                       err_msg=f"Derivative mismatch for {name}")

    def test_paper_outlier_and_gravity_formulas_match_full_derivative(self):
        frame = synthetic_frame()
        x = frame.loc[:, list(FEATURE_NAMES)].to_numpy()
        a, b, c, d, e, f = DEFAULT_COEFFICIENTS
        t1 = x[:, 13] - x[:, 15] - a
        t2 = x[:, 11] + b
        t3 = x[:, 11] + x[:, 9] + x[:, 1]
        t4 = (x[:, 14] + t1) / t1 * x[:, 15] * t2
        t5 = x[:, 10] * t4 / t3
        h = (c + (x[:, 4] - d / x[:, 6]) * (x[:, 13] - x[:, 15])) * x[:, 0] / x[:, 2]
        t6 = h**2 + x[:, 4] + t2
        numerator = (x[:, 17] + 1) * t5 - t6 - (x[:, 5] + x[:, 6]) / e + x[:, 1]
        denominator = t3 * f * (t4 + x[:, 9] + x[:, 1]) - t6
        derivatives = hroch_directional_audit(frame, DEFAULT_COEFFICIENTS)
        np.testing.assert_allclose(derivatives["nr_outliers"], (1 + x[:, 17]) * t4 / (t3 * denominator))
        np.testing.assert_allclose(derivatives["gravity"], 2 * h**2 * (denominator - numerator)
                                   / (x[:, 2] * denominator**2))

    def test_unused_inputs_have_exactly_zero_derivatives(self):
        result = hroch_directional_audit(synthetic_frame(), DEFAULT_COEFFICIENTS)
        for name in ["inst_to_attr", "nr_cor_attr", "nr_inst", "Processing Units Number",
                     "Regularization", "Representational Capacity"]:
            np.testing.assert_array_equal(result[name], np.zeros(len(result)))

    def test_zero_gravity_remains_invalid_and_counts_are_not_hidden(self):
        frame = synthetic_frame()
        frame.loc[0, "gravity"] = 0
        with temporary_output() as folder:
            report = directional_report(frame, DEFAULT_COEFFICIENTS, Path(folder), complete_frame=frame)
            signs = pd.read_csv(Path(folder) / "hroch_derivative_signs.csv")
        self.assertEqual(report["zero_corpus_gravity"], 1)
        self.assertEqual(report["invalid_raw_predictions"], 1)
        self.assertEqual(report["test_metrics_raw"]["invalid_predictions"], 1)
        self.assertIsNone(report["test_metrics_raw"]["r2"])
        np.testing.assert_array_equal(signs[["n_positive", "n_negative", "n_zero", "n_invalid"]].sum(axis=1),
                                      np.full(len(FEATURE_NAMES), len(frame)))


class PartitionTests(unittest.TestCase):
    def test_split_joins_by_row_id_and_rejects_omissions_or_overlap(self):
        frame = synthetic_frame()
        split = pd.DataFrame({"row_id": frame["row_id"].to_numpy()[::-1],
                              "partition": ["test"] * 10 + ["train"] * 30})
        train, test = split_frames(frame, split)
        self.assertEqual(train["row_id"].tolist(), split.loc[split["partition"] == "train", "row_id"].tolist())
        self.assertFalse(set(train["row_id"]) & set(test["row_id"]))
        with self.assertRaisesRegex(ValueError, "exactly the same"):
            split_frames(frame, split.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            split_frames(frame, pd.concat([split, split.iloc[:1]]))

    def test_shap_background_and_explained_rows_are_partitioned_and_reproducible(self):
        frame = synthetic_frame()
        train, test = frame.iloc[:30].copy(), frame.iloc[30:].copy()
        background, explained = shap_row_selection(train, test, background_count=8, sample_count=5)
        repeat_background, repeat_explained = shap_row_selection(train, test, background_count=8, sample_count=5)
        pd.testing.assert_frame_equal(background, repeat_background)
        pd.testing.assert_frame_equal(explained, repeat_explained)
        self.assertTrue(set(background["row_id"]) <= set(train["row_id"]))
        self.assertTrue(set(explained["row_id"]) <= set(test["row_id"]))
        self.assertFalse(set(background["row_id"]) & set(test["row_id"]))

    @unittest.skipUnless(importlib.util.find_spec("shap"), "SHAP not installed")
    def test_actual_tiny_rf_uses_only_descriptor_columns_and_training_background(self):
        frame = synthetic_frame()
        split = pd.DataFrame({"row_id": frame["row_id"], "partition": ["train"] * 30 + ["test"] * 10})
        with temporary_output() as folder:
            report = explain_random_forest(frame, split, Path(folder),
                                           {"n_estimators": 8, "max_depth": 3},
                                           shap_background=8, shap_samples=5)
            values = pd.read_csv(Path(folder) / "random_forest_shap_values.csv")
        self.assertEqual(report["feature_order"], list(FEATURE_NAMES))
        self.assertTrue(set(report["background_row_ids"]) <= set(frame.iloc[:30]["row_id"]))
        self.assertFalse(set(report["background_row_ids"]) & set(frame.iloc[30:]["row_id"]))
        self.assertTrue(set(report["explained_row_ids"]) <= set(frame.iloc[30:]["row_id"]))
        self.assertEqual(values.shape, (5, len(FEATURE_NAMES) + 1))
        self.assertLess(report["maximum_additivity_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
