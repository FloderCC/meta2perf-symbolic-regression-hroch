"""Small real-optimizer checks; no manuscript experiments are run here."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.equations import extended, hroch
from src.experiments.baselines import benchmark_lodo
from src.experiments.recalibration import recalibrate_hroch_equation_lodo as lodo
from src.experiments.recalibration import recalibrate_extended_equation_lodo as deap_lodo

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "tmp"


def temporary_directory():
    TEST_TEMP_ROOT.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="lodo_fit_test_", dir=TEST_TEMP_ROOT)


def example_inputs():
    row = np.array([0.8, 8., 1.2, 50., 20., 0., 2., 0.5, 1000., 1., 50., 2.,
                    2., 2., 2., 1., 3., 2., 3.])
    X = np.tile(row, (8, 1))
    X[:, 2] += np.arange(8) * .05
    X[:, 10] += np.arange(8)
    return X


class LODOCoefficientFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = example_inputs()
        cls.y = np.linspace(.05, .6, len(cls.X))
        cls.hroch_fit = lodo.calibrate_fold(cls.X, cls.y, lodo.coefficient_bounds(.1, 10.),
                                            n_pop=4, n_iter=1, n_jobs=1)
        cls.deap_fit = deap_lodo.calibrate_fold(cls.X, cls.y, deap_lodo.coefficient_bounds(.1, 10.),
                                               n_pop=4, n_iter=1, n_jobs=1)

    def test_real_optimizers_are_bounded_and_use_only_supplied_training(self):
        for module, equation, result in ((lodo, hroch, self.hroch_fit),
                                         (deap_lodo, extended, self.deap_fit)):
            bounds = module.coefficient_bounds(.1, 10.)
            self.assertEqual(set(result["methods"]), {"DE", "EGWO"})
            for detail in result["methods"].values():
                coefficients = np.array(detail["coefficients"])
                self.assertTrue(np.all(coefficients >= bounds[:, 0]))
                self.assertTrue(np.all(coefficients <= bounds[:, 1]))
                prediction = equation.predict(self.X, coefficients)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertAlmostEqual(detail["train_mse"], np.mean((prediction - self.y) ** 2))
                self.assertLessEqual(detail["train_mse"], result["original_train_mse"])
                self.assertEqual(len(detail["best_mse_history"]), 1)

    def test_deap_is_grouped_nineteen_coefficient_form(self):
        for detail in self.deap_fit["methods"].values():
            self.assertEqual(len(detail["coefficients"]), 19)
        self.assertEqual(deap_lodo.COEFFICIENT_NAMES, extended.COEFFICIENT_NAMES)

    def test_negative_reference_bounds_keep_their_sign(self):
        bounds = lodo.coefficient_bounds(.1, 10.)
        self.assertLess(bounds[-1, 0], bounds[-1, 1])
        self.assertLess(bounds[-1, 1], 0)
        np.testing.assert_allclose(bounds[-1], sorted(hroch.DEFAULT_COEFFICIENTS[-1] * np.array([.1, 10.])))

    def test_selecting_one_optimizer_preserves_its_seeded_fit(self):
        repeated = lodo.calibrate_fold(self.X, self.y, lodo.coefficient_bounds(.1, 10.),
                                      n_pop=4, n_iter=1, n_jobs=1, methods=("DE",))
        self.assertEqual(set(repeated["methods"]), {"DE"})
        for field in ("coefficients", "train_mse", "best_mse_history"):
            np.testing.assert_array_equal(repeated["methods"]["DE"][field], self.hroch_fit["methods"]["DE"][field])
        self.assertEqual(lodo.parse_args([]).methods, ["DE", "EGWO"])
        self.assertEqual(deap_lodo.parse_args([]).methods, ["DE", "EGWO"])

    def test_invalid_training_domain_is_not_silently_dropped(self):
        X = self.X.copy()
        X[0, 2] = 0
        with self.assertRaisesRegex(ValueError, "finite training predictions"):
            lodo.calibrate_fold(X, self.y, lodo.coefficient_bounds(.1, 10.),
                                n_pop=4, n_iter=1, n_jobs=1)
        X[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite nonempty"):
            lodo.calibrate_fold(X, self.y, lodo.coefficient_bounds(.1, 10.),
                                n_pop=4, n_iter=1, n_jobs=1)

    def test_clipping_does_not_hide_undefined_predictions(self):
        clipped = lodo.clip_to_training(np.array([np.inf, np.nan, -2., 2.]), np.array([-.2, .6]))
        self.assertTrue(np.isnan(clipped[:2]).all())
        np.testing.assert_array_equal(clipped[2:], [-.2, .6])

    def test_optimizer_selection_validation(self):
        for methods in ((), ("DE", "DE"), ("local",)):
            with self.assertRaises(ValueError):
                lodo.calibrate_fold(self.X, self.y, lodo.coefficient_bounds(.1, 10.),
                                    n_pop=4, n_iter=1, n_jobs=1, methods=methods)

    def test_resume_requires_exact_configuration_and_valid_fold(self):
        with temporary_directory() as folder:
            directory = Path(folder) / "run"
            lodo.prepare_output(directory, {"data_sha256": "new", "methods": ["DE"]}, False)
            lodo.prepare_output(directory, {"data_sha256": "new", "methods": ["DE"]}, True)
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                lodo.prepare_output(directory, {"data_sha256": "old", "methods": ["DE"]}, True)
        fold = copy.deepcopy(self.hroch_fit)
        fold.update(dataset="held-out", row_ids=[8, 9], n_train=len(self.y), run_config_sha256="config")
        settings = dict(label="held-out", row_ids=[8, 9], n_train=len(self.y),
                        bounds=lodo.coefficient_bounds(.1, 10.), methods=["DE", "EGWO"], n_iter=1,
                        run_config_sha256="config")
        lodo.validate_fold_checkpoint(fold, **settings)
        for field, value in (("row_ids", [8]), ("run_config_sha256", "other")):
            broken = copy.deepcopy(fold)
            broken[field] = value
            with self.assertRaises(ValueError):
                lodo.validate_fold_checkpoint(broken, **settings)
        broken = copy.deepcopy(fold)
        broken["methods"]["DE"]["best_mse_history"] = []
        with self.assertRaises(ValueError):
            lodo.validate_fold_checkpoint(broken, **settings)

    def test_baseline_positive_filter_requires_matching_sr_manifest(self):
        frame = pd.DataFrame(self.X, columns=hroch.FEATURE_NAMES)
        frame["Seed"], frame["Sample Size"], frame["Model"] = 42, .1, "MLP"
        frame["Dataset"] = [f"dataset-{index}" for index in range(len(frame))]
        frame["MCC"] = np.linspace(-.2, .6, len(frame))
        with temporary_directory() as folder:
            data_path, config_path = Path(folder) / "data.csv", Path(folder) / "run_config.json"
            frame.to_csv(data_path, index=False)
            positive, count = benchmark_lodo.load_data(data_path)
            self.assertLess(len(positive), count)
            self.assertTrue((positive["MCC"] > 0).all())
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            config = {"original_rows": count, "included_rows": len(positive), "mcc_filter": "MCC > 0",
                      "all_datasets": sorted(positive["Dataset"].unique()), "feature_order": list(hroch.FEATURE_NAMES),
                      "seed": 42, "data_sha256": digest}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            references = benchmark_lodo.verify_sr_inputs([config_path], positive, count, digest, data_path)
            self.assertEqual(references[0]["input_match"], "exact_bytes")
            config["mcc_filter"] = "all"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mcc_filter"):
                benchmark_lodo.verify_sr_inputs([config_path], positive, count, digest, data_path)
            config["mcc_filter"], config["data_sha256"] = "MCC > 0", "historical"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data_sha256"):
                benchmark_lodo.verify_sr_inputs([config_path], positive, count, digest, data_path)

    def test_small_end_to_end_lodo_and_resume_for_symbolic_and_baseline(self):
        frame = pd.DataFrame(self.X, columns=hroch.FEATURE_NAMES)
        frame["Seed"], frame["Sample Size"], frame["Model"] = 42, .1, "MLP"
        frame["Dataset"] = [f"dataset-{index // 2}" for index in range(len(frame))]
        frame["Sample Size"] = np.tile([.1, .2], len(frame) // 2)
        frame["MCC"] = self.y
        with temporary_directory() as folder:
            folder = Path(folder)
            data = folder / "data.csv"
            frame.to_csv(data, index=False)
            config_paths = []
            for name, module in (("hroch", lodo), ("deap", deap_lodo)):
                output = folder / name
                argv = ["--data", str(data), "--output-dir", str(output), "--methods", "DE",
                        "--max-folds", "1", "--population-size", "4",
                        "--iterations", "1", "--n-jobs", "1"]
                summary = module.run(module.parse_args(argv))
                self.assertEqual(set(summary["method"]), {"original_constants", "train_mean", "DE"})
                self.assertEqual(set(summary["evaluation_scope"]), {"selected_datasets"})
                config_paths.append(output / "run_config.json")
                with patch.object(module, "calibrate_fold", side_effect=AssertionError("Resume must not refit")):
                    module.run(module.parse_args([*argv, "--resume"]))
            output = folder / "baseline"
            parameters = folder / "parameters.json"
            parameters.write_text(json.dumps({"LinearRegression": {}}), encoding="utf-8")
            argv = ["--data", str(data), "--output-dir", str(output),
                    "--hyperparameters", str(parameters), "--sr-config", *map(str, config_paths), "--max-folds", "1", "--n-jobs", "1",
                    "--models", "LinearRegression", "--bootstrap", "0"]
            summary = benchmark_lodo.run(benchmark_lodo.parse_args(argv))
            self.assertEqual(set(summary["method"]), {"LinearRegression"})
            self.assertTrue(summary["complete"].all())
            self.assertFalse(summary["full_corpus_complete"].any())
            with patch.object(benchmark_lodo, "fit_fold", side_effect=AssertionError("Resume must not refit")):
                benchmark_lodo.run(benchmark_lodo.parse_args([*argv, "--resume"]))


if __name__ == "__main__":
    unittest.main()
