"""Check corrected descriptor rules and the saved corpus without training models."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ["Seed", "Dataset", "Sample Size", "Model"]
TASK_KEYS = ["Seed", "Dataset", "Sample Size"]
DATASET_FEATURES = [
    "class_ent", "eq_num_attr", "gravity", "inst_to_attr", "nr_attr", "nr_bin",
    "nr_class", "nr_cor_attr", "nr_inst", "nr_norm", "nr_outliers", "ns_ratio",
]


def descriptor_functions():
    """Load only pure definitions, without executing the pipeline's file I/O."""
    source = ROOT / "src/pipeline/build_meta_dataset.py"
    required = {
        "encode_training_complexity", "get_training_complexity_given_model",
        "encode_prediction_complexity", "get_prediction_complexity_given_model",
        "encode_regularization", "get_regularization_number_of_parameters_given_model",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in required]
    if {node.name for node in definitions} != required:
        raise AssertionError("The builder's descriptor definitions are incomplete.")
    namespace = {"ast": ast}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


FUNCTIONS = descriptor_functions()
training = FUNCTIONS["get_training_complexity_given_model"]
inference = FUNCTIONS["get_prediction_complexity_given_model"]
regularization = FUNCTIONS["get_regularization_number_of_parameters_given_model"]


class DescriptorDefinitionTests(unittest.TestCase):
    def test_corrected_complexities(self):
        self.assertEqual(training("HistGradientBoosting"), 1)
        self.assertEqual(inference("HistGradientBoosting"), 0)
        for model in ("MLP", "DNN"):
            with self.subTest(model=model):
                self.assertEqual(training(model), 1)
                self.assertEqual(inference(model), 1)

    def test_positive_alpha_enables_l2_or_additive_smoothing(self):
        for model, expected in (("MLP", 3), ("DNN", 3), ("Ridge", 3), ("BernoulliNB", 1)):
            for alpha in (0.0001, 0.1, 1.0, 1e-12):
                with self.subTest(model=model, alpha=alpha):
                    self.assertEqual(regularization(model, f"'alpha': {alpha!r}"), expected)

    def test_zero_alpha_disables_regularization_or_smoothing(self):
        for model in ("MLP", "DNN", "Ridge", "BernoulliNB"):
            with self.subTest(model=model):
                self.assertEqual(regularization(model, "'alpha': 0"), 0)

    def test_unaffected_definitions(self):
        for model, expected in (("LR", (1, 1)), ("Ridge", (3, 1)), ("RF", (2, 0))):
            with self.subTest(model=model):
                self.assertEqual((training(model), inference(model)), expected)
        self.assertEqual(regularization("SGD", "'alpha': 0.0001"), 3)
        self.assertEqual(regularization("RF", "'n_estimators': 100"), 0)
        self.assertEqual(regularization("HistGradientBoosting", "'l2_regularization': 0"), 0)
        self.assertEqual(regularization("HistGradientBoosting", "'l2_regularization': 1"), 3)


class SavedMetaDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = ROOT / "src/outputs/metadataset/archived"
        required = ("meta_dataset.csv", "base_classifier_hyperparameters.csv",
                    "base_classifier_performance.csv", "dataset_metafeatures.csv")
        if not all((cls.bank / name).is_file() for name in required):
            raise unittest.SkipTest("The saved input bank is unavailable.")
        cls.frame = pd.read_csv(cls.bank / "meta_dataset.csv")
        cls.tuning = pd.read_csv(cls.bank / "base_classifier_hyperparameters.csv")

    def test_corpus_keys_targets_and_dataset_descriptors_are_preserved(self):
        frame = self.frame
        self.assertEqual(len(frame), 25836)
        self.assertFalse(frame.duplicated(MODEL_KEYS).any())
        self.assertEqual(frame["Dataset"].nunique(), 65)
        self.assertEqual(frame["Model"].nunique(), 16)
        self.assertEqual((frame["MCC"] > 0).sum(), 23323)
        self.assertEqual((frame["MCC"] == 0).sum(), 1960)
        self.assertEqual((frame["MCC"] < 0).sum(), 553)
        performance = pd.read_csv(self.bank / "base_classifier_performance.csv")
        targets = frame[MODEL_KEYS + ["MCC"]].merge(
            performance[MODEL_KEYS + ["MCC"]], on=MODEL_KEYS,
            how="outer", validate="one_to_one", suffixes=("", "_saved"), indicator=True,
        )
        self.assertTrue(targets["_merge"].eq("both").all())
        assert_series_equal(targets["MCC"], targets["MCC_saved"], check_names=False)
        features = pd.read_csv(self.bank / "dataset_metafeatures.csv")
        joined = frame[TASK_KEYS + DATASET_FEATURES].merge(
            features[TASK_KEYS + DATASET_FEATURES], on=TASK_KEYS,
            how="left", validate="many_to_one", suffixes=("", "_saved"), indicator=True,
        )
        self.assertTrue(joined["_merge"].eq("both").all())
        for column in DATASET_FEATURES:
            assert_series_equal(joined[column], joined[column + "_saved"], check_names=False)

    def test_corrected_csv_matches_inline_definitions_without_further_changes(self):
        frame = self.frame
        keyed = frame[MODEL_KEYS].merge(
            self.tuning.sample(frac=1, random_state=42)[MODEL_KEYS + ["Model Parameters"]],
            on=MODEL_KEYS, how="left", sort=False, validate="one_to_one",
        )
        assert_frame_equal(keyed[MODEL_KEYS], frame[MODEL_KEYS])
        self.assertFalse(keyed["Model Parameters"].isna().any())
        rebuilt = frame.copy(deep=True)
        rebuilt["Training Complexity"] = frame["Model"].map(training)
        rebuilt["Prediction Complexity"] = frame["Model"].map(inference)
        rebuilt["Regularization"] = [
            regularization(model, parameters)
            for model, parameters in keyed[["Model", "Model Parameters"]].itertuples(index=False, name=None)
        ]
        assert_frame_equal(rebuilt, frame)


if __name__ == "__main__":
    unittest.main()
