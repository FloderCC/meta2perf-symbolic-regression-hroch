"""Verify feature-specific reuse and tiny training-only CV, without full grids."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.equations.hroch import FEATURE_NAMES
from src.experiments.baselines import select_configurations as selection
from src.utils.artifact_integrity import file_sha256
from src.utils.gp_integrity import validate_summary_context


TEMP_ROOT = Path(__file__).resolve().parents[1] / "tmp"


def example_training_frame():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(.2, 2., size=(24, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    frame["row_id"] = np.arange(10_000, 10_024)
    frame["MCC"] = .1 + .3 * frame["class_ent"]
    frame["Dataset"], frame["Model"], frame["Seed"] = "identity", "identity", 999999
    return frame


class ReferenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = example_training_frame()
        TEMP_ROOT.mkdir(exist_ok=True)

    def test_legacy_gp_summary_cannot_authorize_the_corrected_meta_dataset(self):
        legacy = pd.DataFrame({"train_r2": [.5]})
        old_context = {"data_sha256": "e6cecac6725e83c957c0d7fbcb29a5a8d6685ed783a71d5992117b1b6348e5da"}
        data = TEMP_ROOT.parent / "src/outputs/metadataset/archived/meta_dataset.csv"
        current = {"data_sha256": file_sha256(data), "fingerprint": "current-context"}
        archived = TEMP_ROOT / "archived/summary.csv"
        validate_summary_context(legacy, archived, old_context, allow_legacy_archive=True)
        with self.assertRaisesRegex(ValueError, "no compatibility metadata"):
            validate_summary_context(legacy, archived, current, allow_legacy_archive=True)
        with self.assertRaisesRegex(ValueError, "no compatibility metadata"):
            validate_summary_context(legacy, TEMP_ROOT / "new/summary.csv", old_context,
                                     allow_legacy_archive=True)
        validate_summary_context(pd.DataFrame({"context_fingerprint": ["current-context"]}),
                                 TEMP_ROOT / "new/summary.csv", current)

    def test_only_missing_joint_winners_require_new_search(self):
        joint = selection.selection_plan(feature_set="joint")
        dataset = selection.selection_plan(feature_set="dataset_only")
        self.assertEqual(joint["total_cv_fits"], 304)
        self.assertEqual(dataset["total_cv_fits"], 0)
        self.assertEqual(joint["models"]["DecisionTreeRegressor"]["n_candidates"], 48)
        self.assertEqual(joint["models"]["GradientBoostingRegressor"]["n_candidates"], 16)
        self.assertEqual(joint["models"]["MLPRegressor"]["n_candidates"], 12)
        self.assertEqual(joint["models"]["RandomForestRegressor"]["cv_fits"], 0)

    def test_archived_dataset_only_configs_are_reused_without_fitting(self):
        with patch.object(selection, "GridSearchCV", side_effect=AssertionError("Archived settings must not retune")):
            result = selection.select_parameters(self.frame, "dataset_only", n_jobs=1)
        self.assertEqual(result["parameters"]["RandomForestRegressor"]["max_depth"], 8)
        self.assertEqual(result["selection"]["DecisionTreeRegressor"]["method"], "archived_configuration")
        self.assertEqual(result["selection"]["LinearRegression"]["method"], "default_parameters")
        self.assertIsNone(result["selection"]["DecisionTreeRegressor"]["best_cv_mse"])
        self.assertEqual(len(result["feature_order"]), 12)

    def test_joint_rf_uses_its_joint_archive_and_keeps_source_evidence(self):
        with patch.object(selection, "GridSearchCV", side_effect=AssertionError("Joint RF winner is available")):
            result = selection.select_parameters(self.frame, "joint", n_jobs=1,
                                                 models=["LinearRegression", "RandomForestRegressor"])
        self.assertEqual(result["parameters"]["RandomForestRegressor"]["max_depth"], 16)
        self.assertEqual(result["parameters"]["RandomForestRegressor"]["max_features"], .5)
        detail = result["selection"]["RandomForestRegressor"]
        self.assertEqual(detail["selection_features"], "joint")
        self.assertIn("source_sha256", detail)

    def test_real_tiny_cv_receives_only_supplied_training_and_no_identifiers(self):
        captured = []
        real_class = selection.GridSearchCV

        class RecordingSearch(real_class):
            def fit(self, X, y, **kwargs):
                captured.append((X.copy(), y.copy(), self.scoring, len(self.cv), self.refit))
                return super().fit(X, y, **kwargs)

        with patch.object(selection, "GridSearchCV", RecordingSearch):
            result = selection.select_parameters(self.frame, "joint", n_jobs=1,
                                                 models=["DecisionTreeRegressor"], smoke_test=True)
        X, y, scoring, folds, refit = captured[0]
        np.testing.assert_array_equal(X, self.frame.loc[:, list(FEATURE_NAMES)].to_numpy())
        np.testing.assert_array_equal(y, self.frame.MCC.to_numpy())
        self.assertEqual(scoring, "neg_mean_squared_error")
        self.assertEqual(folds, 4)
        self.assertFalse(refit)
        detail = result["selection"]["DecisionTreeRegressor"]
        self.assertEqual(detail["method"], "training_grid_search")
        self.assertEqual(detail["cv_fits"], 4)
        self.assertTrue(detail["smoke_test"])
        self.assertAlmostEqual(detail["best_cv_mse"], -detail["best_score"])

    def test_completed_search_resumes_and_changed_artifact_or_context_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="selection_test_", dir=TEMP_ROOT) as directory:
            settings = dict(n_jobs=1, models=["DecisionTreeRegressor"], smoke_test=True,
                            output_dir=Path(directory) / "search", context={"training_split": "same"})
            first = selection.select_parameters(self.frame, "joint", **settings)
            with patch.object(selection, "GridSearchCV", side_effect=AssertionError("Completed search must recover")):
                recovered = selection.select_parameters(self.frame, "joint", resume=True, **settings)
            self.assertEqual(recovered["parameters"], first["parameters"])
            changed = {**settings, "context": {"training_split": "different"}}
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                selection.select_parameters(self.frame, "joint", resume=True, **changed)
            checkpoint = settings["output_dir"] / "checkpoints" / "DecisionTreeRegressor.json"
            original_payload = json.loads(checkpoint.read_text())
            payload = json.loads(checkpoint.read_text())
            payload["result"]["best_score"] -= .001
            payload["result"]["best_cv_mse"] = -payload["result"]["best_score"]
            checkpoint.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "cross-validation evidence"):
                selection.select_parameters(self.frame, "joint", resume=True, **settings)
            checkpoint.write_text(json.dumps(original_payload))
            cv_path = settings["output_dir"] / "DecisionTreeRegressor_cv_results.csv"
            cv_path.write_text("corrupted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed reference CV"):
                selection.select_parameters(self.frame, "joint", resume=True, **settings)

    def test_nonpositive_or_nonfinite_rows_are_not_silently_removed(self):
        for value in (0, -.1, np.nan):
            frame = self.frame.copy()
            frame.loc[0, "MCC"] = value
            with self.assertRaisesRegex(ValueError, "MCC > 0"):
                selection.select_parameters(frame, "joint", n_jobs=1, models=["DecisionTreeRegressor"])

    def test_archive_checkpoint_does_not_relabel_its_provenance(self):
        with tempfile.TemporaryDirectory(prefix="selection_test_", dir=TEMP_ROOT) as directory:
            settings = dict(n_jobs=1, models=["RandomForestRegressor"],
                            output_dir=Path(directory) / "search", context={"training_split": "same"})
            first = selection.select_parameters(self.frame, "joint", **settings)
            recovered = selection.select_parameters(self.frame, "joint", resume=True, **settings)
            self.assertEqual(first["selection"], recovered["selection"])
            saved = json.loads((settings["output_dir"] / "selection_config.json").read_text())
            self.assertEqual(saved["archived_configuration_source"]["selection_features"], "joint")
            checkpoint = settings["output_dir"] / "checkpoints" / "RandomForestRegressor.json"
            payload = json.loads(checkpoint.read_text())
            payload["result"]["method"] = "training_grid_search"
            checkpoint.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "selection method"):
                selection.select_parameters(self.frame, "joint", resume=True, **settings)


if __name__ == "__main__":
    unittest.main()
