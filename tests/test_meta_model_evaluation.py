"""Integration-boundary checks for the meta-model evaluation pipeline."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.equations import extended, hroch
from src.pipeline import evaluate_meta_models as runner
from src.utils.pipeline_integrity import read_meta_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'src/outputs/metadataset/archived/meta_dataset.csv'
TUNING = DATA.parent / 'base_classifier_hyperparameters.csv'


class MetaModelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = read_meta_dataset(DATA)
        cls.corrected = cls.original


    def test_positive_split_retains_historical_membership_and_saved_fitting_order(self):
        old = runner.select_scope(self.original, 'positive')
        new = runner.select_scope(self.corrected, 'positive')
        old_train, old_test, _ = runner.make_split(old)
        train, test, split = runner.make_split(new)
        self.assertEqual(train.index.tolist(), old_train.index.tolist())
        self.assertEqual(test.index.tolist(), old_test.index.tolist())
        self.assertEqual(split.loc[split.partition == 'train', 'row_id'].tolist(), train.index.tolist())
        self.assertEqual(split.loc[split.partition == 'test', 'row_id'].tolist(), test.index.tolist())
        self.assertEqual((len(train), len(test)), (18658, 4665))

    def test_default_population_is_positive_and_all_range_is_rejected(self):
        args = runner.parse_args([])
        self.assertEqual(args.scopes, ['positive'])
        self.assertEqual(args.output_dir.parent.name, 'model_evaluation')
        frame = runner.select_scope(self.corrected, 'positive')
        self.assertEqual(len(frame), 23323)
        self.assertTrue((frame.MCC > 0).all())
        self.assertEqual(set(frame.row_id), set(self.original.index[self.original.MCC > 0]))
        with self.assertRaisesRegex(ValueError, 'only MCC > 0'):
            runner.select_scope(self.corrected, 'all')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(['--scopes', 'all'])

    def test_dry_run_does_not_fit_write_or_change_sources(self):
        before = (DATA.read_bytes(), TUNING.read_bytes())
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            output = Path(temporary) / 'not_created'
            args = runner.parse_args(['--output-dir', str(output), '--dry-run', '--n-jobs', '1'])
            with patch.object(runner, 'fit_reference', side_effect=AssertionError('dry run must not fit')):
                result = runner.run(args)
            self.assertFalse(output.exists())
            self.assertEqual(set(result['splits']), {'positive'})
            self.assertEqual(result['splits']['positive']['rows'], 23323)
            self.assertEqual(result['mcc_filter'], 'MCC > 0')
            self.assertEqual(result['evaluated_population']['rows'], 23323)
            self.assertEqual(result['source_corpus'], {'rows': 25836})
            self.assertEqual(result['source_sha256']['meta_dataset'], runner.file_sha256(DATA))
            self.assertEqual(result['corrected_data_sha256'], runner.file_sha256(DATA))
        self.assertEqual(before, (DATA.read_bytes(), TUNING.read_bytes()))

    def test_reference_search_budget_reuses_the_correct_feature_sets(self):
        args = runner.parse_args([])
        plan = {
            name: runner.selection_plan(runner.reference_model_names(args, name), feature_set=name)
            for name in ('joint', 'dataset_only')
        }
        self.assertEqual(plan['joint']['total_cv_fits'], 304)
        self.assertEqual(plan['dataset_only']['total_cv_fits'], 0)
        partial = runner.parse_args(['--models', 'LinearRegression'])
        self.assertEqual(runner.reference_model_names(partial, 'joint'),
                         ['LinearRegression', 'RandomForestRegressor'])
        self.assertEqual(runner.reference_model_names(partial, 'dataset_only'), ['LinearRegression'])

    def test_pipeline_uses_original_input_and_writes_only_canonical_metric_tables(self):
        def rowwise(selected, directory, config, args, scope):
            directory.mkdir(parents=True)
            table = pd.DataFrame({'model': ['HROCH'], 'r2': [.5]})
            runner.baselines.atomic_csv(directory / 'metrics.csv', table)
            return ({'HROCH': {'coefficients': hroch.DEFAULT_COEFFICIENTS.tolist()}},
                    table, {'joint': {'parameters': {}}})

        def lodo(data_path, directory, args, scope, parameters_path):
            self.assertEqual(data_path, DATA.resolve())
            directory.mkdir(parents=True)
            table = pd.DataFrame({'model': ['HROCH'], 'scope': [scope],
                                  'full_corpus_complete': [False], 'invalid_predictions': [0],
                                  'r2': [.5], 'mae': [.1], 'rmse': [.2]})
            runner.baselines.atomic_csv(directory / 'comparison.csv', table)
            return table

        def diagnostics(frame, rowwise_dir, output_dir, **kwargs):
            output_dir.mkdir(parents=True)
            runner.save_json(output_dir / 'diagnostics_report.json', {'status': 'complete'})

        before = runner.file_sha256(DATA)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            output = Path(temporary) / 'evaluation'
            args = runner.parse_args(['--output-dir', str(output), '--models', 'LinearRegression',
                                      '--skip-shap', '--n-jobs', '1', '--max-folds', '1'])
            with patch.object(runner, 'run_rowwise', side_effect=rowwise), \
                    patch.object(runner, 'run_lodo', side_effect=lodo), \
                    patch('src.analysis.model_diagnostics.run_diagnostics', side_effect=diagnostics), \
                    redirect_stdout(io.StringIO()):
                runner.run(args)
            self.assertTrue((output / 'positive/rowwise/metrics.csv').is_file())
            self.assertTrue((output / 'positive/lodo/comparison.csv').is_file())
            self.assertFalse((output / 'data').exists())
            self.assertFalse((output / 'rowwise_comparison.csv').exists())
            self.assertFalse((output / 'lodo_comparison.csv').exists())
            config = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['sources']['meta_dataset'], str(DATA.resolve()))
            self.assertEqual(config['corrected_data_sha256'], before)
        self.assertEqual(runner.file_sha256(DATA), before)

    def test_evaluation_ready_requires_all_five_baselines_and_65_positive_folds(self):
        args = runner.parse_args([])
        details = {'positive': {'datasets': 65, 'rows': 23323}}
        table = pd.DataFrame({'model': [*runner.EQUATIONS, *runner.baselines.MODEL_CLASSES],
                              'scope': 'positive', 'completed_folds': 65, 'expected_folds': 65,
                              'n': 23323})
        self.assertTrue(runner.evaluation_ready(args, details, {'complete': True}, table))
        self.assertFalse(runner.evaluation_ready(args, details, {'complete': False}, table))
        for field, value in [('scope', 'all'), ('completed_folds', 64), ('n', 23322)]:
            bad = table.copy()
            bad.loc[0, field] = value
            self.assertFalse(runner.evaluation_ready(args, details, {'complete': True}, bad))
        self.assertFalse(runner.evaluation_ready(args, details, {'complete': True}, table.iloc[:-1]))
        limited = runner.parse_args(['--shap-samples', '10'])
        self.assertFalse(runner.evaluation_ready(limited, details, {'complete': True}, table))

    def test_rowwise_selection_sees_only_train_and_each_fit_uses_its_own_features(self):
        args = runner.parse_args(['--models', 'RandomForestRegressor', '--skip-shap',
                                  '--n-jobs', '1', '--iterations', '1'])
        selected = runner.select_scope(self.corrected, 'positive')
        train, test, _ = runner.make_split(selected)
        recorded_selection, recorded_fit = [], []

        def selection(frame, feature_set, **kwargs):
            recorded_selection.append((feature_set, frame.row_id.tolist(), kwargs))
            count = 11 if feature_set == 'joint' else 3
            return {'parameters': {'RandomForestRegressor': {'n_estimators': count, 'max_depth': 2}},
                    'n_train': len(frame), 'feature_order': list(hroch.FEATURE_NAMES)
                    if feature_set == 'joint' else runner.DATASET_FEATURES}

        def reference(name, parameters, jobs, Xtrain, ytrain, Xtest):
            recorded_fit.append((name, copy.deepcopy(parameters), Xtrain.shape[1], Xtest.shape[1]))
            return {'train_prediction': np.full(len(ytrain), ytrain.mean()).tolist(),
                    'test_prediction': np.full(len(Xtest), ytrain.mean()).tolist()}

        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            directory = Path(temporary) / 'rowwise'
            with patch.object(runner, 'select_parameters', side_effect=selection), \
                    patch.object(runner, 'fit_reference', side_effect=reference), \
                    patch.object(runner, 'plot_symbolic'):
                _, table, evidence = runner.run_rowwise(selected, directory, {'test': True}, args, 'positive')
            coefficients = json.loads((directory / 'coefficients.json').read_text())
            predicted = pd.read_csv(directory / 'predictions.csv')
            joint = json.loads((directory / 'reference_parameters_joint.json').read_text())
            dataset = json.loads((directory / 'reference_parameters_dataset_only.json').read_text())
        self.assertEqual([name for name, _, _ in recorded_selection], ['joint', 'dataset_only'])
        for _, row_ids, kwargs in recorded_selection:
            self.assertEqual(row_ids, train.index.tolist())
            self.assertFalse(set(row_ids) & set(test.index))
            self.assertEqual(kwargs['context']['train_row_ids'], train.index.tolist())
        self.assertEqual([item[2:] for item in recorded_fit], [(19, 19), (12, 12)])
        self.assertEqual(recorded_fit[0][1]['RandomForestRegressor']['n_estimators'], 11)
        self.assertEqual(recorded_fit[1][1]['RandomForestRegressor']['n_estimators'], 3)
        self.assertEqual(joint, evidence['joint']['parameters'])
        self.assertEqual(dataset, evidence['dataset_only']['parameters'])
        self.assertEqual(set(table.scope), {'positive'})
        for name, equation in runner.EQUATIONS.items():
            self.assertEqual(coefficients[name]['coefficients'], equation.DEFAULT_COEFFICIENTS.tolist())
            self.assertEqual(coefficients[name]['procedure'], 'evaluate_supplied_final_equation')
            self.assertNotIn('methods', coefficients[name])
            rows = predicted.loc[predicted.predictor == name].set_index('row_id')
            self.assertEqual(set(rows.optimizer), {'supplied_equation'})
            np.testing.assert_allclose(rows.loc[test.index].y_pred_raw, equation.predict(test), rtol=1e-12, atol=1e-14)

    def test_lodo_receives_the_saved_joint_parameters_and_positive_filter(self):
        args = runner.parse_args(['--models', 'RandomForestRegressor', '--n-jobs', '1', '--max-folds', '1'])
        calls = []
        symbolic_summary = pd.DataFrame([{'method': 'DE', 'prediction_mode': 'raw',
                                          'complete': True, 'evaluation_scope': 'selected_datasets'}])
        reference_summary = pd.DataFrame([{'method': 'RandomForestRegressor', 'prediction_mode': 'raw'}])

        def parse(argv):
            calls.append(list(argv))
            return object()

        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            mapping_path = Path(temporary) / 'reference_parameters_joint.json'
            mapping_path.write_text(json.dumps({'RandomForestRegressor': {'max_depth': 16}}))
            with patch.object(runner.hroch_lodo, 'parse_args', side_effect=parse), \
                    patch.object(runner.deap_lodo, 'parse_args', side_effect=parse), \
                    patch.object(runner.baselines, 'parse_args', side_effect=parse), \
                    patch.object(runner.hroch_lodo, 'run', return_value=symbolic_summary), \
                    patch.object(runner.deap_lodo, 'run', return_value=symbolic_summary), \
                    patch.object(runner.baselines, 'run', return_value=reference_summary), \
                    patch.object(runner.pd, 'read_csv', return_value=pd.DataFrame({'method': ['DE']})), \
                    patch.object(runner.baselines, 'dataset_intervals', return_value={}), \
                    patch.object(runner.baselines, 'atomic_csv'):
                result = runner.run_lodo(DATA, Path(temporary) / 'lodo', args, 'positive', mapping_path)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('--include-nonpositive' not in argv for argv in calls))
        reference_args = calls[-1]
        self.assertEqual(reference_args[reference_args.index('--hyperparameters') + 1], str(mapping_path))
        self.assertEqual(set(result.scope), {'positive'})

    def test_invalid_bounds_fail_before_output_creation_even_in_dry_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            output = Path(temporary) / 'not_created'
            args = runner.parse_args(['--output-dir', str(output), '--dry-run', '--lower-factor', '10', '--upper-factor', '.1'])
            with self.assertRaises(ValueError):
                runner.run(args)
            self.assertFalse(output.exists())

    def test_supplied_equation_metadata_does_not_invent_fitting_history(self):
        self.assertFalse(hasattr(runner, 'refit_symbolic'))
        for name, equation in runner.EQUATIONS.items():
            record = runner.supplied_equation_record(name, equation)
            self.assertEqual(record['coefficients'], equation.DEFAULT_COEFFICIENTS.tolist())
            self.assertEqual(record['coefficient_order'], list(equation.COEFFICIENT_NAMES))
            self.assertEqual(record['source_sha256'], runner.file_sha256(ROOT / record['source_module']))
            self.assertNotIn('methods', record)
            self.assertNotIn('selected_method', record)
            self.assertNotIn('best_mse_history', record)


    def test_zero_zero_smape_is_retained_and_overflow_cannot_look_complete(self):
        self.assertEqual(runner.metrics([0., 1.], [0., 1.], 0)['smape'], 0.)
        with np.errstate(over='ignore', invalid='ignore'), self.assertRaises(ValueError):
            runner.metrics([0., 1.], [1e308, 1e308], 0)


if __name__ == '__main__':
    unittest.main()
