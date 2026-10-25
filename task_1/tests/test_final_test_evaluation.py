"""Искусственные проверки test-evaluator: predict-only, парность и календарь."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from final_test_evaluation import (evaluate_model, summarize, file_hash,
                                   verify_schema, pair_statistics, validate_schema)


class PairStatisticsTest(unittest.TestCase):
    def test_original_group_weights(self):
        reference = pd.DataFrame({'group_id': [1, 2, 3, 4], 'day': [285, 286, 286, 286],
                                  'group_size': [2, 2, 2, 2], 'ndcg': [.2, .5, .5, .5]})
        candidate = reference.copy()
        candidate.loc[0, 'ndcg'] = .4
        point, low, high = pair_statistics(candidate, reference, np.array([[0, 1]]))
        np.testing.assert_allclose([point, low, high], [.05, .05, .05])
        wrong = candidate.copy()
        wrong.loc[0, 'group_id'] = 10
        with self.assertRaises(AssertionError):
            pair_statistics(wrong, reference, np.array([[0, 1]]))

    def test_masks_and_direct_label_are_checked(self):
        plan = {'schemas': {'tiny': ['x', 'history_mean']}}
        model = {'schema': 'tiny', 'ignored_indices': [1],
                 'active_features': ['x'], 'active_count': 1}
        self.assertEqual(verify_schema(model, plan), ['x', 'history_mean'])
        model['active_features'] = ['history_mean']
        with self.assertRaises(ValueError):
            verify_schema(model, plan)
        plan['schemas']['tiny'] = ['x', 'label']
        with self.assertRaises(ValueError):
            verify_schema(model, plan)


class TinyEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.data = self.folder / 'data'
        self.output = self.folder / 'output'
        self.schema = self.data / 'tiny'
        self.schema.mkdir(parents=True)
        labels = np.tile([0, 1, 2, 1], 4).astype('int8')
        days = np.repeat([285, 287, 288, 289], 4).astype('int64')
        groups = np.repeat([1285, 2287, 3288, 4289], 4).astype('int64')
        values = np.c_[np.tile([0., 1., 2., 1.], 4),
                       np.arange(16, dtype='float32')].astype('float32')
        self.arrays = {'label': labels, 'group_id': groups,
                       'log_id': np.arange(16, dtype='int64'), 'day': days}
        np.savez_compressed(self.data / 'targets.npz', **self.arrays)
        np.savez_compressed(self.schema / 'test_targets.npz', **self.arrays)
        names = ['x', 'history_mean']
        pool = Pool(values, label=labels, group_id=groups, feature_names=names)
        # Только локальный fixture; семь настоящих моделей не загружаются.
        model = CatBoostRanker(iterations=3, depth=2, loss_function='YetiRank',
                              random_seed=42, thread_count=1, verbose=False,
                              allow_writing_files=False)
        pool.quantize(border_count=4, random_seed=42)
        model.fit(pool)
        model_path = self.folder / 'fixture.cbm'
        model.save_model(str(model_path))
        pool.save(str(self.schema / 'test.qbin'))
        borders = self.folder / 'fixture_borders.tsv'
        pool.save_quantization_borders(str(borders))
        (self.schema / 'test_applied_borders.tsv').write_bytes(borders.read_bytes())
        self.model = {'id': 'M-sub', 'family': 'sub',
                      'schema': 'tiny', 'active_count': 2, 'ignored_indices': [],
                      'active_features': names, 'tree_count': 3,
                      'model': {'path': str(model_path), 'sha256': file_hash(model_path)}}
        second = copy.deepcopy(self.model)
        second.update({'id': 'D462-full', 'family': 'full'})
        self.plan = {'plan_id': 'synthetic_fixture',
                     'models': [self.model, second], 'schemas': {'tiny': names},
                     'train_borders': {'tiny': {'path': str(borders), 'sha256': file_hash(borders)}},
                     'metrics': {'group_oracle_atol': 1e-12},
                     'bootstrap': {'blocks_days': [3, 4], 'replicates': 20000,
                                   'seed': 42, 'confidence': .95},
                     'comparisons': [{'candidate': 'D462-full', 'reference': 'M-sub',
                                      'role': 'synthetic'}],
                     'limitations': ['synthetic fixture only']}

    def tearDown(self):
        self.temp.cleanup()

    def test_predict_only_and_complete_summary(self):
        original_hash = file_hash(self.model['model']['path'])
        checked = validate_schema(self.plan, self.data, 'tiny')
        self.assertEqual(checked['rows'], 16)
        self.assertEqual(checked['groups'], 4)
        with patch.object(CatBoostRanker, 'fit', side_effect=AssertionError('fit forbidden')):
            for model in self.plan['models']:
                result = evaluate_model(model, self.plan, self.data, self.output)
                self.assertTrue(result['completed'])
                self.assertFalse(result['fit_executed'])
            again = evaluate_model(self.model, self.plan, self.data, self.output)
            self.assertTrue(again['completed'])
            summary = summarize(self.plan, self.data, self.output)
        self.assertEqual(original_hash, file_hash(self.model['model']['path']))
        self.assertFalse((self.output / 'models/M-sub/model.cbm').exists())
        self.assertEqual(summary['rows'], 16)
        self.assertEqual(summary['groups'], 4)
        self.assertEqual(summary['calendar_days'], 5)
        self.assertEqual(len(summary['comparisons']), 2)
        for comparison in summary['comparisons']:
            np.testing.assert_array_equal(
                [comparison['delta_ndcg'], comparison['ci_low'], comparison['ci_high']], [0, 0, 0])
        daily = pd.read_csv(self.output / 'models/M-sub/daily_ndcg.csv')
        self.assertEqual(daily.loc[daily.day == 286, 'group_count'].item(), 0)
        self.assertTrue((self.output / 'comparison_sub.csv').exists())
        self.assertTrue((self.output / 'comparison_full.csv').exists())
        self.assertTrue((self.output / 'REPORT.md').exists())

    def test_changed_log_ids_fail_before_predict(self):
        wrong = {key: value.copy() for key, value in self.arrays.items()}
        wrong['log_id'][0] = 100
        np.savez_compressed(self.schema / 'test_targets.npz', **wrong)
        with patch.object(CatBoostRanker, 'predict', side_effect=AssertionError('not reached')):
            with self.assertRaises(AssertionError):
                evaluate_model(self.model, self.plan, self.data, self.output)

    def test_changed_borders_fail_before_predict(self):
        (self.schema / 'test_applied_borders.tsv').write_text('wrong borders')
        with self.assertRaisesRegex(ValueError, 'другие пороги'):
            evaluate_model(self.model, self.plan, self.data, self.output)

    def test_incomplete_artifacts_are_not_silently_overwritten(self):
        folder = self.output / 'models/M-sub'
        folder.mkdir(parents=True)
        (folder / 'request.json').write_text('{}')
        with self.assertRaises(FileExistsError):
            evaluate_model(self.model, self.plan, self.data, self.output)


if __name__ == '__main__':
    unittest.main()
