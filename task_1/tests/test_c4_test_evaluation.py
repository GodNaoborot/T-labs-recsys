"""Маленькие проверки C4 test: fixed borders, predict-only и парная статистика."""
from pathlib import Path
import ast
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
import c4_test_evaluation as c4
from block_bootstrap import aggregate_days, circular_day_indices
from compare_temporal_controls_pairs import absolute_delta_draws


class FrozenC4Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.data, self.result = self.folder / 'data', self.folder / 'result'
        self.data.mkdir()
        self.result.mkdir()
        days = np.repeat([285, 287, 288, 289, 290], 3).astype('int64')
        group = np.repeat(np.arange(5) * 1000, 3) + days
        labels = np.tile(np.array([0, 1, 2], dtype='int8'), 5)
        self.arrays = {'label': labels, 'group_id': group, 'day': days,
                       'log_id': np.arange(15, dtype='int64')}
        np.savez_compressed(self.data / 'targets.npz', **self.arrays)
        schemas = {'manual49': ['manual_' + str(i) for i in range(9)],
                   'dfs462': ['dfs_' + str(i) for i in range(3)]}
        self.plan = {'test_evaluation_authorized': True, 'new_training_allowed': False,
                     'test_already_open': True, 'schemas': schemas, 'models': [],
                     'train_borders': {}, 'comparisons': [list(pair) for pair in c4.PAIRS],
                     'bootstrap': {'block_days': [3, 4], 'replicates': 20000,
                                   'seed': 42, 'confidence': .95},
                     'limitations': ['synthetic fixture only']}
        self.raw = {}
        for schema, names in schemas.items():
            folder = self.data / schema
            folder.mkdir()
            rng = np.random.default_rng(42)
            train = rng.normal(size=(15, len(names))).astype('float32')
            train[:, 0] = labels
            train[:, 1] = 7
            test = train.copy()
            test[:, 0] += .2
            test[-1, 0] = 10000  # Значение вне диапазона train не создаёт новый порог.
            test[:, 1] = np.arange(15)  # Train-константа меняется только на test.
            np.save(folder / 'test_X.npy', np.asfortranarray(test))
            self.raw[schema] = test
            pool = Pool(train, label=labels, group_id=group, feature_names=names)
            pool.quantize(border_count=4, random_seed=42, nan_mode='Min')
            borders = self.folder / (schema + '_train_borders.tsv')
            pool.save_quantization_borders(str(borders))
            self.plan['train_borders'][schema] = {
                'path': str(borders), 'sha256': c4.file_hash(borders)}
            variants = ['M-t', 'C2b-t'] if schema == 'manual49' else ['D462-t']
            for variant in variants:
                ignored = list(range(2, 9)) if variant == 'M-t' else []
                model = CatBoostRanker(iterations=4, depth=2, loss_function='YetiRank',
                                      random_seed=42, thread_count=1, verbose=False,
                                      allow_writing_files=False, ignored_features=ignored)
                # Обучается только этот крошечный искусственный fixture, до проверяемого API.
                model.fit(pool)
                path = self.folder / (variant + '.cbm')
                model.save_model(str(path))
                self.plan['models'].append({'variant': variant, 'schema': schema,
                    'active_features': len(names) - len(ignored), 'tree_count': 4,
                    'model': {'path': str(path), 'sha256': c4.file_hash(path)}})

    def tearDown(self):
        self.temporary.cleanup()

    def test_native_frozen_borders_constant_drift_and_predict_only(self):
        hashes = [record['model']['sha256'] for record in self.plan['models']]
        with patch.object(CatBoostRanker, 'fit', side_effect=AssertionError('fit forbidden')):
            report = c4.quantize_test(self.data, self.result, self.plan)
            repeated = c4.quantize_test(self.data, self.result, self.plan)
            self.assertEqual(report, repeated)
            result = c4.evaluate_test(self.data, self.result, self.plan)
            again = c4.evaluate_test(self.data, self.result, self.plan)
        self.assertEqual(result, again)
        self.assertTrue(result['completed'])
        self.assertFalse(result['fit_executed'])
        self.assertFalse(result['new_borders_learned'])
        self.assertFalse(result['new_blind_test'])
        self.assertEqual((result['rows'], result['groups'], result['calendar_days']), (15, 5, 6))
        self.assertEqual(len(result['comparisons']), 6)
        self.assertEqual(hashes, [c4.file_hash(r['model']['path']) for r in self.plan['models']])
        for schema in self.plan['schemas']:
            folder = self.result / 'quantized' / schema
            frozen = Path(self.plan['train_borders'][schema]['path']).read_bytes()
            self.assertEqual((folder / 'test_applied_borders.tsv').read_bytes(), frozen)
            self.assertIn(1, report['schemas'][schema]['training_constant_columns'])
            self.assertFalse((folder / 'train.qbin').exists())
            self.assertFalse((folder / 'val.qbin').exists())
            self.assertFalse(any(line.startswith('1\t') for line in frozen.decode().splitlines()))
        for record in self.plan['models']:
            model = CatBoostRanker().load_model(record['model']['path'])
            raw_pool = Pool(self.raw[record['schema']], feature_names=self.plan['schemas'][record['schema']])
            expected = model.predict(raw_pool, thread_count=1)
            actual = np.load(self.result / 'models' / record['variant'] / 'test_score.npy')
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        daily = pd.read_csv(self.result / 'models/M-t/daily_ndcg.csv')
        self.assertEqual(daily.loc[daily.day == 286, 'group_count'].item(), 0)
        reference = pd.read_parquet(self.result / 'models/M-t/group_ndcg.parquet')
        candidate = pd.read_parquet(self.result / 'models/D462-t/group_ndcg.parquet')
        daily_pair = aggregate_days(reference.day, reference.ndcg, candidate.ndcg)
        indices = circular_day_indices(6, 3, 20000, seed=42)
        expected = np.quantile(absolute_delta_draws(daily_pair, indices), [.025, .975], method='linear')
        compared = next(row for row in result['comparisons']
                        if row['candidate'] == 'D462-t' and row['reference'] == 'M-t'
                        and row['block_days'] == 3)
        np.testing.assert_allclose([compared['ci_low'], compared['ci_high']], expected, rtol=0, atol=0)
        self.assertTrue((self.result / 'test_result.json').exists())
        self.assertFalse((self.result / 'result.json').exists())

    def test_changed_targets_and_borders_fail_before_predict(self):
        c4.quantize_test(self.data, self.result, self.plan)
        folder = self.result / 'quantized/manual49'
        original = (folder / 'test_applied_borders.tsv').read_bytes()
        (folder / 'test_applied_borders.tsv').write_text('wrong borders')
        with patch.object(CatBoostRanker, 'predict', side_effect=AssertionError('not reached')):
            with self.assertRaisesRegex(ValueError, 'другие пороги'):
                c4.evaluate_test(self.data, self.result, self.plan)
        (folder / 'test_applied_borders.tsv').write_bytes(original)
        changed = {name: values.copy() for name, values in self.arrays.items()}
        changed['log_id'][0] = 999
        np.savez_compressed(folder / 'test_targets.npz', **changed)
        with patch.object(CatBoostRanker, 'predict', side_effect=AssertionError('not reached')):
            with self.assertRaises(AssertionError):
                c4.evaluate_test(self.data, self.result, self.plan)

    def test_plan_changes_fail_before_opening_matrices(self):
        for field, value in [('test_evaluation_authorized', False), ('new_training_allowed', True)]:
            changed = copy.deepcopy(self.plan)
            changed[field] = value
            with self.assertRaises(ValueError):
                c4.quantize_test(self.folder / 'does_not_exist', self.result, changed)
        changed = copy.deepcopy(self.plan)
        changed['comparisons'].pop()
        with self.assertRaises(ValueError):
            c4.evaluate_test(self.data, self.result, changed)

    def test_production_source_has_no_fit_or_training_quantization_call(self):
        tree = ast.parse(Path(c4.__file__).read_text())
        called = [node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn('fit', called)
        self.assertNotIn('fit_transform', called)
        self.assertNotIn('quantize', called)


if __name__ == '__main__':
    unittest.main()
