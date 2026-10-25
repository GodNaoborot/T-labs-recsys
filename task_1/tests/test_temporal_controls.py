"""Малые проверки общих строк, масок контролей, C2-oracle и парного bootstrap."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import temporal_controls_data as controls
import run_temporal_controls as runner
from user_session_features import SESSION_FEATURES


class TemporalControlsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='temporal-controls-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.result, self.data = self.root / 'results', self.root / 'data'
        self.result.mkdir()
        self.data.mkdir()
        for module in (controls, runner):
            setting = patch.object(module, 'run_paths', return_value=(self.result, self.data))
            setting.start()
            self.addCleanup(setting.stop)
        self.manual = [f'manual_{i}' for i in range(42)]
        self.dfs = ([f'Users__feature_{i}' for i in range(132)] +
                    [f'Items__feature_{i}' for i in range(148)] +
                    [f'UserItem__feature_{i}' for i in range(59)] +
                    [f'UserArtist__feature_{i}' for i in range(123)])
        controls.write_json(self.result / 'protocol.json', {
            'manual_features': self.manual, 'dfs_features': self.dfs,
            'session_features': list(SESSION_FEATURES), 'train_groups': 2, 'seed': 42})
        controls.write_json(self.result / 'dfs_preparation.json', {
            'train_pool': 'dfs_train.qbin', 'val_pool': 'dfs_val.qbin'})
        for split in ('train', 'val'):
            controls.write_json(self.data / f'manual_quantized/{split}_report.json',
                                {'pool_path': f'manual_{split}.qbin'})

    def test_five_variant_masks_have_exact_active_features_and_keep_order(self):
        expected = {'M-sub': 42, 'D462-sub': 462, 'C1': 330, 'C2a': 48, 'C2b': 49}
        for variant, count in expected.items():
            with self.subTest(variant=variant):
                spec = controls.variant_spec('unit', variant)
                self.assertEqual(spec['active_count'], count)
                self.assertEqual(len(spec['active_features']), count)
                self.assertEqual(spec['active_features'], [name for i, name in enumerate(spec['features'])
                                                          if i not in spec['ignored_indices']])
                self.assertEqual(len(set(spec['ignored_indices'])), len(spec['ignored_indices']))
        self.assertEqual(controls.variant_spec('unit', 'M-sub')['active_features'], self.manual)
        self.assertEqual(controls.variant_spec('unit', 'D462-sub')['active_features'], self.dfs)
        self.assertEqual(controls.variant_spec('unit', 'C1')['active_features'], self.dfs[132:])
        self.assertEqual(controls.variant_spec('unit', 'C2a')['active_features'], self.manual + list(SESSION_FEATURES[:-1]))
        self.assertEqual(controls.variant_spec('unit', 'C2b')['active_features'], self.manual + list(SESSION_FEATURES))
        self.assertEqual(controls.variant_spec('unit', 'C2a')['ignored_indices'], [48])
        with self.assertRaisesRegex(ValueError, 'Неизвестный'):
            controls.variant_spec('unit', 'C4')

    def test_paired_interval_weights_groups_on_days_with_different_group_counts(self):
        # День 0: одна группа с приростом .6; день 1: девять групп без прироста.
        daily = pd.DataFrame({'group_count': [1, 9], 'base_sum': [.2, 1.8],
                              'dfs_sum': [.8, 1.8]})
        indices = np.array([[0, 1], [1, 1], [0, 0], [0, 1]])
        actual = runner.paired_delta_interval(daily, indices)
        expected = np.quantile([.06, 0, .6, .06], [.025, .975])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        unweighted = np.quantile([.3, 0, .6, .3], [.025, .975])
        self.assertFalse(np.allclose(actual, unweighted))
        repeated = np.array([[0, 1], [0, 1]])
        np.testing.assert_allclose(runner.paired_delta_interval(daily, repeated), [.06, .06])

    def _targets(self, groups):
        groups = np.asarray(groups, dtype='int64')
        return {'group_id': groups, 'label': np.arange(len(groups), dtype='int8') % 3,
                'log_id': np.arange(len(groups), dtype='int64') + 100,
                'day': groups % 1000}

    def test_selection_keeps_whole_train_groups_all_val_and_canonical_order(self):
        sources = {'train': self._targets([1001, 1001, 1002, 1002, 1002, 2003, 2003, 3004, 3004, 3004, 3004]),
                   'val': self._targets([1005, 1005, 1005, 2006, 2006])}
        with patch.object(controls, '_load_canonical_targets', side_effect=lambda split: sources[split]) as loader:
            report = controls.prepare_selection('unit')
            train, positions = controls.load_selection('unit', 'train')
            self.assertEqual(len(np.unique(train['group_id'])), 2)
            self.assertTrue(np.all(np.diff(positions) > 0))
            for group in np.unique(train['group_id']):
                self.assertEqual(np.sum(train['group_id'] == group), np.sum(sources['train']['group_id'] == group))
            for field, values in train.items():
                np.testing.assert_array_equal(values, sources['train'][field][positions])
            val, val_positions = controls.load_selection('unit', 'val')
            np.testing.assert_array_equal(val_positions, np.arange(len(sources['val']['label'])))
            for field, values in val.items():
                np.testing.assert_array_equal(values, sources['val'][field])
            self.assertEqual(controls.prepare_selection('unit'), report)
            self.assertEqual(loader.call_count, 2)
        path = self.data / 'selection/train.npz'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Selection'):
            controls.prepare_selection('unit')

    def _session_fixture(self):
        times = list(range(1, 19)) + [90000, 91000, 92000, 93000, 94000, 95000, 96000, 99990]
        ratios = [0] * 9 + [100] * 9 + [100, 100, 100, 0, 0, 0, 0, 70]
        history = pd.DataFrame({'uid': [1] * len(times), 'timestamp': times, 'played_ratio_pct': ratios})
        queries = pd.DataFrame({'uid': [1, 2], 'start_ts': [100000, 100000]})
        expected = np.array([[70, 0, .8, .35, .5, 8, 10],
                             [np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan]], dtype='float32')
        history_path = self.root / 'data/listens.parquet'
        history.to_parquet(history_path, index=False, row_group_size=7)
        return queries, history_path, expected

    def test_session_oracle_checks_the_exact_seven_column_order(self):
        queries, history_path, expected = self._session_fixture()
        report = controls._verify_session_probe(queries, expected, history_path)
        self.assertEqual(report, {'rows': 2, 'values': 14, 'exact_float32': True})
        reordered = expected[:, [6, 1, 2, 3, 4, 5, 0]]
        with self.assertRaises(AssertionError):
            controls._verify_session_probe(queries, reordered, history_path)

    def test_manual_session_preparation_appends_api_columns_to_the_same_rows(self):
        queries, history_path, expected = self._session_fixture()
        metadata = queries.rename(columns={'start_ts': 'time'})
        arrays = self._targets([1001, 2001])
        def fill_manual(matrix, names, split, frame, destination):
            self.assertEqual(names, self.manual)
            matrix[:] = np.arange(42, dtype='float32')[None, :] + np.arange(2)[:, None] * 100
        with patch.object(controls, 'ROOT', self.root), \
             patch.object(controls, 'load_selection', return_value=(arrays, np.arange(2))), \
             patch.object(controls, '_manual_metadata', return_value=(pd.DataFrame(), np.arange(2))), \
             patch.object(controls, '_fill_manual_matrix', side_effect=fill_manual), \
             patch.object(controls, '_source_paths', return_value=(self.root / 'targets.npz', self.root / 'metadata.parquet')), \
             patch.object(controls, '_read_full_metadata', return_value=metadata), \
             patch.object(controls, '_validate_queries') as validate:
            report = controls.prepare_manual_sessions('unit', 'train')
        matrix = np.load(report['matrix_path'], mmap_mode='r')
        self.assertEqual(matrix.shape, (2, 49))
        self.assertTrue(matrix.flags.f_contiguous)
        np.testing.assert_array_equal(matrix[:, :42], np.arange(42)[None, :] + np.arange(2)[:, None] * 100)
        np.testing.assert_array_equal(matrix[:, 42:], expected)
        self.assertEqual(report['features'], self.manual + list(SESSION_FEATURES))
        self.assertEqual(report['independent_oracle']['values'], 14)
        self.assertEqual(report['session']['history_cutoff'], 'history.timestamp < query.start_ts')
        validate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
