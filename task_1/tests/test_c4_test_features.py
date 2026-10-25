"""Маленькие проверки повторного test-адаптера; реальный test не читается."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import c4_features as c4
import c4_test_features as module
from generate_cumulative import add_pair_keys
from project_artifacts import file_hash
from temporal_sample_data import _branch_keys
from user_session_features import SESSION_FEATURES


class C4TestFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='c4-test-feature-unit-')
        self.root = Path(self.temporary.name)
        (self.root / 'data').mkdir()
        self.start = c4.TEST_START
        self.listens = pd.DataFrame({
            'uid': [1, 1, 1, 1, 1, 2, 2], 'item_id': [10, 10, 10, 11, 10, 10, 12],
            'artist_id': [3, 3, 3, 4, 3, 3, 5],
            'timestamp': self.start + np.array([-5, 95, 95, 100, 105, 50, 110]),
            'played_ratio_pct': [100, 0, 80, 0, 0, 10, 100],
            'label': [1, 0, 1, 0, 0, 0, 2],
            'track_length_seconds': [100] * 7, 'is_organic': [1] * 7})
        self.events = pd.DataFrame({'uid': [1, 1, 1, 1, 2],
            'item_id': [10, 10, 10, 10, 10],
            'timestamp': self.start + np.array([10, 95, 95, 100, 100]),
            'event_type': ['like', 'like', 'unlike', 'like', 'dislike']})
        self.names = json.loads(c4.OLD_CONTROLS.read_text())['manual_features'] + list(SESSION_FEATURES)
        self.write_sources()

    def tearDown(self):
        self.temporary.cleanup()

    def write_sources(self):
        self.listens.to_parquet(self.root / 'data/listens.parquet', index=False, row_group_size=2)
        self.events.to_parquet(self.root / 'data/events.parquet', index=False, row_group_size=2)

    def history(self, limit=None):
        return module.C4TestManualHistory(limit or self.start + 200,
            self.root / 'data/listens.parquet', self.root / 'data/events.parquet')

    def queries(self):
        result = pd.DataFrame({'log_id': [1000, 1001, 1002, 1003], 'uid': [1, 1, 2, 2],
            'item_id': [10, 11, 10, 12], 'artist_id': [3, 4, 3, 5],
            'time': self.start + np.array([100, 105, 100, 100]),
            'end_ts': [self.start + 110] * 4, 'label': [0, 1, 0, 2],
            'is_organic': [1] * 4, 'track_length_seconds': [100] * 4})
        result['day'] = result.time // c4.DAY
        result['group_id'] = result.uid * 1000 + result.day
        result['t'] = result['start_ts'] = result.time
        add_pair_keys(result)
        return result

    def targets(self, queries):
        return {name: queries[name].to_numpy(dtype='int8' if name == 'label' else 'int64')
                for name in ['label', 'group_id', 'log_id', 'day']}

    def plan(self):
        dfs_names, _ = c4.load_selected_definitions(c4.DEFAULT_SELECTION, c4.DEFAULT_DEFINITIONS)
        return {'schemas': {'manual49': self.names, 'dfs462': dfs_names},
                'selection': {'path': str(c4.DEFAULT_SELECTION), 'sha256': file_hash(c4.DEFAULT_SELECTION)},
                'definitions': {parent: {'path': str(c4.DEFAULT_DEFINITIONS / f'{parent}.json'),
                    'sha256': file_hash(c4.DEFAULT_DEFINITIONS / f'{parent}.json')}
                    for parent, _ in c4.BRANCHES},
                'sources': {name: {'path': str(self.root / 'data' / f'{name}.parquet')}
                            for name in ['listens', 'events']}}

    def test_extended_reader_keeps_source_ids_before_time_filter(self):
        frame = module._read_test_history(self.root / 'data/listens.parquet',
            ['uid', 'timestamp'], self.start + 100, batch_size=2)
        np.testing.assert_array_equal(frame.source_event_id, [0, 1, 2, 5])
        self.assertTrue((frame.timestamp < self.start + 100).all())
        self.assertTrue((frame.timestamp >= self.start).any())
        with self.assertRaisesRegex(ValueError, '285'):
            module._read_test_history(self.root / 'data/listens.parquet', ['timestamp'], self.start - 5)

    def test_test_history_all_49_features_match_independent_oracle(self):
        history, query = self.history(), self.queries()
        actual = history.features(query, self.names)
        expected, recencies = c4._oracle_manual(history, query, self.names)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5, equal_nan=True)
        self.assertGreater(recencies, 0)
        self.assertEqual(actual[0, self.names.index('user_prev_played_ratio')], 80)
        self.assertEqual(actual[0, self.names.index('user_time_since_last')], 5)
        self.assertEqual(actual[0, self.names.index('ui_liked_now')], 0)

    def test_test_current_and_future_outcome_cannot_change_features(self):
        query = self.queries().iloc[:1].copy()
        expected = self.history().features(query, self.names)
        self.listens.loc[self.listens.timestamp >= self.start + 100,
                         ['played_ratio_pct', 'label']] = [100, 2]
        self.events.loc[self.events.timestamp >= self.start + 100, 'event_type'] = 'unlike'
        self.write_sources()
        query['played_ratio_pct'] = 100
        query['label'] = 2
        query['start_ts'] = -100  # features() always uses the canonical time cutoff.
        np.testing.assert_array_equal(self.history().features(query, self.names), expected)

    def test_test_query_loader_rejects_old_start_alias_or_non_test_cutoff(self):
        query = self.queries()
        query.to_parquet(self.root / 'test_queries.parquet', index=False)
        np.savez(self.root / 'test_targets.npz', **self.targets(query))
        loaded, targets = module._load_test_queries(self.root)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(len(targets['day']), 4)
        query.loc[0, 'start_ts'] -= 1
        query.to_parquet(self.root / 'test_queries.parquet', index=False)
        with self.assertRaises(AssertionError):
            module._load_test_queries(self.root)
        query.loc[0, 'time'] = self.start - 5
        query.to_parquet(self.root / 'test_queries.parquet', index=False)
        with self.assertRaisesRegex(ValueError, 'Cutoff'):
            module._load_test_queries(self.root)

    def test_frozen_formula_hash_mismatch_blocks_load(self):
        plan = self.plan()
        plan['definitions']['Users']['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Users'):
            module._feature_spec(plan)

    def test_full_tiny_adapter_all_462_oracles_and_shared_order(self):
        query = self.queries()
        query.to_parquet(self.root / 'test_queries.parquet', index=False)
        np.savez(self.root / 'test_targets.npz', **self.targets(query))
        reports = self.root / 'reports'
        reports.mkdir()
        (reports / 'history_label_availability.json').write_text(json.dumps({
            'completed': True, 'test_used': True, 'frozen_labels_match': True,
            'all_attached_reactions_at_or_before_listen_end': True}))
        def loader(requests, key):
            history = self.listens.rename(columns={'timestamp': 'end_ts'}).copy()
            history['log_id'] = np.arange(len(history))
            history[key] = _branch_keys(self.listens, key)
            return history.loc[(history[key] >= 0) & (history.end_ts < requests.time.max()),
                    ['log_id', key, 'end_ts', 'label', 'track_length_seconds', 'is_organic']]
        with patch.object(module, 'TASK_ROOT', self.root), patch.object(c4, 'load_branch_history', side_effect=loader):
            report = module.prepare_test_features(self.root, reports, self.plan())
        self.assertTrue(report['completed'])
        self.assertTrue(report['test_used'])
        self.assertFalse(report['fitted'])
        for kind, size in [('manual', 49), ('dfs', 462)]:
            record = report[kind]
            matrix = np.load(record['matrix_path'], mmap_mode='r')
            self.assertEqual(matrix.shape, (4, size))
            self.assertTrue(matrix.flags.f_contiguous)
            self.assertTrue(record['test_used'])
            self.assertEqual(record['target_arrays'], report['target_arrays'])
        self.assertEqual(sum(branch['oracle']['values'] for branch in report['dfs']['branches'].values()), 4 * sum('__' in name for name in self.plan()['schemas']['dfs462']))
        self.assertTrue(all(branch['oracle']['all_selected_formulas_checked']
                            for branch in report['dfs']['branches'].values()))
        self.assertEqual(report['manual']['oracle']['values'], 4 * 49)
        with patch.object(module, 'TASK_ROOT', self.root), patch.object(c4, 'load_branch_history', side_effect=loader):
            repeated = module.prepare_test_features(self.root, reports, self.plan())
        self.assertEqual(repeated['dfs']['matrix_sha256'], report['dfs']['matrix_sha256'])
        self.assertEqual(repeated['manual']['matrix_sha256'], report['manual']['matrix_sha256'])


if __name__ == '__main__':
    unittest.main()
