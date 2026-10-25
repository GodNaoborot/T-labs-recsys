"""C4: исходные ID, границы времени, независимый oracle и отсутствие R-канала."""
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
from temporal_dfs_features import build_temporal_definitions, calculate_temporal
from user_session_features import calculate_session_features, SESSION_FEATURES


class C4FeatureTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='c4-feature-test-')
        self.root = Path(self.folder.name)
        self.listens = pd.DataFrame({
            'uid': [1, 1, 1, 1, 1, 2, 2], 'item_id': [10, 10, 10, 11, 10, 10, 12],
            'artist_id': [3, 3, 3, 4, 3, 3, 5],
            'timestamp': [5, 95, 95, 100, 105, 50, 110],
            'played_ratio_pct': [100, 0, 80, 0, 0, 10, 100],
            'label': [1, 0, 1, 0, 0, 0, 2],
            'track_length_seconds': [100] * 7, 'is_organic': [1] * 7})
        self.events = pd.DataFrame({'uid': [1, 1, 1, 1, 2],
            'item_id': [10, 10, 10, 10, 10], 'timestamp': [10, 95, 95, 100, 100],
            'event_type': ['like', 'like', 'unlike', 'like', 'dislike']})
        self.names = json.loads(c4.OLD_CONTROLS.read_text())['manual_features'] + list(SESSION_FEATURES)
        self.write_sources()

    def tearDown(self):
        self.folder.cleanup()

    def write_sources(self):
        self.listens.to_parquet(self.root / 'listens.parquet', index=False, row_group_size=2)
        self.events.to_parquet(self.root / 'events.parquet', index=False, row_group_size=2)

    def history(self, limit=200):
        return c4.C4ManualHistory(limit, self.root / 'listens.parquet', self.root / 'events.parquet')

    def queries(self):
        return pd.DataFrame({'log_id': [1000, 1001, 1002, 1003],
            'uid': [1, 1, 2, 9], 'item_id': [10, 11, 10, 99], 'artist_id': [3, 4, 3, -1],
            'time': [100, 105, 100, 100], 'end_ts': [110, 110, 110, 110],
            'label': [0, 1, 2, 0], 'is_organic': [1] * 4,
            'track_length_seconds': [100] * 4, 'group_id': [1000, 1000, 2000, 9000]})

    def test_all_49_manual_features_match_independent_pandas_oracle(self):
        history, queries = self.history(), self.queries()
        actual = history.features(queries, self.names)
        expected, recencies = c4._oracle_manual(history, queries, self.names)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5, equal_nan=True)
        self.assertGreater(recencies, 0)
        values = dict(zip(self.names, actual[0]))
        self.assertEqual(values['user_prev_played_ratio'], 80)
        self.assertEqual(values['user_time_since_last'], 5)
        self.assertEqual(values['ui_liked_now'], 0)  # unlike is last by physical ID.
        self.assertEqual(values['ui_lst_amt_7'], 3)

    def test_current_and_future_R_label_do_not_change_query_features(self):
        queries = self.queries().iloc[:1].copy()
        baseline = self.history().features(queries, self.names)
        self.listens.loc[self.listens.timestamp >= 100, ['played_ratio_pct', 'label']] = [100, 2]
        self.events.loc[self.events.timestamp >= 100, 'event_type'] = 'unlike'
        self.write_sources()
        queries['label'] = 2
        queries['played_ratio_pct'] = 100
        queries['start_ts'] = 1  # This old field must never select the new cutoff.
        np.testing.assert_array_equal(self.history().features(queries, self.names), baseline)

    def test_source_row_ids_assigned_before_filter_and_test_cap_rejected(self):
        old = c4._read_before(self.root / 'listens.parquet', ['uid', 'timestamp'], 100, batch_size=2)
        np.testing.assert_array_equal(old.source_event_id, [0, 1, 2, 5])
        self.assertTrue((old.timestamp < 100).all())
        for limit in [c4.TEST_START, c4.TEST_START + 5]:
            with self.assertRaisesRegex(ValueError, 'TEST_START'):
                c4._read_before(self.root / 'listens.parquet', ['uid', 'timestamp'], limit)

    def test_fixed_source_ids_make_historical_ties_permutation_invariant(self):
        old = self.listens[['uid', 'timestamp', 'played_ratio_pct']].copy()
        old['source_event_id'] = np.arange(len(old))
        shuffled = old.sample(frac=1, random_state=42)
        query = pd.DataFrame({'uid': [1], 'start_ts': [100]})
        expected = calculate_session_features(query, c4._history_order(old))
        actual = calculate_session_features(query, c4._history_order(shuffled))
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.user_prev_played_ratio.iloc[0], 80)

    def test_cutoff_value_and_static_length_have_separate_roles(self):
        q = self.queries().iloc[:1].copy()
        baseline = self.history().features(q, self.names)
        q['track_length_seconds'] = 205
        changed = self.history().features(q, self.names)
        length = self.names.index('track_length_seconds')
        self.assertEqual(changed[0, length], 205)
        np.testing.assert_array_equal(np.delete(baseline, length, axis=1), np.delete(changed, length, axis=1))

    def test_dfs_oracle_all_supported_formulas_and_exact_float32_day_cast(self):
        definitions = build_temporal_definitions('Users', 'uid', windows=[1])
        history = self.listens.rename(columns={'timestamp': 'end_ts'}).copy()
        history['log_id'] = np.arange(len(history))
        queries = pd.DataFrame({'log_id': [1000, 1001, 1002], 'uid': [1, 1, 9],
                                'time': [100, 200 * c4.DAY + 5, 100]})
        actual = calculate_temporal(history, queries, 'Users', 'uid', definitions)
        report = c4.verify_dfs_probe(history, queries, 'uid', list(enumerate(definitions)), actual)
        self.assertEqual(report['values'], len(queries) * len(definitions))
        last = [i for i, feature in enumerate(definitions) if feature.primitive.name == 'time_since_last'][0]
        self.assertEqual(actual[0, last], np.float32(5 / c4.DAY))
        self.assertEqual(actual[1, last], np.float32((200 * c4.DAY + 5 - 105) / c4.DAY))
        self.assertTrue(np.isnan(actual[2, last]))

    def test_dfs_current_outcome_and_future_mutations_cannot_enter_history(self):
        definitions = build_temporal_definitions('Users', 'uid', windows=[1])
        history = self.listens.rename(columns={'timestamp': 'end_ts'}).copy()
        history['log_id'] = np.arange(len(history))
        queries = pd.DataFrame({'log_id': [1000], 'uid': [1], 'time': [100]})
        expected = calculate_temporal(history, queries, 'Users', 'uid', definitions)
        history.loc[history.end_ts >= 100, 'label'] = 2
        history.loc[history.end_ts >= 100, 'track_length_seconds'] = 995
        actual = calculate_temporal(history, queries, 'Users', 'uid', definitions)
        np.testing.assert_array_equal(actual, expected)

    def test_chunk_writers_keep_frozen_columns_fortran_layout_and_resume(self):
        from generate_cumulative import add_pair_keys
        from temporal_sample_data import _branch_keys
        queries = self.queries()
        add_pair_keys(queries)
        result = self.root / 'reports'
        result.mkdir()
        report = c4._manual_split(queries, self.history(), self.names,
                                 self.root / 'manual49', result, 'train', None)
        matrix = np.load(report['matrix_path'], mmap_mode='r')
        self.assertEqual(matrix.shape, (len(queries), 49))
        self.assertTrue(matrix.flags.f_contiguous)
        repeated = c4._manual_split(queries, self.history(), self.names,
                                   self.root / 'manual49', result, 'train', None)
        self.assertEqual(repeated, report)
        changed = queries.copy()
        changed.loc[0, 'time'] += 5
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            c4._manual_split(changed, self.history(), self.names,
                             self.root / 'manual49', result, 'train', None)
        names, definitions = c4.load_selected_definitions(c4.DEFAULT_SELECTION, c4.DEFAULT_DEFINITIONS)
        self.assertEqual(len(names), 462)
        def history_loader(requests, key):
            child = self.listens.rename(columns={'timestamp': 'end_ts'}).copy()
            child['log_id'] = np.arange(len(child))
            child[key] = _branch_keys(self.listens, key)
            return child.loc[(child[key] >= 0) & (child.end_ts < requests.time.max()),
                             ['log_id', key, 'end_ts', 'label', 'track_length_seconds', 'is_organic']]
        arrays = {name: queries[name].to_numpy() for name in ('label', 'group_id', 'log_id')}
        with patch.object(c4, 'load_branch_history', side_effect=history_loader):
            dfs = c4._dfs_split(queries, arrays, names, definitions,
                               self.root / 'dfs462', result, 'train', None)
        matrix = np.load(dfs['matrix_path'], mmap_mode='r')
        self.assertEqual(matrix.shape, (len(queries), 462))
        self.assertTrue(matrix.flags.f_contiguous)
        self.assertEqual(set(dfs['branches']), {'Users', 'Items', 'UserItem', 'UserArtist'})
        self.assertTrue(all(record['oracle']['rows'] == len(queries)
                            for record in dfs['branches'].values()))

    def test_missing_or_invalid_historical_label_validation_blocks_preparation(self):
        (self.root / 'history_label_availability.json').write_text(json.dumps({
            'completed': True, 'frozen_labels_match': True,
            'all_attached_reactions_at_or_before_listen_end': False, 'test_used': False}))
        with self.assertRaisesRegex(ValueError, 'меток'):
            c4.prepare_features(self.root, self.root)


if __name__ == '__main__':
    unittest.main()
