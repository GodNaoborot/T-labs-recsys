"""Малые проверки выборки и истории; большой датасет не читается."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import temporal_sample_data as data


class TemporalSampleDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='temporal-data-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        setting = patch.object(data, 'ROOT', self.root)
        setting.start()
        self.addCleanup(setting.stop)
        self.metadata = self._make_queries()
        self._save_sources()
        self._save_history()

    def _make_queries(self):
        groups = np.array([2001, 2001, 1002, 1002, 1001, 1001], dtype='int64')
        uid = groups // 1000
        frame = pd.DataFrame({'log_id': np.arange(6, dtype='int64'),
                              'uid': uid, 'item_id': np.full(6, 10, dtype='int64'),
                              'artist_id': np.array([5, -1, 5, 5, 5, 5]),
                              'time': groups % 1000 * data.DAY + 10,
                              'label': np.array([0, 2, 0, 1, 1, 2], dtype='int8'),
                              'track_length_seconds': np.full(6, 100, dtype='uint16'),
                              'is_organic': np.ones(6, dtype='uint8'),
                              'group_id': groups})
        frame['end_ts'] = frame['time'] + 7
        frame['user_item_id'] = (uid << 32) | 10
        artist = frame['artist_id'].to_numpy()
        keys = pd.array((uid << 32) | np.maximum(artist, 0), dtype='Int64')
        keys[artist < 0] = pd.NA
        frame['user_artist_id'] = keys
        return frame[data.QUERY_COLUMNS]

    def _save_sources(self):
        ordered = self.metadata.sort_values('group_id', kind='stable')
        for split in ['train', 'val']:
            targets, metadata = data._source_paths(split)
            targets.parent.mkdir(parents=True, exist_ok=True)
            metadata.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(targets, **{n: ordered[n].to_numpy()
                                           for n in ['label', 'group_id', 'log_id']})
            self.metadata.to_parquet(metadata, index=False, row_group_size=2)

    def _save_history(self):
        self.history = pd.DataFrame({
            'uid': [1, 2, 1, 1, 3, 1, 2, 1],
            'item_id': [10, 10, 11, 10, 10, 12, 11, 10],
            'artist_id': [5, 5, 6, 5, 5, -1, 6, 5],
            'timestamp': [5, 10, 11, 20, 21, 8, 12, 19],
            'label': np.array([0, 2, 1, 0, 1, 0, 1, 2], dtype='int8'),
            'track_length_seconds': np.full(8, 100, dtype='uint16'),
            'is_organic': np.ones(8, dtype='uint8')})
        path = self.root / 'data/listens.parquet'
        path.parent.mkdir(exist_ok=True)
        self.history.to_parquet(path, index=False, row_group_size=2)

    def test_group_sample_is_whole_deterministic_and_in_original_order(self):
        groups = np.array([1, 1, 2, 2, 2, 3, 3, 4], dtype='int64')
        positions = data._whole_group_positions(groups, 5, 42)
        np.testing.assert_array_equal(positions, data._whole_group_positions(groups, 5, 42))
        self.assertLessEqual(len(positions), 5)
        self.assertTrue(np.all(np.diff(positions) > 0))
        for group in np.unique(groups[positions]):
            self.assertEqual(np.sum(groups[positions] == group), np.sum(groups == group))

    def test_sample_aligns_metadata_to_canonical_group_order(self):
        queries, targets, provenance = data.choose_sample('train', 6, 42)
        self.assertEqual(queries['log_id'].tolist(), [4, 5, 2, 3, 0, 1])
        np.testing.assert_array_equal(queries['log_id'], targets['log_id'])
        np.testing.assert_array_equal(targets['day'], [1, 1, 2, 2, 1, 1])
        self.assertEqual(provenance['rows'], 6)
        self.assertEqual(provenance['groups'], 3)
        self.assertFalse(provenance['test_used'])
        self.assertEqual(len(provenance['sources']['metadata']['sha256']), 64)

    def test_small_sample_keeps_full_ranking_groups(self):
        queries, targets, _ = data.choose_sample('train', 4, 7)
        self.assertEqual(len(queries), 4)
        self.assertEqual(len(np.unique(targets['group_id'])), 2)
        self.assertTrue(all(queries.groupby('group_id').size() == 2))
        with self.assertRaisesRegex(ValueError, 'ни одна целая группа'):
            data.choose_sample('train', 1, 42)

    def test_metadata_label_mismatch_is_rejected(self):
        _, path = data._source_paths('train')
        changed = self.metadata.copy()
        changed.loc[0, 'label'] = 2
        changed.to_parquet(path, index=False)
        with self.assertRaises(AssertionError):
            data.choose_sample('train', 6, 42)

    def test_item_history_includes_other_users_and_keeps_global_log_id(self):
        queries = pd.DataFrame({'item_id': [10], 'time': [20]})
        history = data.load_branch_history(queries, 'item_id', batch_size=2)
        self.assertEqual(history['log_id'].tolist(), [0, 1, 7])
        self.assertEqual(history['label'].tolist(), [0, 2, 2])
        self.assertTrue((history['end_ts'] < 20).all())

    def test_user_and_pair_histories_have_correct_keys(self):
        query = pd.DataFrame({'uid': [1], 'user_item_id': [(1 << 32) | 10],
                              'user_artist_id': pd.array([(1 << 32) | 5], dtype='Int64'),
                              'time': [20]})
        expected = {'uid': [0, 2, 5, 7], 'user_item_id': [0, 7], 'user_artist_id': [0, 7]}
        for key, log_ids in expected.items():
            with self.subTest(key=key):
                result = data.load_branch_history(query, key, batch_size=2)
                self.assertEqual(result['log_id'].tolist(), log_ids)
                self.assertTrue((result[key] == query.loc[0, key]).all())

    def test_missing_artist_has_empty_history(self):
        queries = pd.DataFrame({'user_artist_id': pd.array([pd.NA], dtype='Int64'), 'time': [20]})
        self.assertTrue(data.load_branch_history(queries, 'user_artist_id').empty)

    def test_test_split_is_never_allowed(self):
        with self.assertRaisesRegex(ValueError, 'train и val'):
            data.choose_sample('test', 10, 42)

    def test_invalid_group_order_and_branch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'возрастанию'):
            data._whole_group_positions(np.array([2, 1]), 2, 42)
        with self.assertRaisesRegex(ValueError, 'Неизвестная ветвь'):
            data.load_branch_history(pd.DataFrame(), 'wrong')


if __name__ == '__main__':
    unittest.main()
