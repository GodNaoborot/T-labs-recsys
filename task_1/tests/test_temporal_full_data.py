"""Проверки полного батчевого расчета на маленьких связанных таблицах."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import featuretools as ft
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import temporal_full_data as full
import temporal_sample_data as sample
from temporal_dfs_features import build_temporal_definitions, calculate_temporal


class FullTemporalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='full-temporal-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in [full, sample]:
            setting = patch.object(module, 'ROOT', self.root)
            setting.start()
            self.addCleanup(setting.stop)
        self.queries = self.make_queries()
        self.save_sources()
        self.selection_path, self.definition_dir = self.save_definitions()

    def make_queries(self):
        groups = np.array([2001, 2001, 1002, 1002, 1001, 1001], dtype='int64')
        uid = groups // 1000
        q = pd.DataFrame({'log_id': np.arange(6, dtype='int64'), 'uid': uid,
            'item_id': [10, 11, 10, 12, 10, 10], 'artist_id': [5, -1, 5, 6, 5, 5],
            'time': groups % 1000 * 86400 + np.arange(6) + 20,
            'label': np.array([0, 2, 0, 1, 1, 2], dtype='int8'),
            'track_length_seconds': np.full(6, 100, dtype='uint16'),
            'is_organic': np.ones(6, dtype='uint8'), 'group_id': groups})
        q['end_ts'] = q['time'] + 7
        q['user_item_id'] = (uid << 32) | q['item_id'].to_numpy()
        artist = q['artist_id'].to_numpy()
        keys = pd.array((uid << 32) | np.maximum(artist, 0), dtype='Int64')
        keys[artist < 0] = pd.NA
        q['user_artist_id'] = keys
        return q[full.METADATA_COLUMNS]

    def save_sources(self):
        ordered = self.queries.sort_values('group_id', kind='stable')
        for split in ['train', 'val']:
            targets, metadata = sample._source_paths(split)
            self.assertTrue(targets.resolve().is_relative_to(self.root.resolve()))
            self.assertTrue(metadata.resolve().is_relative_to(self.root.resolve()))
            targets.parent.mkdir(parents=True, exist_ok=True)
            metadata.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(targets, **{name: ordered[name].to_numpy()
                for name in ['label', 'group_id', 'log_id']})
            self.queries.to_parquet(metadata, index=False, row_group_size=2)
        self.history = pd.DataFrame({'uid': [1, 2, 1, 1, 3, 1, 2, 1],
            'item_id': [10, 10, 11, 10, 10, 12, 11, 10],
            'artist_id': [5, 5, 6, 5, 5, -1, 6, 5],
            'timestamp': [5, 10, 11, 20, 21, 8, 12, 19],
            'label': np.array([0, 2, 1, 0, 1, 0, 1, 2], dtype='int8'),
            'track_length_seconds': np.full(8, 100, dtype='uint16'),
            'is_organic': np.ones(8, dtype='uint8')})
        self.history.to_parquet(self.root / 'data/listens.parquet', index=False, row_group_size=2)

    def save_definitions(self):
        definition_dir = self.root / 'definitions'
        definition_dir.mkdir()
        names = []
        for parent, key in full.BRANCHES:
            definitions = build_temporal_definitions(parent, key, windows=[1])
            ft.save_features(definitions, str(definition_dir / f'{parent}.json'))
            wanted = [f for f in definitions if f.get_name() in [
                'COUNT(Interactions)', 'MEAN(Interactions.label)',
                'COUNT(Interactions, Last 1 days)', 'TIME_SINCE_LAST(Interactions.end_ts, unit=days)']]
            self.assertGreaterEqual(len(wanted), 2)
            names.extend(parent + '__' + f.get_name() for f in wanted)
        selection_path = self.root / 'selection.json'
        selection_path.write_text(json.dumps({'features': names}))
        return selection_path, definition_dir

    def test_metadata_matches_canonical_order_and_rejects_bad_labels(self):
        queries, targets = full.load_full_queries('train', batch_size=2)
        self.assertEqual(queries['log_id'].tolist(), [4, 5, 2, 3, 0, 1])
        np.testing.assert_array_equal(queries['log_id'], targets['log_id'])
        _, path = sample._source_paths('train')
        changed = self.queries.copy()
        changed.loc[0, 'label'] = 2
        changed.to_parquet(path, index=False)
        with self.assertRaises(AssertionError):
            full.load_full_queries('train', batch_size=2)

    def test_entity_blocks_never_split_one_entity(self):
        q = np.array([1, 1, 2, 3, 3, 4, 5])
        h = np.array([1] * 10 + [2, 2, 3, 4, 4])
        blocks = list(full.entity_blocks(q, h, 2, 3))
        self.assertEqual(blocks[0], (0, 2, 0, 10))
        self.assertEqual(blocks[-1][1], len(q))
        for i, (left, right, hleft, hright) in enumerate(blocks):
            if right < len(q):
                self.assertNotEqual(q[right - 1], q[right])
            if i:
                self.assertEqual(blocks[i - 1][1], left)

    def test_batched_full_matrix_matches_direct_and_resumes(self):
        out = self.root / 'output'
        report = full.generate_full_split('train', out, selection_path=self.selection_path,
            definition_dir=self.definition_dir, history_chunk=2, query_chunk=2)
        self.assertTrue(report['completed'])
        self.assertEqual(report['feature_count'], 12)
        matrix = np.load(report['matrix_path'])
        self.assertTrue(matrix.flags.f_contiguous)
        queries, targets = full.load_full_queries('train', batch_size=2)
        names, definitions = full.load_selected_definitions(self.selection_path, self.definition_dir)
        expected = np.empty_like(matrix)
        for parent, key in full.BRANCHES:
            history = sample.load_branch_history(queries, key, batch_size=2)
            chosen = definitions[parent]
            expected[:, [i for i, _ in chosen]] = calculate_temporal(
                history, queries, parent, key, [f for _, f in chosen])
        np.testing.assert_array_equal(matrix, expected)
        repeated = full.generate_full_split('train', out, selection_path=self.selection_path,
            definition_dir=self.definition_dir, resume=True, history_chunk=2, query_chunk=2)
        self.assertEqual(repeated, report)
        self.assertFalse(list(out.glob('*scratch*')))

    def test_resume_skips_committed_branch_after_interruption(self):
        out = self.root / 'interrupted'
        def interrupt_second_branch(event):
            if event.get('stage') == 'history' and event.get('branch') == 'Items':
                raise RuntimeError('simulated interruption')
        arguments = dict(selection_path=self.selection_path, definition_dir=self.definition_dir,
                         history_chunk=2, query_chunk=2)
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            full.generate_full_split('train', out, progress=interrupt_second_branch, **arguments)
        checkpoint = json.loads((out / 'train_preparation.json').read_text())
        self.assertTrue(checkpoint['branches']['Users']['completed'])
        self.assertFalse(checkpoint['completed'])
        with patch.object(full, 'load_branch_history', wraps=sample.load_branch_history) as loader:
            report = full.generate_full_split('train', out, resume=True, **arguments)
        self.assertTrue(report['completed'])
        self.assertEqual(loader.call_count, 3)
        self.assertEqual([call.args[1] for call in loader.call_args_list],
                         ['item_id', 'user_item_id', 'user_artist_id'])

    def test_test_split_and_changed_selection_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'train и val'):
            full.load_full_queries('test')
        out = self.root / 'output'
        full.generate_full_split('val', out, selection_path=self.selection_path,
            definition_dir=self.definition_dir, history_chunk=2, query_chunk=2)
        selected = json.loads(self.selection_path.read_text())
        selected['features'] = selected['features'][::-1]
        self.selection_path.write_text(json.dumps(selected))
        with self.assertRaisesRegex(ValueError, 'Источники изменились'):
            full.generate_full_split('val', out, selection_path=self.selection_path,
                definition_dir=self.definition_dir, resume=True)


if __name__ == '__main__':
    unittest.main()
