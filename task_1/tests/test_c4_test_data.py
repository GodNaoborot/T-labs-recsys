"""Небольшие C4 test fixtures; настоящие test-данные здесь не читаются."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import c4_test_data as test_data
import c4_targets as c4


def listens(times, labels=None, users=None, items=None):
    size = len(times)
    labels = [1] * size if labels is None else labels
    return pd.DataFrame({'uid': [1] * size if users is None else users,
        'item_id': list(range(10, 10 + size)) if items is None else items,
        'artist_id': [100] * size, 'timestamp': times,
        'start_ts': [value - 50 for value in times],
        'played_ratio_pct': [10 if label == 0 else 100 for label in labels],
        'track_length_seconds': [100] * size, 'is_organic': [1] * size,
        'label': labels})


def frozen_plan(source_dir):
    return {'test_evaluation_authorized': True, 'new_training_allowed': False,
            'test_already_open': True,
            'sources': {name: {'path': str(source_dir / f'{name}.parquet'),
                               'sha256': c4._sha256(source_dir / f'{name}.parquet')}
                        for name in ('listens', 'events')}}


class C4TestDataTest(unittest.TestCase):
    def test_boundary_uses_cutoff_and_keeps_observed_tail_without_gap(self):
        base = c4.TEST_START
        frame = listens([base, base + 5, base + 10, base + 10000])
        cutoff = np.array([base - 5, base, base + 5, base + 9995])
        masks = test_data.test_target_masks(frame, cutoff, np.full(4, c4.NO_REACTION))
        np.testing.assert_array_equal(masks['test'], [False, True, True, True])
        # Нет выдуманной конечной границы и GAP на конце test.
        self.assertFalse(masks['first_packet'].any())

    def test_tied_target_packet_keeps_common_cutoff_under_permutation(self):
        base = c4.TEST_START
        frame = listens([base - 5, base, base + 5, base + 5, base + 10],
                        labels=[1, 1, 0, 1, 0])
        frame['log_id'] = [7, 3, 4, 5, 9]
        def selected(raw):
            cutoff = c4.packet_cutoffs(raw)
            mask = test_data.test_target_masks(raw, cutoff,
                np.full(len(raw), c4.NO_REACTION))['test']
            query = c4._queries(raw, cutoff, mask)
            return c4.select_queries(query)
        expected, expected_arrays, expected_stats = selected(frame)
        actual, actual_arrays, actual_stats = selected(frame.iloc[[3, 4, 1, 0, 2]])
        pd.testing.assert_frame_equal(expected, actual)
        for name in expected_arrays:
            np.testing.assert_array_equal(expected_arrays[name], actual_arrays[name])
        self.assertEqual(expected_stats['shared_order_sha256'], actual_stats['shared_order_sha256'])
        np.testing.assert_array_equal(expected.time, [base, base, base + 5])

    def test_guard_checks_any_reaction_but_allows_equality(self):
        base = c4.TEST_START
        frame = listens([base + 10] * 3, labels=[2, 2, 1])
        cutoff = np.array([base + 5] * 3)
        masks = test_data.test_target_masks(frame, cutoff,
            [base + 4, base + 5, c4.NO_REACTION])
        np.testing.assert_array_equal(masks['test'], [False, True, True])
        with self.assertRaises(AssertionError):
            test_data.test_target_masks(frame, [base + 10] * 3,
                np.full(3, c4.NO_REACTION))

    def test_reader_keeps_full_users_source_ids_and_post285_rows(self):
        base = c4.TEST_START
        frame = listens([base - 5, base, base + 5, base + 100, base + 105],
                        users=[1, 1, 1, 2, 2])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'listens.parquet'
            frame.to_parquet(source, index=False)
            counters = {}
            blocks = list(test_data._listen_blocks(source, counters, batch_size=2))
            self.assertEqual(len(blocks), 2)
            np.testing.assert_array_equal(blocks[0].log_id, [0, 1, 2])
            np.testing.assert_array_equal(blocks[1].log_id, [3, 4])
            self.assertEqual(counters['source_rows_scanned'], 5)
            self.assertTrue((blocks[1].timestamp >= base).all())

    def test_reader_rejects_uid_order_break_at_batch_boundary(self):
        frame = listens([1, 2, 3], users=[1, 2, 1])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'listens.parquet'
            frame.to_parquet(source, index=False)
            with self.assertRaises(ValueError):
                list(test_data._listen_blocks(source, {}, batch_size=2))

    def test_small_prepare_verifies_labels_guard_mixed_groups_and_alias(self):
        base = c4.TEST_START
        frame = listens([base - 5, base, base + 5, base + 5, base + 10,
                         base + 15, base + 20, base + 100, base + 105],
                        labels=[1, 1, 0, 1, 2, 2, 1, 1, 1],
                        users=[1] * 7 + [2, 2])
        events = pd.DataFrame({'uid': [1] * 4, 'item_id': [14, 14, 15, 15],
            'timestamp': [base + 4, base + 9, base + 10, base + 100],
            'event_type': ['dislike', 'like', 'like', 'unlike']})
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            frame.to_parquet(folder / 'listens.parquet', index=False, row_group_size=2)
            events.to_parquet(folder / 'events.parquet', index=False)
            updates = []
            report = test_data.prepare_test_targets(folder / 'data', folder / 'result',
                                                    frozen_plan(folder), updates.append)
            self.assertTrue(report['completed'])
            self.assertFalse(report['new_training'])
            stats = report['test']
            self.assertEqual(stats['rows_before_mixed'], 5)
            self.assertEqual(stats['constant_group_rows_removed'], 1)
            self.assertEqual(stats['rows'], 4)
            self.assertEqual(stats['groups'], 1)
            query = pd.read_parquet(folder / 'data/test_queries.parquet')
            np.testing.assert_array_equal(query.log_id, [2, 3, 5, 6])
            np.testing.assert_array_equal(query.time, [base, base, base + 10, base + 15])
            np.testing.assert_array_equal(query.start_ts, query.time)
            with np.load(folder / 'data/test_targets.npz') as arrays:
                np.testing.assert_array_equal(arrays['label'], [0, 1, 2, 1])
                np.testing.assert_array_equal(arrays['log_id'], query.log_id)
            self.assertEqual(c4._sha256(folder / 'data/targets.npz'),
                             c4._sha256(folder / 'data/test_targets.npz'))
            guard = report['exclusions']['by_split']['test']['attached_reaction_before_cutoff']
            self.assertEqual(guard['rows'], 1)
            self.assertEqual(guard['labels'], {'0': 0, '1': 0, '2': 1})
            self.assertEqual(report['history_label_availability']['verified_labels'], 9)
            self.assertTrue(report['history_label_availability']['frozen_labels_match'])
            attached = pd.read_parquet(folder / 'data/reaction_attachments.parquet')
            np.testing.assert_array_equal(attached.reaction_source_id, [0, 1, 2])
            np.testing.assert_array_equal(attached.log_id, [4, 4, 5])
            self.assertTrue(updates)
            self.assertFalse(any(path.name.startswith('train_') or path.name.startswith('val_')
                                 for path in (folder / 'data').iterdir()))
            with self.assertRaises(FileExistsError):
                test_data.prepare_test_targets(folder / 'data', folder / 'result', frozen_plan(folder))

    def test_plan_rejects_training_and_source_changes_before_outputs(self):
        frame = listens([c4.TEST_START, c4.TEST_START + 5])
        events = pd.DataFrame({'uid': [1], 'item_id': [10], 'timestamp': [0],
                               'event_type': ['unlike']})
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            frame.to_parquet(folder / 'listens.parquet', index=False)
            events.to_parquet(folder / 'events.parquet', index=False)
            plan = frozen_plan(folder)
            plan['new_training_allowed'] = True
            with self.assertRaises(ValueError):
                test_data.prepare_test_targets(folder / 'data', folder / 'result', plan)
            self.assertFalse((folder / 'data').exists())
            plan = frozen_plan(folder)
            plan['sources']['listens']['sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                test_data.prepare_test_targets(folder / 'data', folder / 'result', plan)
            self.assertFalse((folder / 'data').exists())


if __name__ == '__main__':
    unittest.main()
