"""C4 проверяется на маленьких событиях, без обучения и доступа к настоящему test."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import c4_targets as c4


def listens(times, labels=None, users=None, items=None, starts=None, ids=None):
    n = len(times)
    labels = [1] * n if labels is None else labels
    return pd.DataFrame({'log_id': list(range(n)) if ids is None else ids,
        'uid': [1] * n if users is None else users,
        'item_id': [10] * n if items is None else items,
        'artist_id': [100] * n, 'timestamp': times,
        'start_ts': [value - 50 for value in times] if starts is None else starts,
        'played_ratio_pct': [10 if label == 0 else 100 for label in labels],
        'track_length_seconds': [100] * n, 'is_organic': [1] * n, 'label': labels})


def reactions(times, kinds, users=None, items=None):
    n = len(times)
    return pd.DataFrame({'reaction_source_id': list(range(n)),
        'uid': [1] * n if users is None else users,
        'item_id': [10] * n if items is None else items,
        'timestamp': times, 'event_type': kinds})


class C4TargetsTest(unittest.TestCase):
    def test_packet_cutoff_is_previous_distinct_time(self):
        frame = listens([95, 100, 100, 90, 110], users=[1, 1, 1, 2, 2])
        np.testing.assert_array_equal(c4.packet_cutoffs(frame), [-1, 95, 95, -1, 90])

    def test_packet_permutation_preserves_source_id_results(self):
        frame = listens([95, 100, 100, 110], ids=[8, 3, 4, 10])
        expected = pd.Series(c4.packet_cutoffs(frame), index=frame.log_id).sort_index()
        shuffled = frame.iloc[[2, 3, 0, 1]]
        actual = pd.Series(c4.packet_cutoffs(shuffled), index=shuffled.log_id).sort_index()
        pd.testing.assert_series_equal(expected, actual)

    def test_packet_permutation_preserves_split_group_and_feature_query_order(self):
        base = c4.VAL_START
        frame = listens([base - 50, base + 10, base + 10, base + 20],
                        labels=[1, 0, 1, 0], ids=[2, 8, 4, 9])
        def selected(raw):
            cutoffs = c4.packet_cutoffs(raw)
            masks = c4.target_masks(raw, cutoffs, np.full(len(raw), c4.NO_REACTION))
            query = c4._queries(raw, cutoffs, masks['val'])
            query, arrays, report = c4.select_queries(query)
            return query, arrays, report
        # Пакет с cutoff до val исключён по GAP; следующий пакет делает val-группу.
        frame = pd.concat([frame, listens([base + 20], labels=[1], ids=[11])], ignore_index=True)
        expected, expected_arrays, expected_report = selected(frame)
        actual, actual_arrays, actual_report = selected(frame.iloc[[3, 2, 0, 4, 1]])
        pd.testing.assert_frame_equal(expected, actual)
        for name in expected_arrays:
            np.testing.assert_array_equal(expected_arrays[name], actual_arrays[name])
        self.assertEqual(expected_report['shared_order_sha256'], actual_report['shared_order_sha256'])

    def test_mutating_current_ratio_and_length_does_not_change_cutoff(self):
        frame = listens([95, 100, 100])
        expected = c4.packet_cutoffs(frame)
        frame['played_ratio_pct'], frame['track_length_seconds'], frame['label'] = 0, 1000, 0
        np.testing.assert_array_equal(c4.packet_cutoffs(frame), expected)

    def test_attachment_next_same_pair_and_tie_first_source_id(self):
        frame = listens([100, 100, 110, 120], items=[10, 10, 20, 10],
                        ids=[7, 5, 9, 10], starts=[50, 50, 90, 100])
        events = reactions([80, 105, 85, 90], ['like'] * 4, items=[10, 10, 20, 999])
        attached = c4.attach_reactions(frame, events)
        np.testing.assert_array_equal(attached.reaction_source_id, [0, 1])
        np.testing.assert_array_equal(attached.log_id, [5, 10])
        shuffled = c4.attach_reactions(frame.iloc[::-1], events.iloc[::-1])
        pd.testing.assert_frame_equal(attached.sort_values('reaction_source_id').reset_index(drop=True),
                                      shuffled.sort_values('reaction_source_id').reset_index(drop=True))

    def test_any_earlier_reaction_excluded_despite_like_priority(self):
        frame = listens([100], labels=[2], starts=[50])
        attached = c4.attach_reactions(frame, reactions([60, 80], ['dislike', 'like']))
        earliest, liked, disliked = c4.reaction_state(frame, attached)
        self.assertTrue(liked[0] and disliked[0])
        np.testing.assert_array_equal(earliest, [60])
        cutoff = np.array([c4.TRAIN_START + 75])
        frame['timestamp'] += c4.TRAIN_START
        earliest += c4.TRAIN_START
        mask = c4.target_masks(frame, cutoff, earliest)
        self.assertTrue(mask['attached_reaction_before_cutoff'][0])
        self.assertFalse(mask['train'][0])

    def test_reaction_at_cutoff_allowed(self):
        frame = listens([c4.TRAIN_START + 100], labels=[2], starts=[c4.TRAIN_START + 50])
        attached = c4.attach_reactions(frame, reactions([c4.TRAIN_START + 75], ['like']))
        earliest, _, _ = c4.reaction_state(frame, attached)
        masks = c4.target_masks(frame, [c4.TRAIN_START + 75], earliest)
        self.assertTrue(masks['train'][0])
        self.assertFalse(masks['attached_reaction_before_cutoff'][0])

    def test_reaction_attached_to_other_listen_does_not_exclude_target(self):
        base = c4.TRAIN_START
        frame = listens([base + 70, base + 100], labels=[2, 1], starts=[base + 10, base + 80])
        attached = c4.attach_reactions(frame, reactions([base + 60], ['like']))
        earliest, _, _ = c4.reaction_state(frame, attached)
        masks = c4.target_masks(frame, [base + 50, base + 90], earliest)
        self.assertTrue(masks['train'][1])
        self.assertEqual(earliest[1], c4.NO_REACTION)

    def test_unknown_future_label_component_fails(self):
        frame = listens([100])
        attached = pd.DataFrame({'log_id': [0], 'timestamp': [101], 'event_type': ['like']})
        with self.assertRaises(AssertionError):
            c4.reaction_state(frame, attached)

    def test_frozen_label_mismatch_fails(self):
        frame = listens([100], labels=[1])
        attached = c4.attach_reactions(frame, reactions([80], ['like']))
        with self.assertRaises(AssertionError):
            c4.reaction_state(frame, attached)

    def test_gap_is_on_target_end_not_on_reconstructed_start(self):
        ends = [c4.VAL_START - c4.GAP - 1, c4.VAL_START - c4.GAP,
                c4.TEST_START - c4.GAP - 1, c4.TEST_START - c4.GAP,
                c4.TEST_START + 10]
        frame = listens(ends)
        cutoff = [c4.VAL_START - 2000, c4.VAL_START - 2000,
                  c4.TEST_START - 2000, c4.TEST_START - 2000, c4.TEST_START - 100]
        masks = c4.target_masks(frame, cutoff, np.full(5, c4.NO_REACTION))
        np.testing.assert_array_equal(masks['train'], [True, False, False, False, False])
        np.testing.assert_array_equal(masks['val'], [False, False, True, False, False])

    def test_mixed_groups_and_order_rebuilt_from_t(self):
        base = c4.TRAIN_START
        frame = listens([base + 100] * 6, labels=[1, 0, 2, 1, 1, 1],
                        users=[2, 1, 2, 1, 3, 3], ids=[20, 11, 21, 10, 30, 31])
        query = c4._queries(frame, np.full(6, base), np.ones(6, dtype='bool'))
        query, arrays, report = c4.select_queries(query, train_groups=2)
        np.testing.assert_array_equal(arrays['log_id'], [10, 11, 20, 21])
        np.testing.assert_array_equal(arrays['group_id'], [1225, 1225, 2225, 2225])
        self.assertEqual(report['constant_group_rows_removed'], 2)
        self.assertTrue((query.start_ts == base).all())
        with self.assertRaises(ValueError):
            c4.select_queries(query, train_groups=3)

    def test_small_end_to_end_preserves_ids_and_only_builds_train_val(self):
        times = [c4.TRAIN_START - 10, c4.TRAIN_START + 10,
                 c4.TRAIN_START + 20, c4.TRAIN_START + 30,
                 c4.VAL_START + 10, c4.VAL_START + 20, c4.VAL_START + 30,
                 c4.TEST_START + 100]
        frame = listens(times, labels=[1, 1, 0, 1, 1, 0, 1, 0])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            pq.write_table(pa.Table.from_pandas(frame.drop(columns='log_id'), preserve_index=False),
                           root / 'data/listens.parquet', row_group_size=2)
            events = reactions([c4.TRAIN_START - 100], ['unlike']).drop(columns='reaction_source_id')
            events.to_parquet(root / 'data/events.parquet', index=False)
            with patch.object(c4, 'ROOT', root):
                report = c4.prepare_targets(root / 'output', root / 'reports', train_groups=1)
            self.assertTrue(report['completed'])
            self.assertTrue(report['history_label_availability']['frozen_labels_match'])
            self.assertEqual(report['history_label_availability']['historical_listen_rows'], 7)
            with np.load(root / 'output/train_targets.npz') as arrays:
                np.testing.assert_array_equal(arrays['log_id'], [2, 3])
            with np.load(root / 'output/val_targets.npz') as arrays:
                np.testing.assert_array_equal(arrays['log_id'], [5, 6])
            self.assertFalse(any('test' in path.name for path in (root / 'output').iterdir()))
            self.assertEqual(report['train']['groups'], 1)
            self.assertEqual(report['val']['groups'], 1)
            with patch.object(c4, 'ROOT', root), self.assertRaises(FileExistsError):
                c4.prepare_targets(root / 'output', root / 'reports', train_groups=1)

    def test_user_blocks_keep_physical_ids_and_do_not_break_user(self):
        frame = listens([10, c4.TEST_START + 1, 20, 30, 40], users=[1, 1, 2, 2, 3])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'listens.parquet'
            frame.drop(columns='log_id').to_parquet(path, index=False)
            counters = {}
            blocks = list(c4._listen_blocks(path, counters, batch_size=2))
            ids = np.concatenate([block.log_id.to_numpy() for block in blocks])
            np.testing.assert_array_equal(ids, [0, 2, 3, 4])
            owners = {}
            for i, block in enumerate(blocks):
                for uid in block.uid.unique():
                    self.assertNotIn(uid, owners)
                    owners[uid] = i
            self.assertEqual(counters['source_rows_scanned'], 5)


if __name__ == '__main__':
    unittest.main()
