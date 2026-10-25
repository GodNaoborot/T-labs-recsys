"""Подготовка истории проверяется на маленьком журнале без сети и обучения."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import prepare_history as history


def raw_fixture():
    rows = [
        (1, 10, 100, 'listen', 34, 100, 1),
        (1, 10, 100, 'listen', 100, 100, 0),
        (1, 10, 70, 'like', np.nan, np.nan, np.nan),
        (1, 10, 80, 'dislike', np.nan, np.nan, np.nan),
        (1, 10, 100, 'like', np.nan, np.nan, np.nan),
        (1, 10, 200, 'listen', 35, 100, 1),
        (1, 10, 150, 'like', np.nan, np.nan, np.nan),
        (1, 10, 195, 'dislike', np.nan, np.nan, np.nan),
        (1, 10, 101, 'unlike', np.nan, np.nan, np.nan),
        (1, 11, 300, 'listen', 35, 100, 1),
        (1, 12, 400, 'listen', 100, 100, 1),
        (1, 12, 100, 'like', np.nan, np.nan, np.nan),
        (1, 13, 500, 'listen', 10, 100, 1),
        (1, 13, 500, 'like', np.nan, np.nan, np.nan),
        (1, 999, 500, 'dislike', np.nan, np.nan, np.nan),
        (1, 11, 301, 'undislike', np.nan, np.nan, np.nan),
        (2, 10, 0, 'listen', 100, 50, 1),
    ]
    return pd.DataFrame(rows, columns=history.RAW_COLUMNS)


def artist_fixture():
    return pd.DataFrame({'item_id': [10, 10, 11, 12], 'artist_id': [111, 222, 333, 444]})


def notebook_reference(raw, artists):
    """Буквальные формулы ячеек 21/24/27/33/37, без EDA и печати."""
    frame = raw.copy()
    is_listen = frame.event_type.eq('listen').to_numpy()
    starts = np.floor(frame.timestamp.to_numpy('float64')
                      - frame.track_length_seconds.to_numpy('float64')
                      * frame.played_ratio_pct.to_numpy('float64') / 100 - 5)
    frame['start_ts'] = pd.array(np.where(is_listen, starts, np.nan)).astype('Int32')
    frame['label'] = pd.Series((frame.played_ratio_pct >= 35).to_numpy('int8')).astype('Int8').where(is_listen)
    timestamps = frame.timestamp.to_numpy('int64')
    pair = pd.factorize(frame.uid.to_numpy('int64') * 10 ** 7
                        + frame.item_id.to_numpy('int64'))[0]
    listen_rows = np.flatnonzero(is_listen)
    code = pair[listen_rows] * (2 ** 26) + timestamps[listen_rows]
    order = np.argsort(code, kind='stable')
    code, listen_rows = code[order], listen_rows[order]
    start_values = frame.start_ts.to_numpy('float64', na_value=np.nan)
    attached = {}
    for kind in ['like', 'dislike']:
        event_rows = np.flatnonzero(frame.event_type.eq(kind).to_numpy())
        positions = np.searchsorted(code, pair[event_rows] * (2 ** 26) + timestamps[event_rows], side='left')
        safe = np.minimum(positions, len(code) - 1)
        same = (positions < len(code)) & (code[safe] // (2 ** 26) == pair[event_rows])
        chosen, times = listen_rows[safe[same]], timestamps[event_rows][same]
        attached[kind] = np.unique(chosen[times >= start_values[chosen]])
    frame.loc[attached['dislike'], 'label'] = 0
    frame.loc[attached['like'], 'label'] = 2
    frame['day'] = (frame.start_ts // 86400).clip(lower=0).astype('Int16')
    frame['group_id'] = frame.uid.astype('int64') * 1000 + frame.day
    selected = frame.loc[is_listen]
    mapping = artists.drop_duplicates('item_id', keep='first').set_index('item_id').artist_id
    listens = pd.DataFrame({
        'uid': selected.uid.to_numpy('uint32'), 'item_id': selected.item_id.to_numpy('uint32'),
        'artist_id': selected.item_id.map(mapping).fillna(-1).to_numpy('int32'),
        'timestamp': selected.timestamp.to_numpy('int32'),
        'start_ts': selected.start_ts.to_numpy('int32'),
        'played_ratio_pct': selected.played_ratio_pct.to_numpy('uint8'),
        'track_length_seconds': selected.track_length_seconds.to_numpy('uint16'),
        'is_organic': selected.is_organic.to_numpy('uint8'),
        'label': selected.label.to_numpy('int8'),
        'group_id': selected.group_id.to_numpy('int64'),
    })
    events = frame.loc[frame.event_type.isin(history.REACTIONS),
                       ['uid', 'item_id', 'timestamp', 'event_type']]
    return listens, events.astype({'timestamp': 'int32', 'event_type': 'category'})


class PrepareHistoryTest(unittest.TestCase):
    def test_matches_notebook_with_ties_reactions_threshold_and_artist_mapping(self):
        raw, artists = raw_fixture(), artist_fixture()
        expected_listens, expected_events = notebook_reference(raw, artists)
        listens, events, report = history.build_history(raw, artists)
        pd.testing.assert_frame_equal(listens, expected_listens)
        pd.testing.assert_frame_equal(events, expected_events)
        self.assertEqual(listens.label.tolist(), [2, 1, 0, 1, 1, 2, 1])
        self.assertEqual(listens.artist_id.tolist(), [111, 111, 111, 333, 444, -1, 111])
        self.assertEqual(listens.group_id.iloc[-1], 2000)
        self.assertEqual(report['listens_with_both'], 1)

    def test_columnwise_inplace_work_does_not_mutate_raw_input(self):
        raw = raw_fixture().query("event_type == 'listen'").reset_index(drop=True)
        expected = raw.copy(deep=True)
        history.build_history(raw, artist_fixture())
        pd.testing.assert_frame_equal(raw, expected)

    def test_same_timestamp_reaction_attaches_first_physical_listen(self):
        raw = raw_fixture()
        raw.iloc[[0, 1]] = raw.iloc[[1, 0]].to_numpy()
        listens, _, _ = history.build_history(raw, artist_fixture())
        self.assertEqual(listens.label.iloc[0], 2)
        self.assertEqual(listens.label.iloc[1], 0)

    def test_no_reactions_and_empty_reaction_output(self):
        raw = raw_fixture().query("event_type == 'listen'").reset_index(drop=True)
        expected, expected_events = notebook_reference(raw, artist_fixture())
        actual, events, _ = history.build_history(raw, artist_fixture())
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(events, expected_events)
        self.assertEqual(len(events), 0)

    def test_invalid_or_incompatible_inputs_fail_without_reordering(self):
        for column, value in [('timestamp', history.SHIFT), ('item_id', history.PAIR_BASE),
                              ('played_ratio_pct', np.nan), ('track_length_seconds', 65536)]:
            raw = raw_fixture()
            raw.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                history.build_history(raw, artist_fixture())
        raw = raw_fixture().iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, 'порядок'):
            history.build_history(raw, artist_fixture())
        with self.assertRaisesRegex(ValueError, 'нет listen'):
            history.build_history(raw_fixture().query("event_type != 'listen'"), artist_fixture())

    def test_disk_outputs_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw_path, artists_path = folder / 'raw.parquet', folder / 'artists.parquet'
            raw_fixture().to_parquet(raw_path, index=False)
            artist_fixture().to_parquet(artists_path, index=False)
            output = folder / 'data'
            report = history.prepare_history(raw_path, artists_path, output)
            loaded = pq.read_table(raw_path).to_pandas(categories=['event_type'])
            expected, events = notebook_reference(loaded, artist_fixture())
            pd.testing.assert_frame_equal(pd.read_parquet(output / 'listens.parquet'), expected)
            pd.testing.assert_frame_equal(pd.read_parquet(output / 'events.parquet'),
                                          events.reset_index(drop=True))
            saved = json.loads((output / 'history_preparation.json').read_text())
            self.assertEqual(saved['listen_rows'], len(expected))
            for name, record in saved['outputs'].items():
                self.assertEqual(record['sha256'], history.file_hash(output / name))
                self.assertEqual(record['path'], str(output / name))
            before = (output / 'listens.parquet').read_bytes()
            with self.assertRaises(FileExistsError):
                history.prepare_history(raw_path, artists_path, output)
            self.assertEqual(before, (output / 'listens.parquet').read_bytes())
            self.assertFalse((output / '.history-preparation.lock').exists())
            self.assertEqual(report['sources']['raw_events']['sha256'], history.file_hash(raw_path))

    def test_failure_does_not_publish_partial_history(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw_path, artists_path = folder / 'raw.parquet', folder / 'artists.parquet'
            raw_fixture().to_parquet(raw_path, index=False)
            artist_fixture().to_parquet(artists_path, index=False)
            output = folder / 'data'
            with patch.object(history, 'build_history', side_effect=ValueError('invalid fixture')):
                with self.assertRaisesRegex(ValueError, 'invalid fixture'):
                    history.prepare_history(raw_path, artists_path, output)
            self.assertEqual(list(output.iterdir()), [])

    def test_publish_collision_rolls_back_only_own_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            staging, output = folder / 'staging', folder / 'output'
            staging.mkdir()
            output.mkdir()
            for name in history.OUTPUT_FILES:
                (staging / name).write_text('new')
            (output / 'events.parquet').write_text('existing')
            with self.assertRaises(FileExistsError):
                history.publish_staged(staging, output)
            self.assertFalse((output / 'listens.parquet').exists())
            self.assertFalse((output / 'history_preparation.json').exists())
            self.assertEqual((output / 'events.parquet').read_text(), 'existing')

    def test_default_download_revision_is_pinned_source_commit(self):
        with patch.object(sys, 'argv', ['prepare_history.py', '--download']):
            args = history.parse_args()
        self.assertEqual(args.revision, 'dd6f3a19eef5866e346c3270e098baa641a44948')
        self.assertEqual(history.DATASET, 'yandex/yambda')

    def test_check_only_download_has_no_network_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            arguments = ['prepare_history.py', '--download', '--check-only', '--output-dir', str(output)]
            with patch.object(sys, 'argv', arguments), patch.object(history, 'download_inputs') as download:
                history.main()
            download.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
