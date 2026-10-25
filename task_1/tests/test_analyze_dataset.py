"""Порционное EDA должно совпадать с прямыми pandas-расчётами."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import analyze_dataset as module


def fixture():
    return pd.DataFrame([
        (1, 10, 100, 'listen', 10., 100., 0),
        (1, 10, 100, 'listen', 10., 100., 0),
        (1, 10, 120, 'like', np.nan, np.nan, 0),
        (1, 10, 150, 'listen', 80., 100., 1),
        (2, 20, 200, 'listen', 100., 200., 0),
        (2, 20, 210, 'like', np.nan, np.nan, 0),
        (3, 30, 300, 'dislike', np.nan, np.nan, 1),
    ], columns=module.RAW_COLUMNS)


class AnalyzeDatasetTests(unittest.TestCase):
    def test_streaming_matches_whole_table_with_boundary_duplicates(self):
        frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.parquet'
            frame.to_parquet(path, index=False)
            iterator = module.user_blocks
            with patch.object(module, 'user_blocks', side_effect=lambda source: iterator(source, batch_size=2)):
                counts, schema, ratios, _, _ = module.collect_statistics(path)
        self.assertEqual(counts['rows'], len(frame))
        self.assertEqual(counts['duplicated_rows'], int(frame.duplicated().sum()))
        self.assertEqual(counts['users'], frame.uid.nunique())
        self.assertEqual(counts['items'], frame.item_id.nunique())
        for name in module.RAW_COLUMNS:
            row = schema.loc[schema.column.eq(name)].iloc[0]
            self.assertEqual(row['missing'], frame[name].isna().sum())
            self.assertEqual(row['unique'], frame[name].nunique())
        self.assertEqual(ratios['count'].sum(), frame.event_type.eq('listen').sum())

    def test_saved_timeline_preserves_normalized_counts_and_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.parquet'
            fixture().to_parquet(path, index=False)
            _, _, _, edges, histograms = module.collect_statistics(path)
        table = module.timeline_table(edges, histograms)
        self.assertTrue((table.bin_right > table.bin_left).all())
        self.assertFalse(table.duplicated(['histogram', 'event', 'bin_left']).any())
        for name in table.histogram.unique():
            totals = table.loc[table.histogram.eq(name)].groupby('event').normalized_value.sum()
            self.assertAlmostEqual(totals['listen'], 2.)
            self.assertAlmostEqual(totals['like'], 1.)

    def test_first_like_and_first_listen_definition(self):
        timeline, pairs = module.liked_timeline(fixture())
        self.assertEqual(pairs, 2)
        user = timeline.loc[timeline.uid.eq(1)]
        self.assertTrue(user.like_ts.eq(120).all())
        self.assertTrue(user.first_ts.eq(100).all())
        self.assertEqual(user.event.eq('listen').sum(), 3)

    def test_rejects_unordered_users(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.parquet'
            fixture().iloc[::-1].to_parquet(path, index=False)
            with self.assertRaises(ValueError):
                list(module.user_blocks(path, batch_size=2))


if __name__ == '__main__':
    unittest.main()
