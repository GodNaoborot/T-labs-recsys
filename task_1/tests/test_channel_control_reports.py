"""Готовый контроль должен сохранять метрики и сопоставимые ссылки на исходные модели."""
from pathlib import Path
import hashlib
import json
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'task_1/results/c4/no_current_channel'
VARIANT = 'manual48_no_current_channel'


class ChannelControlReportTests(unittest.TestCase):
    def test_reference_rows_match_original_results(self):
        comparison = pd.read_csv(CONTROL / 'comparison.csv', float_precision='round_trip')
        result = json.loads((CONTROL / 'result.json').read_text())
        for split, path in [('validation', 'c4/c4_packet_20261024_v1'),
                            ('test', 'c4_test/c4_test_20261024_v1')]:
            original = pd.read_csv(ROOT / 'task_1/results' / path / 'comparison.csv',
                                   float_precision='round_trip').set_index('variant')
            rows = comparison.loc[comparison['split'] == split].set_index('variant')
            for variant in ['C2b-t', 'D462-t']:
                for column in ['NDCG@10', 'QueryAUC', 'rows', 'groups', 'tree_count']:
                    self.assertEqual(rows.loc[variant, column], original.loc[variant, column])
            for metric, value in result[split]['metrics'].items():
                self.assertEqual(rows.loc[VARIANT, metric], value)

    def test_intervals_refer_to_matched_control_delta(self):
        comparison = pd.read_csv(CONTROL / 'comparison.csv')
        intervals = pd.read_csv(CONTROL / 'bootstrap_summary.csv')
        self.assertEqual(len(intervals), 4)
        for split in ['validation', 'test']:
            rows = comparison.loc[comparison['split'] == split].set_index('variant')
            delta = rows.loc[VARIANT, 'NDCG@10'] - rows.loc['D462-t', 'NDCG@10']
            subset = intervals.loc[intervals['split'] == split]
            self.assertEqual(set(subset['block_days']), {3, 4})
            for row in subset.itertuples(index=False):
                self.assertEqual(row.candidate, VARIANT)
                self.assertEqual(row.reference, 'D462-t')
                self.assertAlmostEqual(row.delta_ndcg, delta, places=12)
                self.assertLessEqual(row.ci_low, row.ci_high)

    def test_mask_and_report_hashes_are_consistent(self):
        protocol = json.loads((CONTROL / 'protocol.json').read_text())
        names = protocol['physical_features']
        active = [name for i, name in enumerate(names) if i not in protocol['ignored_indices']]
        self.assertEqual(active, protocol['active_features'])
        self.assertNotIn('is_organic', active)
        self.assertEqual(len(active), 48)
        provenance = json.loads((CONTROL / 'provenance.json').read_text())
        for name, digest in provenance['published_report_sha256'].items():
            self.assertEqual(hashlib.sha256((CONTROL / name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
