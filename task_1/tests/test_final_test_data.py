"""Общие test-запросы проверяются только на искусственных строках."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from final_test_data import canonical_queries, test_mask


class FinalTestDataTest(unittest.TestCase):
    def test_mapping_and_constant_group_removal(self):
        frame = pd.DataFrame({'log_id': [0,1,2,3,4,5], 'uid': [2,1,2,1,3,3],
            'item_id':[1]*6,'artist_id':[5]*6,'start_ts':[285*86400]*6,
            'timestamp':[285*86400+100]*6,'label':[0,0,2,1,1,1],
            'track_length_seconds':[200]*6,'is_organic':[0]*6,
            'group_id':[2285,1285,2285,1285,3285,3285]})
        queries, targets, stats = canonical_queries(frame)
        np.testing.assert_array_equal(targets['log_id'], [1,3,0,2])
        np.testing.assert_array_equal(targets['source_position'], [1,3,0,2])
        self.assertEqual(stats['groups_used'], 2)
        self.assertEqual(len(queries), 4)

    def test_lower_bound_and_no_day300_truncation(self):
        start = 285*86400
        table=pa.table({'start_ts':[start-1,start,301*86400],
                        'timestamp':[start+1,start+2,301*86400+50]})
        np.testing.assert_array_equal(test_mask(table).to_numpy(), [False,True,True])

    def test_duplicate_log_ids_fail(self):
        with self.assertRaises(ValueError):
            canonical_queries(pd.DataFrame({'log_id':[1,1]}))


if __name__=='__main__':
    unittest.main()
