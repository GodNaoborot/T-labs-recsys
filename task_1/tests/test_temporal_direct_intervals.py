"""Проверки прямых разностей без моделей и реальных данных."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from compare_temporal_controls_pairs import absolute_delta_draws, pair_daily


class DirectIntervalsTest(unittest.TestCase):
    def test_group_weighting_and_common_draws(self):
        daily = pd.DataFrame({'group_count': [1, 3], 'base_sum': [.2, 1.5],
                              'dfs_sum': [.4, 2.1]})
        indices = np.array([[0, 1], [1, 1], [0, 0]])
        np.testing.assert_allclose(absolute_delta_draws(daily, indices), [.2, .2, .2])
        daily['dfs_sum'] = daily['base_sum']
        np.testing.assert_array_equal(absolute_delta_draws(daily, indices), np.zeros(3))

    def test_actual_group_weights_are_not_daily_means(self):
        daily = pd.DataFrame({'group_count': [1, 3], 'base_sum': [.2, 1.5],
                              'dfs_sum': [.4, 1.5]})
        np.testing.assert_allclose(absolute_delta_draws(daily, np.array([[0, 1]])), [.05])

    def test_misaligned_pairs_fail(self):
        frame = pd.DataFrame({'group_id': [10, 20], 'day': [270, 271],
                              'group_size': [2, 3], 'ndcg': [.4, .8]})
        wrong = frame.copy()
        wrong.loc[1, 'group_id'] = 21
        with self.assertRaises(AssertionError):
            pair_daily(frame, wrong)
        daily = pair_daily(frame, frame)
        np.testing.assert_array_equal(daily['group_count'], [1, 1])

    def test_empty_replicate_fails(self):
        daily = pd.DataFrame({'group_count': [0], 'base_sum': [0.], 'dfs_sum': [0.]})
        with self.assertRaises(ValueError):
            absolute_delta_draws(daily, np.array([[0]]))


if __name__ == '__main__':
    unittest.main()
