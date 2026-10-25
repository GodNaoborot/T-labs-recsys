"""Скользящий STD постоянного окна не зависит от промежуточных запросов."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from temporal_dfs_features import _rolling_column


class TemporalStabilityTests(unittest.TestCase):
    def test_constant_tail_has_exact_zero_std(self):
        random = np.random.default_rng(0)
        values = np.r_[random.integers(0, 500, 5000), np.repeat(100., 1000)]
        right = np.arange(1, len(values) + 1, dtype='int64')
        left = np.maximum(right - 500, 0)
        starts = np.zeros(len(right), dtype='int64')
        actual = _rolling_column(values, starts, left, right, len(values))
        np.testing.assert_array_equal(actual[-501:, 1], 0.)
        np.testing.assert_array_equal(actual[-501:, 0], 100.)
        sparse = _rolling_column(values, starts[[-1]], left[[-1]], right[[-1]], len(values))
        np.testing.assert_array_equal(actual[-1], sparse[0])

    def test_nonconstant_window_preserves_sample_std(self):
        values = np.array([4., 4., 4., 5., 4., 3.])
        right = np.arange(1, 7, dtype='int64')
        left = np.maximum(right - 3, 0)
        actual = _rolling_column(values, np.zeros(6, dtype='int64'), left, right, len(values))
        for row in range(1, len(values)):
            window = values[left[row]:right[row]]
            self.assertAlmostEqual(actual[row, 1], window.std(ddof=1), places=12)


if __name__ == '__main__':
    unittest.main()
