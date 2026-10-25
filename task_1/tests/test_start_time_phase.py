"""Small mathematical checks for the second-phase diagnostic."""
import unittest

import numpy as np

from audit_start_time_phase import (frequency_probabilities, label_phase_table,
                                   quantize, quantizer_phase_sensitivity,
                                   singleton_integer_bins)


class PhaseAuditTests(unittest.TestCase):
    def test_reconstruction_phase_identity(self):
        length = np.repeat(np.array([5, 125, 225, 300, 605]), 101)
        ratio = np.tile(np.arange(101), 5)
        end = np.full(len(length), 24000000, dtype=np.int64)
        start = np.floor(end - length * ratio / 100 - 5).astype(np.int64)
        expected = (-np.ceil(length * ratio / 100).astype(np.int64)) % 5
        np.testing.assert_array_equal(start % 5, expected)

    def test_recency_keeps_start_phase(self):
        start = np.array([101, 102, 103, 104, 105])
        last_end = np.array([90, 95, 95, 100, 100])
        np.testing.assert_array_equal((start - last_end) % 5, start % 5)

    def test_quantizer_strict_border_and_singleton(self):
        borders = np.array([-1e30, 1.5, 2.5, 3.5, 8.0])
        np.testing.assert_array_equal(quantize([1.5, 2, 2.5, 8, 9], borders), [1, 2, 2, 4, 5])
        bins = singleton_integer_bins(borders)
        self.assertTrue(bins.loc[2, 'singleton_integer_second'])
        self.assertEqual(bins.loc[2, 'first_nonnegative_integer'], 2)
        self.assertEqual(bins.loc[4, 'integer_count'], 5)

    def test_quantizer_phase_sensitivity(self):
        result = quantizer_phase_sensitivity([1, 3, 7, 12, np.nan], [1.5, 2.5, 8.5])
        np.testing.assert_array_equal(result, [True, True, True, False, False])

    def test_quantizer_days_rounding(self):
        borders = np.asarray([1.5 / 86400, 2.5 / 86400, 8.5 / 86400], dtype='float32')
        result = quantizer_phase_sensitivity([1, 3, 7, 12, np.nan], borders, unit='days')
        np.testing.assert_array_equal(result, [True, True, True, False, False])

    def test_lookup_uses_train_counts_and_prior_for_unseen(self):
        probabilities, unseen, keys = frequency_probabilities(
            [10, 10, 20], [0, 0, 1], [10, 20, 30])
        np.testing.assert_allclose(probabilities[0], [3/5, 1/5, 1/5])
        np.testing.assert_allclose(probabilities[1], [1/4, 2/4, 1/4])
        np.testing.assert_allclose(probabilities[2], [3/6, 2/6, 1/6])
        self.assertEqual((unseen, keys), (1, 2))

    def test_phase_table_counts(self):
        table = label_phase_table([0, 1, 0, 2], [0, 0, 1, 1], 'train', 'start')
        self.assertEqual(table.loc[0, 'rows'], 2)
        self.assertEqual(table.loc[1, 'label_0_share'], 0.5)
        self.assertEqual(table.loc[4, 'rows'], 0)


if __name__ == '__main__':
    unittest.main()
