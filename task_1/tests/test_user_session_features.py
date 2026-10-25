"""Малые проверки C2: временные границы, last-N и запись матрицы."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import user_session_features as session


class UserSessionFeaturesTests(unittest.TestCase):
    def history(self, times, ratios, users=None):
        return pd.DataFrame({'uid': users if users is not None else [1] * len(times),
                             'timestamp': times, 'played_ratio_pct': ratios})

    def queries(self, times, users=None, index=None):
        return pd.DataFrame({'uid': users if users is not None else [1] * len(times),
                             'start_ts': times}, index=index)

    def test_strict_boundary_excludes_all_equal_and_future_events(self):
        history = self.history([5, 10, 10, 20], [100, 0, 50, 0])
        queries = self.queries([11, 10, 5])
        actual = session.calculate_session_features(queries, history)
        np.testing.assert_array_equal(actual['user_prev_played_ratio'][:2], [50, 100])
        np.testing.assert_array_equal(actual['user_lst_cnt_1d'], [3, 1, 0])
        np.testing.assert_array_equal(actual['user_time_since_last'][:2], [1, 5])
        self.assertTrue(actual.iloc[2].drop('user_lst_cnt_1d').isna().all())

    def test_day_lower_boundary_is_included_and_upper_is_excluded(self):
        history = self.history([3599, 3600, 89999, 90000, 90001], [0, 100, 20, 0, 0])
        actual = session.calculate_session_features(self.queries([90000]), history).iloc[0]
        self.assertEqual(actual['user_lst_cnt_1d'], 2)
        self.assertEqual(actual['user_skip_rate_1d'], .5)
        self.assertEqual(actual['user_prev_played_ratio'], 20)
        self.assertEqual(actual['user_prev_skip'], 1)
        self.assertEqual(actual['user_time_since_last'], 1)
        self.assertAlmostEqual(actual['user_skip_rate_last5'], 2 / 3)

    def test_last_n_uses_events_and_partial_history(self):
        history = self.history(np.arange(1, 26), [0] * 10 + [100] * 15)
        actual = session.calculate_session_features(self.queries([26, 4]), history)
        np.testing.assert_array_equal(actual['user_skip_rate_last5'], [0, 1])
        np.testing.assert_array_equal(actual['user_skip_rate_last20'], [.25, 1])
        self.assertAlmostEqual(actual.loc[0, 'user_skip_rate_1d'], .4)

    def test_stable_ties_keep_original_log_order_after_time_sort(self):
        history = self.history([10, 5, 10], [20, 100, 80])
        actual = session.calculate_session_features(self.queries([11, 10]), history)
        np.testing.assert_array_equal(actual['user_prev_played_ratio'], [80, 100])
        self.assertAlmostEqual(actual.loc[0, 'user_skip_rate_last5'], 1 / 3)

    def test_query_order_index_users_and_ratio_clipping(self):
        history = self.history([2, 5], [150, 20], users=[1, 2])
        queries = self.queries([6, 3, 7], users=[2, 1, 2], index=[9, 9, 2])
        actual = session.calculate_session_features(queries, history)
        pd.testing.assert_index_equal(actual.index, queries.index)
        np.testing.assert_array_equal(actual['user_prev_played_ratio'], [20, 100, 20])
        np.testing.assert_array_equal(actual['user_prev_skip'], [1, 0, 1])
        np.testing.assert_array_equal(actual['user_time_since_last'], [1, 1, 2])
        self.assertTrue(all(dtype == np.dtype('float32') for dtype in actual.dtypes))

    def test_empty_day_keeps_older_last_event_and_no_history_gives_nan(self):
        history = self.history([1], [100])
        actual = session.calculate_session_features(self.queries([100000, 10], [1, 2]), history)
        self.assertEqual(actual.loc[0, 'user_lst_cnt_1d'], 0)
        self.assertTrue(np.isnan(actual.loc[0, 'user_skip_rate_1d']))
        self.assertEqual(actual.loc[0, 'user_prev_played_ratio'], 100)
        self.assertEqual(actual.loc[0, 'user_skip_rate_last5'], 0)
        self.assertTrue(actual.iloc[1].drop('user_lst_cnt_1d').isna().all())
        empty = session.calculate_session_features(self.queries([10]), self.history([], []))
        self.assertEqual(empty.loc[0, 'user_lst_cnt_1d'], 0)
        self.assertTrue(empty.iloc[0].drop('user_lst_cnt_1d').isna().all())
        self.assertEqual(session.calculate_session_features(self.queries([]), history).shape, (0, 7))

    def test_current_and_future_values_and_query_labels_cannot_change_features(self):
        history = self.history([1, 5, 20, 30], [0, 100, 0, 0])
        queries = self.queries([20]);queries['label'] = 0;queries['played_ratio_pct'] = 0
        expected = session.calculate_session_features(queries, history)
        changed = history.copy();changed.loc[2:, 'played_ratio_pct'] = 100
        queries['label'] = 2;queries['played_ratio_pct'] = 100
        pd.testing.assert_frame_equal(session.calculate_session_features(queries, changed), expected)

    def test_bad_schema_order_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'по uid'):
            session.calculate_session_features(self.queries([10]), self.history([1, 2], [0, 100], [2, 1]))
        with self.assertRaisesRegex(ValueError, 'start_ts'):
            session.calculate_session_features(self.queries([np.nan]), self.history([1], [100]))
        with self.assertRaisesRegex(ValueError, 'Отсутствуют'):
            session.calculate_session_features(pd.DataFrame({'uid': [1]}), self.history([1], [100]))

    def test_writer_matches_direct_is_fortran_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='session-features-test-') as folder:
            root = Path(folder);history_path = root / 'listens.parquet';output = root / 'features.npy'
            history = self.history([1, 5, 5, 10], [120, 0, 100, 0])
            history.to_parquet(history_path, index=False, row_group_size=2)
            queries = self.queries([6, 5, 11])
            report = session.write_session_feature_matrix(queries, history_path, output, batch_size=2)
            matrix = np.load(output, mmap_mode='r')
            np.testing.assert_array_equal(matrix, session.calculate_session_features(queries, history).to_numpy())
            self.assertTrue(matrix.flags.f_contiguous)
            self.assertEqual(matrix.dtype, np.dtype('float32'))
            self.assertEqual(report['rows'], 3)
            self.assertEqual(report['features'], list(session.SESSION_FEATURES))
            self.assertFalse(output.with_name(output.name + '.incomplete').exists())
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                session.write_session_feature_matrix(queries, history_path, output)
            self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
