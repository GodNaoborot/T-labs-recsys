"""Сверка ускоренных окон DFS с библиотекой и независимыми временными границами."""
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import featuretools as ft
import numpy as np
import pandas as pd
from woodwork.logical_types import Datetime, Double, Integer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from temporal_dfs_features import build_temporal_definitions, calculate_temporal, DAY


def history_fixture():
    """Повторы timestamp, пропуски и несколько сущностей с разной историей."""
    return pd.DataFrame({
        'log_id': np.arange(14),
        'uid': [1] * 10 + [3] * 4,
        'end_ts': np.array([0, 1, 10, 12, 13, 18, 20, 20, 21, 50, 0, 8, 9, 50]) * DAY,
        'label': [0., np.nan, 1., 2., 0., 2., 0., 1., 2., 2., np.nan, 1., 0., 2.],
        'track_length_seconds': [100., 150., 200., 250., 300., np.nan, 50., 100., 50., 80.,
                                 np.nan, 250., 250., 350.],
        'is_organic': [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., np.nan, 1., 0., 0.],
    })


def queries_fixture():
    """Запросы намеренно перемешаны; отсутствующий пользователь и одинаковые cutoff."""
    return pd.DataFrame({'log_id': np.arange(100, 109),
                         'uid': [1, 3, 1, 2, 1, 99, 3, 1, 1],
                         'time': np.array([20, 20, 5, 20, 21, 20, 50, 50, 20]) * DAY})


def reference_matrix(history, queries, features):
    """Вычислить те же Featuretools-деревья native backend с Datetime-временем."""
    child = history.copy()
    child['end_ts'] = pd.Timestamp('2000-01-01') + pd.to_timedelta(child['end_ts'], unit='s')
    parent = pd.DataFrame({'uid': np.sort(queries['uid'].unique())})
    entityset = ft.EntitySet(id='Users')
    entityset.add_dataframe(dataframe_name='Users', dataframe=parent, index='uid',
                            logical_types={'uid': Integer})
    entityset.add_dataframe(
        dataframe_name='Interactions', dataframe=child, index='log_id', time_index='end_ts',
        logical_types={'log_id': Integer, 'uid': Integer, 'end_ts': Datetime,
                       'label': Double, 'track_length_seconds': Double, 'is_organic': Double})
    entityset.add_relationship('Users', 'uid', 'Interactions', 'uid')
    cutoffs = queries[['uid', 'time']].drop_duplicates().copy()
    cutoffs['time'] = pd.Timestamp('2000-01-01') + pd.to_timedelta(cutoffs['time'], unit='s')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='featuretools.*')
        matrix = ft.calculate_feature_matrix(
            features=features, entityset=entityset, cutoff_time=cutoffs,
            cutoff_time_in_index=True, include_cutoff_time=False, n_jobs=1)
    requests = queries[['uid', 'time']].copy()
    requests['time'] = pd.Timestamp('2000-01-01') + pd.to_timedelta(requests['time'], unit='s')
    return requests.merge(matrix.reset_index(), on=['uid', 'time'], how='left',
                          validate='many_to_one')[[feature.get_name() for feature in features]]


class TemporalDFSFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = build_temporal_definitions('Users', 'uid')
        cls.names = [feature.get_name() for feature in cls.features]

    def test_no_arbitrary_formula_limit(self):
        self.assertEqual(len(self.features), 103 * 31 + 2)
        self.assertEqual(len(set(self.names)), len(self.names))
        self.assertIn('COUNT(Interactions, Last 30 Days)', self.names)
        self.assertIn('TIME_SINCE_LAST(Interactions.end_ts, unit=days)', self.names)
        self.assertIn('TIME_SINCE_FIRST(Interactions.end_ts, unit=days)', self.names)

    def test_matches_native_featuretools(self):
        history, queries = history_fixture(), queries_fixture()
        features = build_temporal_definitions('Users', 'uid', windows=[1, 7, 10, 30])
        actual = calculate_temporal(history, queries, 'Users', 'uid', features)
        expected = reference_matrix(history, queries, features)
        np.testing.assert_allclose(actual, expected.to_numpy(dtype=float),
                                   rtol=2e-5, atol=2e-5, equal_nan=True)

    def test_every_window_matches_pandas_and_strict_boundaries(self):
        history, queries = history_fixture(), queries_fixture()
        actual = calculate_temporal(history, queries, 'Users', 'uid', self.features)
        for row, query in queries.iterrows():
            for days in range(1, 31):
                selected = history.loc[(history['uid'] == query['uid'])
                                       & (history['end_ts'] < query['time'])
                                       & (history['end_ts'] >= query['time'] - days * DAY)]
                expected = {
                    f'COUNT(Interactions, Last {days} Days)': len(selected),
                    f'MEAN(Interactions.label, Last {days} Days)': selected['label'].mean(),
                    f'STD(Interactions.label, Last {days} Days)': selected['label'].std(),
                    f'MIN(Interactions.label, Last {days} Days)': selected['label'].min(),
                    f'MAX(Interactions.label, Last {days} Days)': selected['label'].max(),
                }
                for name, value in expected.items():
                    np.testing.assert_allclose(actual[row, self.names.index(name)], value,
                                               rtol=2e-5, atol=2e-5, equal_nan=True)

    def test_future_and_equal_timestamp_labels_do_not_leak(self):
        history, queries = history_fixture(), queries_fixture()
        for cutoff in [5 * DAY, 20 * DAY, 50 * DAY]:
            current = queries.loc[queries['time'] == cutoff]
            actual = calculate_temporal(history, current, 'Users', 'uid', self.features)
            changed = history.copy()
            changed.loc[changed['end_ts'] >= cutoff,
                        ['label', 'track_length_seconds', 'is_organic']] = [999., 9999., 999.]
            again = calculate_temporal(changed, current, 'Users', 'uid', self.features)
            np.testing.assert_array_equal(actual, again)

    def test_selected_columns_memmap_and_missing_key(self):
        history, queries = history_fixture(), queries_fixture()
        queries.loc[len(queries)] = [110, np.nan, 20 * DAY]
        selected = self.features[::97] + self.features[-2:]
        expected = calculate_temporal(history, queries, 'Users', 'uid', selected)
        with tempfile.TemporaryDirectory() as folder:
            output = np.memmap(Path(folder) / 'features.dat', mode='w+', dtype='float32',
                               shape=expected.shape)
            result = calculate_temporal(history, queries, 'Users', 'uid', selected, output=output)
            self.assertIs(result, output)
            np.testing.assert_array_equal(output, expected)
        self.assertTrue(np.isnan(expected[-1]).all())
        whole = calculate_temporal(history, queries, 'Users', 'uid', self.features)
        positions = [self.names.index(feature.get_name()) for feature in selected]
        np.testing.assert_array_equal(expected, whole[:, positions])

    def test_invalid_windows_and_duplicate_queries_are_rejected(self):
        for windows in [[0], [-1], [1.5], [1, 1]]:
            with self.assertRaises(ValueError):
                build_temporal_definitions('Users', 'uid', windows=windows)
        queries = queries_fixture()
        queries.loc[1, 'log_id'] = queries.loc[0, 'log_id']
        with self.assertRaises(ValueError):
            calculate_temporal(history_fixture(), queries, 'Users', 'uid', self.features)


if __name__ == '__main__':
    unittest.main()
