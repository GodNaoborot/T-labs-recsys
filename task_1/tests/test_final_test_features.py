"""Проверки test-адаптера только на искусственных данных."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import final_test_features as engine


class FinalTestFeatures(unittest.TestCase):
    def test_manual_mapping_preserves_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base = {'uid': [1, 1, 1], 'item_id': [8, 8, 8],
                    'start_ts': [24624000] * 3, 'label': [0, 0, 1],
                    'group_id': [1285] * 3, 'is_organic': [1] * 3,
                    'track_length_seconds': [225] * 3}
            raw = pd.DataFrame(base)
            raw.to_parquet(folder / 'raw.parquet', index=False)
            features = raw.assign(split='test', f0=[10., 20., 30.], f1=[11., 21., 31.])
            irrelevant = features.iloc[[0]].assign(split='val')
            pd.concat([features.iloc[:1], irrelevant, features.iloc[1:]], ignore_index=True).to_parquet(
                folder / 'features.parquet', index=False)
            order = np.array([2, 0, 1])
            queries = raw.iloc[order].reset_index(drop=True).rename(columns={'start_ts': 'time'})
            queries['source_position'] = order
            outputs = [np.lib.format.open_memmap(folder / (name + '.npy'), mode='w+',
                shape=(3, 2), dtype='float32', fortran_order=True) for name in ['a', 'b']]
            record = engine._manual_columns(outputs, ['f0', 'f1'], queries,
                folder / 'raw.parquet', folder / 'features.parquet')
            for values in outputs:
                np.testing.assert_array_equal(values, [[30, 31], [10, 11], [20, 21]])
            self.assertEqual(record['all_split_rows'], 3)
            self.assertEqual(record['retained_rows'], 3)

    def test_manual_alignment_checks_removed_rows_too(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = pd.DataFrame({'uid': [1, 2], 'item_id': [8, 9],
                'start_ts': [24624000] * 2, 'label': [0, 1], 'group_id': [1285, 2285],
                'is_organic': [1, 1], 'track_length_seconds': [225, 300]})
            raw.to_parquet(folder / 'raw.parquet', index=False)
            features = raw.assign(split='test', f0=[10., 20.])
            features.loc[1, 'item_id'] = 999
            features.to_parquet(folder / 'features.parquet', index=False)
            queries = raw.iloc[:1].rename(columns={'start_ts': 'time'}).copy()
            queries['source_position'] = [0]
            output = np.lib.format.open_memmap(folder / 'out.npy', mode='w+', shape=(1, 1),
                dtype='float32', fortran_order=True)
            with self.assertRaises(AssertionError):
                engine._manual_columns([output], ['f0'], queries,
                    folder / 'raw.parquet', folder / 'features.parquet')

    def test_quantization_uses_complete_train_borders_and_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data, results = folder / 'data', folder / 'results'
            schema = data / 'manual42'
            schema.mkdir(parents=True)
            results.mkdir()
            names = ['f0', 'f1', 'f2']
            train_values = np.array([[-2, 1, np.nan], [0, 1, 0], [1, 1, 2],
                                    [2, 1, np.nan], [3, 1, 3], [4, 1, 4]], dtype='float32')
            train = Pool(train_values, feature_names=names)
            train.quantize(border_count=4, nan_mode='Min', random_seed=42)
            borders = folder / 'frozen_train_borders.tsv'
            train.save_quantization_borders(str(borders))
            labels = np.array([0, 1, 2, 0, 1, 2], dtype='int8')
            groups = np.array([1000] * 3 + [2000] * 3, dtype='int64')
            ids = np.arange(6, dtype='int64')
            np.savez_compressed(data / 'targets.npz', label=labels, group_id=groups,
                                log_id=ids, day=groups % 1000)
            values = np.lib.format.open_memmap(schema / 'test_X.npy', mode='w+',
                shape=(6, 3), dtype='float32', fortran_order=True)
            values[:] = train_values
            values[:, 1] = np.arange(6) + 100
            values.flush()
            del values
            sha = hashlib.sha256(borders.read_bytes()).hexdigest()
            plan = {'schemas': {'manual42': names},
                    'train_borders': {'manual42': {'path': str(borders), 'sha256': sha}}}
            with patch.object(engine, '_column_record', wraps=engine._column_record) as tracked:
                report = engine.quantize_schema(data, results, plan, 'manual42')
                self.assertEqual(tracked.call_count, 3)
                for call in tracked.call_args_list:
                    self.assertIsInstance(call.args[-1], list)
            self.assertTrue(report['completed'])
            self.assertFalse(report['training_performed'])
            self.assertEqual((schema / 'test_applied_borders.tsv').read_bytes(), borders.read_bytes())
            restored = Pool('quantized://' + str(schema / 'test.qbin'))
            self.assertEqual(restored.get_feature_names(), names)
            np.testing.assert_array_equal(np.asarray(restored.get_label(), dtype='float64'), labels)
            with np.load(schema / 'test_targets.npz') as saved:
                for key, expected in [('label', labels), ('group_id', groups), ('log_id', ids), ('day', groups % 1000)]:
                    np.testing.assert_array_equal(saved[key], expected)
            repeated = engine.quantize_schema(data, results, plan, 'manual42')
            self.assertEqual(repeated['pool_sha256'], report['pool_sha256'])
            borders.write_text(borders.read_text() + '0\t999\n')
            with self.assertRaises(ValueError):
                engine.quantize_schema(data, results, plan, 'manual42')


if __name__ == '__main__':
    unittest.main()
