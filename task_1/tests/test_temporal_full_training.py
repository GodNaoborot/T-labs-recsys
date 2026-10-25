"""Проверка потокового чтения без большой матрицы и неизменности train-порогов."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from catboost import CatBoostRanker, Pool
from catboost.utils import quantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from temporal_full_training import (
    _quantize_file, fit_ranker, load_targets, prepare_quantized_split, write_numeric_dataset,
)


class FullTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        rng = np.random.default_rng(42)
        self.matrix = rng.normal(size=(2000, 462)).astype('float32')
        self.matrix[::11, ::7] = np.nan
        self.names = [f'feature_{index}' for index in range(462)]
        self.targets = {'label': (np.arange(2000) % 3).astype('int8'),
                        'group_id': (np.arange(2000) // 10 * 1000 + 270).astype('int64'),
                        'log_id': np.arange(2000, dtype='int64')}
        self.matrix_path = self.folder / 'matrix.npy'
        np.save(self.matrix_path, np.asfortranarray(self.matrix))
        self.targets_path = self.folder / 'targets.npz'
        np.savez(self.targets_path, **self.targets)

    def tearDown(self):
        self.temporary.cleanup()

    def test_single_native_read_matches_public_quantize_and_numpy(self):
        columns = self.folder / 'columns.tsv'
        columns.write_text('0\tLabel\n1\tGroupId\n' +
                           ''.join(f'{i + 2}\tNum\t{name}\n' for i, name in enumerate(self.names)))
        dsv = self.folder / 'matrix.tsv'
        write_numeric_dataset(self.matrix_path, self.targets, self.names, dsv, 'train', batch_rows=127)
        streamed = _quantize_file(dsv, columns)
        public = quantize(str(dsv), column_description=str(columns), thread_count=4,
                          border_count=254, random_seed=42, nan_mode='Min',
                          task_type='CPU', used_ram_limit='3GB', dev_block_size=8192)
        streamed.save_quantization_borders(str(self.folder / 'streamed_borders.tsv'))
        public.save_quantization_borders(str(self.folder / 'public_borders.tsv'))
        self.assertEqual((self.folder / 'streamed_borders.tsv').read_bytes(),
                         (self.folder / 'public_borders.tsv').read_bytes())
        raw = Pool(self.matrix, label=self.targets['label'], group_id=self.targets['group_id'],
                   feature_names=self.names, thread_count=4)
        raw.quantize(input_borders=str(self.folder / 'streamed_borders.tsv'), nan_mode='Min')
        model = CatBoostRanker(iterations=12, depth=4, random_seed=42, thread_count=4,
                               verbose=False, allow_writing_files=False, loss_function='YetiRank').fit(raw)
        np.testing.assert_array_equal(model.predict(raw), model.predict(streamed))
        np.testing.assert_array_equal(model.predict(public), model.predict(streamed))
        np.testing.assert_array_equal(raw.get_group_id_hash(), streamed.get_group_id_hash())
        self.assertEqual(streamed.get_feature_names(), self.names)

    def test_validation_reuses_train_borders_and_saved_model(self):
        train = prepare_quantized_split(self.matrix_path, self.targets_path, self.names,
                                         self.folder / 'pools', 'train')
        borders = Path(train['borders_path']).read_bytes()
        # Validation содержит значения вне диапазона train: новые пороги учить нельзя.
        val_matrix = self.matrix.copy()
        val_matrix[np.isfinite(val_matrix)] *= np.float32(100)
        val_path = self.folder / 'val_matrix.npy'
        np.save(val_path, np.asfortranarray(val_matrix))
        val = prepare_quantized_split(val_path, self.targets_path, self.names,
                                       self.folder / 'pools', 'val', train['borders_path'])
        self.assertEqual(borders, Path(train['borders_path']).read_bytes())
        self.assertEqual(train['borders_sha256'], val['borders_sha256'])
        self.assertFalse((self.folder / 'pools/train.tsv').exists())
        self.assertFalse((self.folder / 'pools/val.tsv').exists())
        parameters = dict(iterations=8, early_stopping_rounds=3, depth=4, learning_rate=.1,
                          random_seed=42, loss_function='YetiRank',
                          eval_metric='NDCG:top=10;type=Base;denominator=LogPosition',
                          use_best_model=True, thread_count=4, verbose=False,
                          allow_writing_files=False)
        result = fit_ranker(train['pool_path'], val['pool_path'], self.targets_path,
                            self.names, self.folder / 'model', parameters)
        self.assertTrue(result['evaluation']['reload_predictions_exact'])
        self.assertTrue(0 <= result['evaluation']['metrics']['NDCG@10'] <= 1)
        self.assertGreater(result['fit_seconds'], 0)

    def test_interrupted_export_can_restart_only_with_matching_request(self):
        def interrupt(event):
            raise RuntimeError('simulate interruption')
        folder = self.folder / 'resume'
        with self.assertRaisesRegex(RuntimeError, 'simulate interruption'):
            prepare_quantized_split(self.matrix_path, self.targets_path, self.names,
                                     folder, 'train', progress=interrupt)
        report = prepare_quantized_split(self.matrix_path, self.targets_path, self.names,
                                          folder, 'train')
        reused = prepare_quantized_split(self.matrix_path, self.targets_path, self.names,
                                          folder, 'train')
        self.assertEqual(report, reused)
        self.assertFalse((folder / 'train.tsv').exists())
        with self.assertRaisesRegex(ValueError, 'другие входы'):
            prepare_quantized_split(self.matrix_path, self.targets_path, self.names[::-1],
                                     folder, 'train')

    def test_bad_inputs_are_rejected_before_large_reads(self):
        with self.assertRaisesRegex(ValueError, 'input_borders'):
            prepare_quantized_split(self.matrix_path, self.targets_path, self.names,
                                     self.folder / 'pools', 'val')
        with self.assertRaisesRegex(ValueError, 'уникальными'):
            write_numeric_dataset(self.matrix_path, self.targets, ['same', 'same'],
                                  self.folder / 'bad.tsv', 'train')
        broken = self.targets.copy()
        broken['group_id'] = broken['group_id'][::-1]
        path = self.folder / 'bad_targets.npz'
        np.savez(path, **broken)
        with self.assertRaisesRegex(ValueError, 'group_id'):
            load_targets(path)


if __name__ == '__main__':
    unittest.main()
