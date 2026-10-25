"""Малые end-to-end проверки native qbin merge, val-порогов и resume."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from catboost import Pool, CatBoostRanker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import temporal_column_quantization as column


class ColumnQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='column-quantization-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.train = np.asfortranarray(rng.normal(size=(2000, 5)).astype('float32'))
        self.val = np.asfortranarray(rng.normal(size=(400, 5)).astype('float32'))
        for matrix in [self.train, self.val]:
            matrix[::11, 0] = np.nan
            matrix[:, 2] = 7
            matrix[:, 3] = np.nan
        self.names = ['a', 'b', 'constant', 'all_nan', 'e']
        for split, matrix in [('train', self.train), ('val', self.val)]:
            np.save(self.root / f'{split}.npy', matrix)
            rows = len(matrix)
            np.savez_compressed(self.root / f'{split}_targets.npz',
                label=np.arange(rows) % 3, group_id=np.arange(rows) // 4,
                log_id=np.arange(rows, dtype='int64'))

    def prepare(self, split, **kwargs):
        return column.prepare_quantized_split(self.root / f'{split}.npy',
            self.root / f'{split}_targets.npz', self.names,
            self.root / 'pools', split, **kwargs)

    def test_train_merge_matches_native_saved_pool_and_validation_borders(self):
        report = self.prepare('train')
        merged = Pool('quantized://' + report['pool_path'], thread_count=1)
        targets = column.load_targets(self.root / 'train_targets.npz')
        native = Pool(self.train, label=targets['label'], group_id=targets['group_id'],
                      feature_names=self.names, thread_count=1)
        native.quantize(border_count=254, nan_mode='Min', random_seed=42)
        native.save(str(self.root / 'native.qbin'))
        native = Pool('quantized://' + str(self.root / 'native.qbin'), thread_count=1)
        native.save_quantization_borders(str(self.root / 'native.tsv'))
        self.assertEqual(Path(report['borders_path']).read_text(), (self.root / 'native.tsv').read_text())
        np.testing.assert_array_equal(merged.get_group_id_hash(), native.get_group_id_hash())
        parameters = dict(loss_function='YetiRank', iterations=20, depth=3,
                          thread_count=1, random_seed=42, verbose=False, allow_writing_files=False)
        one, two = CatBoostRanker(**parameters), CatBoostRanker(**parameters)
        one.fit(native); two.fit(merged)
        np.testing.assert_array_equal(one.predict(self.val), two.predict(self.val))
        validation = self.prepare('val', input_borders=report['borders_path'])
        self.assertTrue(validation['applied_borders_verified'])
        self.assertEqual(validation['borders_sha256'], report['borders_sha256'])
        self.assertEqual(Pool('quantized://' + validation['pool_path']).shape, (400, 5))
        repeated = self.prepare('train')
        self.assertEqual(repeated, report)

    def test_column_checkpoints_resume_without_requantizing_first(self):
        def interrupt(event):
            if event['rows'] == 2:
                raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.prepare('train', progress=interrupt)
        with patch.object(column, '_quantize_column', wraps=column._quantize_column) as quantize:
            result = self.prepare('train')
        self.assertTrue(result['completed'])
        self.assertEqual(quantize.call_count, 3)
        self.assertEqual([call.args[1] for call in quantize.call_args_list], [2, 3, 4])

    def test_changed_request_and_unknown_catboost_are_rejected(self):
        self.prepare('train')
        with self.assertRaisesRegex(ValueError, 'изменились'):
            self.prepare('train', matrix_sha256='changed')
        with patch.object(column.catboost, '__version__', 'other'):
            with self.assertRaisesRegex(RuntimeError, 'другой версии'):
                self.prepare('train')

    def test_first_constant_and_unseen_validation_values_stay_ignored(self):
        self.names = ['constant', 'b', 'a', 'all_nan', 'e']
        for split, matrix in [('train', self.train), ('val', self.val)]:
            matrix = np.asfortranarray(matrix[:, [2, 1, 0, 3, 4]])
            if split == 'val':
                matrix[:, 0] = np.arange(len(matrix))
            np.save(self.root / f'{split}.npy', matrix)
        train = self.prepare('train')
        val = self.prepare('val', input_borders=train['borders_path'])
        pool = Pool('quantized://' + val['pool_path'])
        self.assertEqual(pool.get_feature_names(), self.names)
        self.assertEqual(val['borders_sha256'], train['borders_sha256'])
        checkpoint = json.loads((self.root / 'pools/val_columns/00000.json').read_text())
        self.assertTrue(checkpoint['training_constant_ignored'])


if __name__ == '__main__':
    unittest.main()
