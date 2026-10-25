"""Проверки старых/новых порогов и констант augmented Pool без больших данных."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from catboost import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import controls_quantization as controls


class ControlsQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='controls-quantization-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        self.train = np.asfortranarray(rng.normal(size=(600, 49)).astype('float32'))
        self.val = np.asfortranarray(rng.normal(size=(120, 49)).astype('float32'))
        self.names = [f'base_{i}' for i in range(42)] + [f'session_{i}' for i in range(7)]
        self.train[:, 0] = 5
        self.train[:, 2] = np.nan
        self.train[:, 47] = np.nan
        self.train[:, 48] = 7
        self.train[::11, 1] = np.nan
        # Validation меняет все train-константы: их запрещено квантовать заново.
        self.val[:, [0, 2, 47, 48]] = np.arange(len(self.val))[:, None]
        self.save_inputs()
        self.base_borders = self.root / 'base_borders.tsv'
        base = Pool(self.train[:, :42], feature_names=self.names[:42], thread_count=1)
        base.quantize(border_count=8, nan_mode='Min', random_seed=42)
        base.save_quantization_borders(str(self.base_borders))

    def save_inputs(self):
        for split, matrix in [('train', self.train), ('val', self.val)]:
            np.save(self.root / (split + '.npy'), matrix)
            rows = len(matrix)
            np.savez_compressed(self.root / (split + '_targets.npz'),
                label=np.arange(rows, dtype='int8') % 3,
                group_id=np.arange(rows, dtype='int64') // 6,
                log_id=np.arange(rows, dtype='int64'))

    def prepare(self, split):
        return controls.prepare_augmented_pool(
            self.root / (split + '.npy'), self.root / (split + '_targets.npz'),
            self.names, self.root / 'pools', split,
            base_borders=self.base_borders if split == 'train' else None)

    def test_base_borders_preserved_added_borders_learned_and_constants_frozen(self):
        train = self.prepare('train')
        text = Path(train['borders_path']).read_text()
        base_text = ''.join(line + '\n' for line in text.splitlines()
                            if int(line.split('\t')[0]) < 42)
        self.assertEqual(base_text, self.base_borders.read_text())
        for feature in range(42, 47):
            count = sum(int(line.split('\t')[0]) == feature for line in text.splitlines())
            self.assertGreater(count, 32)  # Partial input_borders ошибочно дает только 32.
            self.assertLessEqual(count, 254)
        # Для all-NaN базовой колонки native сохраняет исходный NaN-порог.
        self.assertEqual(train['ignored_features'], [0, 47, 48])
        validation = self.prepare('val')
        self.assertEqual(validation['borders_sha256'], train['borders_sha256'])
        self.assertEqual(validation['ignored_features'], train['ignored_features'])
        for split, report in [('train', train), ('val', validation)]:
            self.assertEqual(Path(report['borders_path']).read_bytes(),
                             (self.root / 'pools' / (split + '_applied_borders.tsv')).read_bytes())
            pool = Pool('quantized://' + report['pool_path'], thread_count=1)
            self.assertEqual(pool.get_feature_names(), self.names)
            self.assertEqual(pool.shape, (len(getattr(self, split)), 49))
            targets = controls.load_targets(self.root / (split + '_targets.npz'))
            controls._check_pool(pool, targets, self.names)
        self.assertEqual(self.prepare('train'), train)
        self.assertEqual(self.prepare('val'), validation)

    def test_all_session_constants_do_not_learn_validation_borders(self):
        self.train[:, 42:] = 3
        self.save_inputs()
        train = self.prepare('train')
        validation = self.prepare('val')
        self.assertTrue(set(range(42, 49)).issubset(train['ignored_features']))
        self.assertEqual(train['borders_sha256'], validation['borders_sha256'])
        self.assertEqual((self.root / 'pools/session_train_borders.tsv').read_text(), '')

    def test_validation_requires_train_and_changed_request_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, 'train'):
            self.prepare('val')
        self.prepare('train')
        changed_names = list(self.names)
        changed_names[-1] = 'changed'
        with self.assertRaisesRegex(ValueError, 'изменились'):
            controls.prepare_augmented_pool(
                self.root / 'train.npy', self.root / 'train_targets.npz', changed_names,
                self.root / 'pools', 'train', self.base_borders)

    def test_modified_completed_pool_is_not_reused(self):
        report = self.prepare('train')
        with Path(report['pool_path']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'артефакт'):
            self.prepare('train')


if __name__ == '__main__':
    unittest.main()
