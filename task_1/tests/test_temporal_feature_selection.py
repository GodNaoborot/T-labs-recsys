"""Маленькие проверки top-k: контракт, важности и полное переобучение."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from temporal_feature_selection import (select_top_k, choose_best_candidate,
                                         _rank_importance, _k_grid)


class ImportanceProbe:
    def get_feature_importance(self, **kwargs):
        self.arguments = kwargs
        return np.array([-1., 2., 2.])


class TemporalSelectionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='temporal-selection-test-')
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / 'selection'
        self.train = np.asfortranarray(np.array([
            [0, 1, 3], [1, 0, 2], [2, 1, 1],
            [2, 0, 1], [0, 1, 3], [1, 0, 2],
            [1, 0, 2], [2, 1, 1], [0, 0, 3]], dtype='float32'))
        self.val = self.train.copy(order='F')
        self.targets = {'label': self.train[:, 0].astype('int8'),
                        'group_id': np.repeat([10, 20, 30], 3).astype('int64'),
                        'log_id': np.arange(9)}
        self.val_targets = {**self.targets, 'group_id': self.targets['group_id'] + 100,
                            'log_id': np.arange(100, 109)}
        self.names = ['signal', 'noise', 'inverse_signal']

    def select(self, **kwargs):
        return select_top_k(self.train, self.val, self.targets, self.val_targets,
                            self.names, self.output, k_values=kwargs.pop('k_values', [1, 2]),
                            iterations=2, threads=1, **kwargs)

    def test_importance_is_train_loss_change_and_ties_use_name(self):
        model, pool = ImportanceProbe(), object()
        order, table = _rank_importance(model, pool, ['negative', 'z', 'a'], 1)
        self.assertIs(model.arguments['data'], pool)
        self.assertEqual(model.arguments['type'], 'LossFunctionChange')
        np.testing.assert_array_equal(order, [2, 1, 0])
        self.assertEqual(table.feature.tolist(), ['a', 'z', 'negative'])
        self.assertEqual(table.importance.tolist(), [2., 2., -1.])

    def test_tied_quality_prefers_smaller_k(self):
        records = [{'k': 100, 'metrics': {'NDCG@10': .75}},
                   {'k': 10, 'metrics': {'NDCG@10': .75 - 5e-13}},
                   {'k': 1, 'metrics': {'NDCG@10': .74}}]
        self.assertEqual(choose_best_candidate(records)['k'], 10)
        self.assertEqual(_k_grid([1, 3, 100], 3), [1, 3])

    def test_invalid_k_and_kind_do_not_create_artifacts(self):
        for values in [[], [0], [-1], [1.2], [True]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.select(k_values=values)
        with self.assertRaises(ValueError):
            self.select(model_kind='pointwise')
        self.assertFalse(self.output.exists())

    def test_invalid_alignment_and_constant_groups_are_rejected(self):
        cases = [('group_id', np.array([10, 20, 10, 20, 20, 20, 30, 30, 30])),
                 ('label', np.array([0, 0, 0, 2, 0, 1, 1, 2, 0])),
                 ('log_id', np.ones(9, dtype='int64'))]
        for key, values in cases:
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_top_k(self.train, self.val, {**self.targets, key: values},
                             self.val_targets, self.names, self.output, k_values=[1])
        with self.assertRaises(ValueError):
            select_top_k(self.train[:-1], self.val, self.targets, self.val_targets,
                         self.names, self.output, k_values=[1])
        self.assertFalse(self.output.exists())

    def test_constant_or_infinite_features_and_overlapping_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            select_top_k(np.ones_like(self.train), self.val, self.targets,
                         self.val_targets, self.names, self.output, k_values=[1])
        self.train[0, 1] = np.inf
        with self.assertRaises(ValueError):
            self.select()
        self.train[0, 1] = 1
        with self.assertRaises(ValueError):
            select_top_k(self.train, self.val, self.targets, self.targets,
                         self.names, self.output, k_values=[1])

    def test_real_tiny_models_are_refit_saved_and_reloaded(self):
        summary = self.select()
        self.assertTrue(summary['completed'])
        self.assertIn(summary['selected_k'], [1, 2, 3])
        curve = pd.read_csv(self.output / 'k_curve.csv')
        self.assertEqual(curve.k.tolist(), [1, 2, 3])
        self.assertEqual(len(set(curve.model_path)), 3)
        self.assertTrue(all(Path(path).exists() for path in curve.model_path))
        protocol = json.loads((self.output / 'protocol.json').read_text())
        self.assertEqual(protocol['importance_split'], 'train')
        self.assertEqual(protocol['importance_type'], 'LossFunctionChange')
        self.assertFalse(protocol['test_used'])
        for subfolder in ['all_features', 'top_1', 'top_2']:
            metrics = json.loads((self.output / 'models' / subfolder / 'metrics.json').read_text())
            self.assertTrue(metrics['reload_predictions_exact'])
            self.assertEqual(len(metrics['features']), metrics['k'])
            self.assertEqual(len(metrics['indices']), metrics['k'])
        saved = (self.output / 'selected_features.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.select()
        self.assertEqual(saved, (self.output / 'selected_features.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
