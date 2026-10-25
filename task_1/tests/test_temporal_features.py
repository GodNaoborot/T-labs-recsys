"""Проверки на маленькой истории: границы времени, холодные ключи и парный bootstrap."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cumulative_features import make_definitions, calculate_cumulative
from block_bootstrap import aggregate_days, circular_day_indices, relative_loss_draws
from project_artifacts import CURRENT_MANIFEST, artifact_path, verify_current_artifacts


class TemporalFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formulas = make_definitions('Users', 'uid', agg_primitives=['count', 'mean'],
                                       trans_primitives=[], max_depth=1, max_features=-1)
        cls.history = pd.DataFrame({
            'log_id': [0, 1, 2, 3], 'uid': [1, 1, 1, 2], 'end_ts': [10, 20, 30, 10],
            'label': [0., 2., 1., 2.], 'track_length_seconds': [100.] * 4,
            'is_organic': [1.] * 4,
        })

    def calculate(self, history, keys, times):
        queries = pd.DataFrame({'log_id': np.arange(len(keys)), 'uid': keys, 'time': times})
        return calculate_cumulative(history, queries, 'Users', 'uid', self.formulas)

    def test_strict_cutoff_and_unknown_user(self):
        values = self.calculate(self.history, [1, 1, 1, 99], [10, 20, 21, 99])
        np.testing.assert_array_equal(values['COUNT(Interactions)'], [0, 1, 2, 0])
        self.assertEqual(values['MEAN(Interactions.label)'].iloc[1], 0.)
        self.assertEqual(values['MEAN(Interactions.label)'].iloc[2], 1.)
        self.assertTrue(np.isnan(values['MEAN(Interactions.label)'].iloc[3]))

    def test_future_outcomes_do_not_change_past_features(self):
        before = self.calculate(self.history, [1], [20])
        changed = self.history.copy()
        changed.loc[changed.end_ts >= 20, 'label'] = 1000.
        pd.testing.assert_frame_equal(before, self.calculate(changed, [1], [20]))

    def test_bootstrap_preserves_pairs(self):
        days = np.repeat(np.arange(15), 2)
        base = np.linspace(.4, .9, len(days))
        daily = aggregate_days(days, base, base * .99)
        for block in (3, 4):
            indices = circular_day_indices(15, block, 200, seed=42)
            np.testing.assert_allclose(relative_loss_draws(daily, indices), .01, atol=1e-14)
            self.assertEqual(indices.shape, (200, 15))

    def test_saved_model_versions(self):
        # Модели не входят в Git; реестр и небольшие отчёты обязательны.
        manifest = json.loads(CURRENT_MANIFEST.read_text())
        missing_models = []
        for method in ('manual', 'dfs'):
            entry = manifest[method]
            for item in (entry['config'], entry['metrics']):
                path = artifact_path(item['path'])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])
            for item in entry['models'].values():
                path = artifact_path(item['path'])
                if not path.is_file():
                    missing_models.append(item['path'])
                else:
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])
        if missing_models:
            self.skipTest('Локальные checkpoints не опубликованы: ' + ', '.join(missing_models))
        result = verify_current_artifacts()
        self.assertFalse(result['test_used'])


if __name__ == '__main__':
    unittest.main()
