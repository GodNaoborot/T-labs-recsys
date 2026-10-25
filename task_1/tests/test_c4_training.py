"""Маленькие проверки новых порогов, общего бюджета и парных val-отчётов C4."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import c4_training as c4


def targets(log_start=0, first_day=240, groups=8):
    days = np.repeat(first_day + np.arange(groups) % 4, 3)
    group_id = np.repeat(np.arange(groups) * 1000, 3) + days
    return {'label': np.tile(np.array([0, 1, 2], dtype='int8'), groups),
            'group_id': group_id.astype('int64'),
            'log_id': np.arange(log_start, log_start + groups * 3, dtype='int64'),
            'day': days.astype('int64')}


class C4TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_parameters_match_controls_and_only_manual_mask_differs(self):
        baseline = c4.variant_parameters('M-t')
        augmented = c4.variant_parameters('C2b-t')
        dfs = c4.variant_parameters('D462-t')
        self.assertEqual(baseline.pop('ignored_features'), list(range(42, 49)))
        self.assertEqual(baseline, augmented)
        self.assertEqual(augmented, dfs)
        self.assertEqual((dfs['iterations'], dfs['early_stopping_rounds'],
                          dfs['depth'], dfs['thread_count']), (1600, 80, 6, 4))
        self.assertEqual(dfs['loss_function'], 'YetiRank')
        self.assertEqual(dfs['l2_leaf_reg'], 1e-20)

    def test_targets_reject_wrong_cutoff_day_and_constant_groups(self):
        path = self.folder / 'targets.npz'
        arrays = targets()
        np.savez(path, **arrays)
        np.testing.assert_array_equal(c4.load_targets(path)['label'], arrays['label'])
        arrays['day'][0] += 1
        np.savez(path, **arrays)
        with self.assertRaises(AssertionError):
            c4.load_targets(path)
        arrays = targets()
        arrays['label'][:3] = 1
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'label'):
            c4.load_targets(path)

    def test_new_train_borders_handle_validation_drift_and_train_constant(self):
        data, output = self.folder / 'data', self.folder / 'result'
        schema = data / 'manual49'
        schema.mkdir(parents=True)
        names = ['feature_' + str(i) for i in range(49)]
        (schema / 'features.json').write_text(json.dumps({'features': names,
            'feature_count': 49, 'dtype': 'float32', 'layout': 'F-order'}))
        rng = np.random.default_rng(42)
        for split, ids, day in [('train', 0, 240), ('val', 100, 270)]:
            arrays = targets(ids, day)
            np.savez(data / (split + '_targets.npz'), **arrays)
            matrix = rng.normal(size=(len(arrays['label']), 49)).astype('float32')
            matrix[:, 1] = 7 if split == 'train' else np.arange(len(matrix))
            if split == 'val':
                matrix[:, 0] += 10000
            np.save(schema / (split + '_X.npy'), np.asfortranarray(matrix))
        with patch.object(c4, 'COUNTS', {'manual49': 49}), patch.dict(
                c4.PARAMETERS, {'iterations': 8, 'early_stopping_rounds': 3, 'verbose': False}):
            reports = c4.prepare_pools(data, output)
            # Повторное чтение готовых артефактов не учит новые пороги.
            repeated = c4.prepare_pools(data, output)
        self.assertEqual(reports, repeated)
        train, val = reports['manual49']['train'], reports['manual49']['val']
        self.assertEqual(train['borders_sha256'], val['borders_sha256'])
        borders = Path(train['borders_path']).read_text()
        self.assertFalse(any(line.startswith('1\t') for line in borders.splitlines()))
        self.assertEqual(Path(train['borders_path']).read_bytes(),
                         (output / 'quantized/manual49/val_applied_borders.tsv').read_bytes())
        self.assertEqual(train['quantization']['borders_origin'], 'c4_common_selected_train')
        self.assertFalse(val['test_used'])
        frozen = json.loads((output / 'training_plan.json').read_text())
        self.assertEqual(frozen['inputs']['val_targets']['rows'], 24)
        self.assertEqual(frozen['schemas']['manual49'], names)
        fitted = c4.fit_variant(data, output, 'M-t')
        self.assertTrue(fitted['completed'])
        self.assertEqual(fitted['active_features'], 42)
        self.assertEqual(fitted['ignored_indices'], list(range(42, 49)))
        self.assertTrue((output / 'models/M-t/model.cbm').is_file())
        self.assertLessEqual(fitted['evaluation']['tree_count'], 8)
        self.assertFalse(fitted['test_used'])

    def test_fit_worker_hard_timeout_stops_own_process(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(c4.subprocess, 'Popen', return_value=process) as launched, \
                patch.object(c4, '_stop_worker') as stopped:
            with self.assertRaises(TimeoutError):
                c4.run_fit_worker(self.folder, self.folder, 'M-t', .02)
        self.assertTrue(launched.call_args.kwargs['start_new_session'])
        stopped.assert_called_once_with(process)
        self.assertTrue((self.folder / 'logs/fit_M-t.log').is_file())

    def test_total_budget_is_shared_and_cannot_reset(self):
        seen = []

        def fake_worker(data, output, variant, remaining, progress=None):
            seen.append((variant, remaining))
            if variant == 'C2b-t':
                raise TimeoutError('budget')
            return .01

        with patch.object(c4, 'prepare_pools'), patch.object(c4, 'run_fit_worker', fake_worker):
            with self.assertRaises(TimeoutError):
                c4.run_training(self.folder, self.folder, max_training_seconds=1)
            with self.assertRaises(FileExistsError):
                c4.run_training(self.folder, self.folder, max_training_seconds=1)
        self.assertEqual([value[0] for value in seen], ['M-t', 'C2b-t'])
        self.assertGreater(seen[0][1], seen[1][1])
        stopped = json.loads((self.folder / 'training_stopped.json').read_text())
        self.assertEqual(stopped['completed_variants'], ['M-t'])
        self.assertFalse(stopped['test_used'])
        with self.assertRaises(ValueError):
            c4.run_training(self.folder, self.folder, max_training_seconds=7201)

    def test_summary_uses_three_paired_differences_and_internal_empty_day(self):
        plan = {'bootstrap': {'blocks_days': [3, 4], 'replicates': 20000, 'seed': 42}}
        (self.folder / 'training_plan.json').write_text(json.dumps(plan))
        plan_hash = c4._sha256(self.folder / 'training_plan.json')
        days = np.array([270, 270, 271, 273, 274], dtype='int64')
        sizes = np.array([3, 5, 2, 4, 9], dtype='int64')
        baseline = np.array([.3, .6, .5, .7, .6])
        for variant, delta, count in [('M-t', 0., 42), ('C2b-t', .02, 49), ('D462-t', .01, 462)]:
            folder = self.folder / 'models' / variant
            folder.mkdir(parents=True)
            model_path = folder / 'model.cbm'
            model_path.write_bytes(b'fixed-model-' + variant.encode())
            frame = pd.DataFrame({'group_id': np.arange(5) * 1000 + days,
                                  'day': days, 'group_size': sizes, 'ndcg': baseline + delta})
            frame.to_parquet(folder / 'group_ndcg.parquet', index=False)
            record = {'completed': True, 'training_plan_sha256': plan_hash,
                      'model_sha256': c4._sha256(model_path), 'active_features': count,
                      'physical_features': 462 if count == 462 else 49, 'fit_seconds': 20.,
                      'evaluation': {'metrics': {'NDCG@10': float((baseline + delta).mean()),
                                                 'QueryAUC': .6}, 'tree_count': 12}}
            (folder / 'control.json').write_text(json.dumps(record))
        result = c4.summarize(self.folder)
        self.assertEqual(result['calendar_days'], 5)
        self.assertEqual(len(result['comparisons']), 6)
        for row in result['comparisons']:
            expected = .01 if row['reference'] == 'D462-t' or row['candidate'] == 'D462-t' else .02
            self.assertAlmostEqual(row['delta_ndcg'], expected)
            self.assertAlmostEqual(row['ci_low'], expected)
            self.assertAlmostEqual(row['ci_high'], expected)
        daily = pd.read_csv(self.folder / 'daily_C2b-t_minus_M-t.csv')
        self.assertEqual(daily['day'].tolist(), list(range(270, 275)))
        self.assertEqual(daily.loc[daily['day'] == 272, 'group_count'].item(), 0)
        self.assertFalse(result['test_used'])


if __name__ == '__main__':
    unittest.main()
