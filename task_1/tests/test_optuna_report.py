"""Синтетические проверки отчета Optuna без обучения и исходных датасетов."""
from pathlib import Path
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import optuna_report as report
from block_bootstrap import circular_day_indices
from project_artifacts import file_hash


def recorded_models():
    """Подготовить маленькие исходные результаты вместо чтения реальных моделей."""
    models = {}
    for method in ('manual', 'dfs'):
        for objective in ('ranker', 'pointwise'):
            result = {'metrics': {'NDCG@10': .7 if method == 'manual' else .693,
                                  'QueryAUC': .6, 'logloss': None if objective == 'ranker' else .5},
                      'best_iteration': 599, 'tree_count': 600, 'fit_seconds': 10.,
                      'params': {'iterations': 600, 'early_stopping_rounds': 50}}
            models[(method, objective)] = {'status': 'completed', 'result': result}
    return models


def recorded_intervals():
    """Подготовить старые интервалы с уже выбранным знаком относительного прироста."""
    return [{'configuration': 'fixed_600', 'objective': objective, 'block_days': block,
             'relative_gain': -.01, 'ci_lower_relative_gain': -.012,
             'ci_upper_relative_gain': -.008, 'n_resamples': 20000}
            for objective in ('ranker', 'pointwise') for block in (3, 4)]


class OptunaReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='optuna-report-test-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.paths = {'results': self.folder / 'run'}
        self.days = np.repeat(np.arange(15), 2 + np.arange(15) % 3)
        self.base = np.linspace(.4, .8, len(self.days))
        self.patches = [patch.object(report, '_recorded_models', side_effect=recorded_models),
                        patch.object(report, '_recorded_intervals', side_effect=recorded_intervals)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def stage(self, method, values, prefix='control', objective='pointwise'):
        """Сохранить только небольшие групповые значения и метаданные одного fit."""
        path = self.folder / f'{prefix}_{method}_{objective}.parquet'
        pd.DataFrame({'group_id': np.arange(len(values)), 'day': self.days,
                      'group_size': np.arange(len(values)) % 10 + 2, 'ndcg': values}).to_parquet(path, index=False)
        result = {'metrics': {'NDCG@10': float(np.mean(values)), 'QueryAUC': .6},
                  'best_iteration': 1599, 'tree_count': 1600, 'fit_seconds': 11.,
                  'params': {'iterations': 1600, 'early_stopping_rounds': 80},
                  'artifacts': {'groups': str(path)}, 'artifact_sha256': {'groups': file_hash(path)}}
        return {'status': 'completed', 'result': result}

    def stages(self, gain=.02):
        """Создать завершенную пару с постоянным относительным эффектом."""
        return {'control_manual_pointwise': self.stage('manual', self.base),
                'control_dfs_pointwise': self.stage('dfs', self.base * (1 + gain))}

    def test_positive_gain_and_interval_signs(self):
        manifest = report.publish_reports(self.paths, self.stages(.02))
        summary = pd.read_csv(self.paths['results'] / 'comparison/summary.csv')
        selected = summary.query("configuration == 'control_1600' and objective == 'pointwise'").iloc[0]
        self.assertAlmostEqual(selected.relative_gain, .02)
        self.assertGreater(selected.ndcg_difference, 0.)
        for block in (3, 4):
            self.assertAlmostEqual(selected[f'ci95_L{block}_lower_relative_gain'], .02)
            self.assertAlmostEqual(selected[f'ci95_L{block}_upper_relative_gain'], .02)
        self.assertEqual(manifest['pairs']['control_pointwise']['status'], 'completed')
        self.assertFalse(manifest['test_used'])
        self.assertIn('patience', manifest['limitations'][2])
        json.dumps(manifest, allow_nan=False)

    def test_paired_draw_uses_group_weighted_daily_sums(self):
        stages = self.stages()
        values = self.base * (.98 + .002 * self.days)
        stages['control_dfs_pointwise'] = self.stage('dfs', values)
        report.publish_reports(self.paths, stages)
        folder = self.paths['results'] / 'comparison/control_pointwise'
        daily = pd.read_csv(folder / 'daily.csv')
        indices = circular_day_indices(len(daily), 3, 20000, seed=42)
        chosen = indices[0]
        expected = 1 - daily.dfs_sum.to_numpy()[chosen].sum() / daily.base_sum.to_numpy()[chosen].sum()
        with np.load(folder / 'bootstrap_losses.npz') as saved:
            self.assertAlmostEqual(saved['loss_L3'][0], expected, places=14)
        self.assertEqual(int(daily.group_count.sum()), len(self.base))

    def test_missing_and_failed_models_are_explicit(self):
        stages = self.stages()
        stages['control_dfs_pointwise']['status'] = 'failed'
        manifest = report.publish_reports(self.paths, stages)
        self.assertEqual(manifest['pairs']['control_pointwise']['status'], 'incomplete_pair')
        models = pd.read_csv(self.paths['results'] / 'comparison/models.csv')
        failed = models.query("configuration == 'control_1600' and method == 'dfs' and objective == 'pointwise'").iloc[0]
        self.assertEqual(failed.status, 'failed')
        self.assertTrue(pd.isna(failed['NDCG@10']))
        self.assertFalse((self.paths['results'] / 'comparison/control_pointwise/bootstrap_losses.npz').exists())

    def test_mismatched_day_does_not_get_an_interval(self):
        stages = self.stages()
        result = stages['control_dfs_pointwise']['result']
        path = Path(result['artifacts']['groups'])
        groups = pd.read_parquet(path)
        groups.loc[0, 'day'] += 1
        groups.to_parquet(path, index=False)
        result['artifact_sha256']['groups'] = file_hash(path)
        manifest = report.publish_reports(self.paths, stages)
        self.assertEqual(manifest['pairs']['control_pointwise']['status'], 'invalid_pair')
        intervals = pd.read_csv(self.paths['results'] / 'comparison/bootstrap_summary.csv')
        self.assertEqual(set(intervals.configuration), {'fixed_600'})

    def test_cache_is_reused_after_deadline_and_sources_unchanged(self):
        stages = self.stages()
        source_path = Path(stages['control_dfs_pointwise']['result']['artifacts']['groups'])
        source_before = source_path.read_bytes()
        report.publish_reports(self.paths, stages)
        folder = self.paths['results'] / 'comparison/control_pointwise'
        before = (folder / 'bootstrap_losses.npz').read_bytes()
        with patch.object(report, 'circular_day_indices', side_effect=AssertionError('no new draws')):
            manifest = report.publish_reports(self.paths, stages, deadline_unix=time.time() - 1)
        self.assertEqual(manifest['pairs']['control_pointwise']['status'], 'cached')
        self.assertEqual((folder / 'bootstrap_losses.npz').read_bytes(), before)
        self.assertEqual(source_path.read_bytes(), source_before)

    def test_deadline_does_not_start_bootstrap(self):
        with patch.object(report, 'circular_day_indices', side_effect=AssertionError('no draws after deadline')):
            manifest = report.publish_reports(self.paths, self.stages(), deadline_unix=time.time() - 1)
        self.assertEqual(manifest['pairs']['control_pointwise']['status'], 'skipped_deadline')
        self.assertTrue((self.paths['results'] / 'comparison/summary.csv').exists())

    def test_tuned_reuse_of_control_is_visible(self):
        stages = self.stages(-.01)
        for method in ('manual', 'dfs'):
            source = stages[f'control_{method}_pointwise']['result']
            stages[f'tuned_{method}_pointwise'] = {'status': 'completed',
                'result': {**source, 'reused_from': f'control_{method}_pointwise'}}
        report.publish_reports(self.paths, stages)
        summary = pd.read_csv(self.paths['results'] / 'comparison/summary.csv')
        values = summary.query("objective == 'pointwise' and configuration != 'fixed_600'").relative_gain
        np.testing.assert_allclose(values, -.01, rtol=0, atol=1e-14)
        models = pd.read_csv(self.paths['results'] / 'comparison/models.csv')
        reused = models.query("configuration == 'tuned_1600' and objective == 'pointwise'")
        self.assertTrue(reused.reused_from.notna().all())

    def test_modified_source_hash_is_rejected(self):
        stages = self.stages()
        result = stages['control_dfs_pointwise']['result']
        path = Path(result['artifacts']['groups'])
        path.write_bytes(path.read_bytes() + b'changed')
        manifest = report.publish_reports(self.paths, stages)
        self.assertEqual(manifest['pairs']['control_pointwise']['status'], 'invalid_pair')
        self.assertIn('Изменился', manifest['pairs']['control_pointwise']['error'])


class OldIntervalsSignTest(unittest.TestCase):
    def test_loss_interval_endpoints_reverse_for_gain(self):
        frame = pd.DataFrame([{'period': 'original_val', 'objective': 'ranker', 'block_days': 3,
                               'relative_loss': .011, 'ci_lower': .009, 'ci_upper': .012,
                               'upper_bound': .0118, 'n_resamples': 20000, 'source': 'old.npz'}])
        with patch.object(report, 'compare_saved_intervals', return_value=frame):
            row = report._recorded_intervals()[0]
        self.assertEqual(row['relative_gain'], -.011)
        self.assertEqual(row['ci_lower_relative_gain'], -.012)
        self.assertEqual(row['ci_upper_relative_gain'], -.009)


if __name__ == '__main__':
    unittest.main()
