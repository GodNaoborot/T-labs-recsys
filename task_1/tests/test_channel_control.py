"""Безопасный план и сопоставимость контроля текущего канала без fit."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_channel_control as module
from notebook_workflow import NotebookWorkflow


class ChannelControlTests(unittest.TestCase):
    def test_default_only_prints_plan_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, 'TASK_ROOT', Path(directory)), \
                patch.object(module, 'fit_control') as fit, \
                patch.object(module, 'evaluate_control') as evaluate, \
                redirect_stdout(StringIO()) as stream:
            module.main([])
            report = json.loads(stream.getvalue())
            self.assertFalse(report['executed'])
            self.assertEqual(list(Path(directory).iterdir()), [])
        fit.assert_not_called()
        evaluate.assert_not_called()

    def test_mask_excludes_only_current_channel_by_name(self):
        names = ['Users__MEAN(Interactions.is_organic)', 'duration', 'is_organic']
        parameters = module.channel_parameters(names, {'random_seed': 42})
        self.assertEqual(parameters, {'random_seed': 42, 'ignored_features': [2]})
        self.assertEqual(names[0], 'Users__MEAN(Interactions.is_organic)')
        for names, parameters in [(['duration'], {}), (['is_organic', 'is_organic'], {}),
                                  (['is_organic'], {'ignored_features': [0]})]:
            with self.subTest(names=names, parameters=parameters), self.assertRaises(ValueError):
                module.channel_parameters(names, parameters)

    def test_evaluate_requires_explicit_test_flag(self):
        with patch.object(module, 'evaluate_control') as evaluate, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                module.main(['--stage', 'evaluate'])
        evaluate.assert_not_called()

    def test_fit_refuses_existing_output_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'TASK_ROOT', Path(directory)), \
                patch.object(module, 'fit_ranker') as fit:
            (Path(directory) / 'results/c4/existing').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                module.fit_control('source', 'existing')
        fit.assert_not_called()

    def test_matched_bootstrap_rejects_different_groups(self):
        frame = pd.DataFrame({'group_id': [1, 2, 3, 4], 'day': [1, 2, 3, 4],
                              'group_size': [2] * 4, 'ndcg': [.5] * 4})
        reference = frame.copy()
        candidate = frame.assign(ndcg=.6)
        intervals = module.paired_intervals(candidate, reference, 'validation')
        self.assertEqual(intervals.block_days.tolist(), [3, 4])
        self.assertTrue((intervals.delta_ndcg - .1).abs().lt(1e-12).all())
        reference.loc[0, 'group_id'] = 9
        with self.assertRaises(AssertionError):
            module.paired_intervals(candidate, reference, 'validation')

    def test_test_reference_fit_times_come_from_training_without_fit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'fit_ranker') as fit_call:
            folder = Path(directory)
            training = folder / 'training'
            test = folder / 'test'
            training.mkdir()
            test.mkdir()
            rows = pd.DataFrame({'variant': ['C2b-t', 'D462-t'], 'active_features': [49, 462],
                                 'NDCG@10': [.8, .81], 'QueryAUC': [.67, .68],
                                 'tree_count': [900, 1000], 'rows': [10, 10], 'groups': [2, 2]})
            rows.to_csv(test / 'comparison.csv', index=False)
            rows.assign(fit_seconds=[100., 200.]).to_csv(training / 'comparison.csv', index=False)
            groups = pd.DataFrame({'group_size': [5, 5]})
            fitted = {'fit_seconds': 150., 'evaluation': {'tree_count': 911}}
            result = module.comparison_rows(test, 'test', {'NDCG@10': .82, 'QueryAUC': .69},
                                            groups, fitted, training_dir=training).set_index('variant')
            self.assertEqual(result.loc['C2b-t', 'fit_seconds'], 100.)
            self.assertEqual(result.loc['D462-t', 'fit_seconds'], 200.)
            self.assertEqual(result.loc[module.VARIANT, 'fit_seconds'], 150.)
        fit_call.assert_not_called()

    def test_workflow_uses_fresh_sources_and_explicit_test(self):
        report = NotebookWorkflow()
        self.assertTrue(str(report.path('channel-control')).endswith('/results/c4/no_current_channel'))
        fresh = NotebookWorkflow('recompute', 'fresh')
        self.assertEqual(fresh.dependencies('channel-control'), ['corrected'])
        self.assertEqual(len(fresh.commands('channel-control')), 1)
        self.assertIn('fresh_packet', fresh.commands('channel-control')[0])
        self.assertNotIn('--include-test', fresh.commands('channel-control')[0])
        fresh_test = NotebookWorkflow('recompute', 'fresh', include_test=True)
        self.assertEqual(fresh_test.dependencies('channel-control'), ['corrected', 'corrected-test'])
        commands = fresh_test.commands('channel-control')
        self.assertEqual(len(commands), 2)
        self.assertIn('--include-test', commands[1])
        self.assertIn('fresh_packet_test', commands[1])
        with patch('notebook_workflow.subprocess.run') as process:
            report.ensure('channel-control')
        process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
