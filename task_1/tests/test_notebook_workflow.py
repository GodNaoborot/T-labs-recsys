"""Свежие notebook-маршруты не должны подмешивать сохранённые результаты."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import notebook_workflow as module
from run_all_experiments import ordered_stages


class WorkflowTests(unittest.TestCase):
    def test_report_does_not_start_commands(self):
        with patch.object(module.subprocess, 'run') as command:
            module.NotebookWorkflow().ensure('corrected-test')
        command.assert_not_called()

    def test_new_run_links_sources_and_outputs(self):
        workflow = module.NotebookWorkflow('recompute', 'fresh')
        full = workflow.commands('full')[0]
        self.assertEqual(full[full.index('--selection-run') + 1], 'fresh_used')
        self.assertEqual(full[full.index('--definitions-run') + 1], 'fresh_topk')
        used = workflow.commands('used')[0]
        self.assertEqual(used[used.index('--source-run') + 1], 'fresh_topk')
        self.assertIn('rolling_windows_runs/fresh_windows', str(workflow.path('rolling')))
        self.assertNotIn('20261023', json.dumps([workflow.commands(stage) for stage in ['full', 'controls', 'old-test']]))

    def test_full_dependency_graph_and_explicit_test(self):
        workflow = module.NotebookWorkflow('recompute', 'fresh')
        stages = ordered_stages(workflow, ['full'])
        for before, after in [('history', 'legacy'), ('baseline', 'dfs130'),
                              ('dfs130', 'bootstrap'), ('bootstrap', 'optuna'), ('used', 'full')]:
            self.assertLess(stages.index(before), stages.index(after))
        with self.assertRaises(ValueError), patch.object(module.subprocess, 'run') as command:
            workflow.ensure('old-test')
        command.assert_not_called()

    def test_incomplete_optuna_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'TASK_ROOT', Path(directory)):
            workflow = module.NotebookWorkflow('recompute', 'fresh')
            folder = workflow.path('optuna').parent
            folder.mkdir(parents=True)
            (folder / 'status.json').write_text(json.dumps({'completed': False, 'outcome': 'finished_with_failures'}))
            with self.assertRaisesRegex(RuntimeError, 'не полностью'):
                workflow.verify_outputs('optuna')


if __name__ == '__main__':
    unittest.main()
