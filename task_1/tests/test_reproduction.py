"""План воспроизведения не должен неявно запускать обучение или старые модели."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import reproduce


class ReproductionTests(unittest.TestCase):
    def call_main(self, arguments):
        output = StringIO()
        with redirect_stdout(output):
            reproduce.main(arguments)
        return output.getvalue()

    def test_default_only_prints_plan_without_running_commands(self):
        with patch.object(reproduce.subprocess, 'run') as run:
            plan = json.loads(self.call_main([]))
        run.assert_not_called()
        self.assertFalse(plan['executed'])
        self.assertIn('--training-run', plan['commands']['evaluate'])

    def test_report_route_only_executes_safe_report_runner(self):
        with patch.object(reproduce.subprocess, 'run') as run:
            self.call_main(['--stage', 'reports'])
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[1], 'src/run_report_notebooks.py')
        self.assertNotIn('src/run_c4.py', command)

    def test_new_training_name_is_forwarded_to_test(self):
        plan = json.loads(self.call_main(['--run-name', 'new_run', '--download']))
        command = plan['commands']['evaluate']
        self.assertEqual(command[command.index('--training-run') + 1], 'new_run')
        self.assertEqual(command[command.index('--run-name') + 1], 'new_run_test')

    def test_all_rejects_missing_inputs_before_any_subprocess(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(reproduce, 'TASK_ROOT', Path(directory)), \
                patch.object(reproduce.subprocess, 'run') as run:
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.call_main(['--stage', 'all'])
        run.assert_not_called()

    def test_all_orders_stages_and_uses_same_training_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in reproduce.FORMULA_INPUTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            with patch.object(reproduce, 'TASK_ROOT', root), \
                    patch.object(reproduce.subprocess, 'run') as run:
                self.call_main(['--stage', 'all', '--download', '--run-name', 'fresh'])
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([c[1] for c in commands],
                             ['src/prepare_history.py', 'src/run_c4.py', 'src/run_c4_test.py'])
            self.assertIn('fresh', commands[1])
            self.assertEqual(commands[2][-2:], ['--training-run', 'fresh'])

    def test_existing_history_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'data/listens.parquet').touch()
            with patch.object(reproduce, 'TASK_ROOT', root), \
                    patch.object(reproduce.subprocess, 'run') as run:
                with self.assertRaises(FileExistsError):
                    self.call_main(['--stage', 'prepare', '--download'])
            run.assert_not_called()

    def test_unsafe_name_and_invalid_budget_are_rejected(self):
        for arguments in [['--run-name', '../old'], ['--max-training-seconds', '7201']]:
            with self.subTest(arguments=arguments), patch('sys.stderr', StringIO()), \
                    patch.object(reproduce.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    self.call_main(arguments)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
