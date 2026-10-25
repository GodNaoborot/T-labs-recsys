"""Просмотр отчётов сохраняет исходники, запись требует явного параметра."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_report_notebooks as reports
import run_experiment_notebooks as experiments


class FakeClient:
    def __init__(self, notebook, **kwargs):
        self.notebook = notebook

    def execute(self, **kwargs):
        count = 0
        for cell in self.notebook.cells:
            if cell.cell_type == 'code':
                count += 1
                cell.execution_count = count
                cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text='computed\n')]


class NotebookRunnerTests(unittest.TestCase):
    def run_case(self, module, arguments):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'notebooks/tiny.ipynb'
            source.parent.mkdir()
            notebook = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_code_cell('print(2)')],
                metadata={'tlab_report': True})
            nbformat.write(notebook, source)
            original = source.read_bytes()
            arguments = [str(root / 'copies') if value == 'COPIES' else value for value in arguments]
            with patch.object(module, 'TASK_ROOT', root), patch.object(module, 'NOTEBOOKS', ('tiny.ipynb',)), \
                    patch.object(reports, 'make_kernel', return_value=SimpleNamespace(has_kernel=False)), \
                    patch.object(reports, 'NotebookClient', FakeClient), redirect_stdout(StringIO()):
                module.main(arguments)
            return original, source.read_bytes(), {
                str(path.relative_to(root)): nbformat.read(path, as_version=4)
                for path in root.rglob('*.ipynb')}

    def test_default_report_preserves_source_bytes(self):
        for module in (reports, experiments):
            with self.subTest(runner=module.__name__):
                original, current, files = self.run_case(module, [])
                self.assertEqual(current, original)
                self.assertEqual(set(files), {'notebooks/tiny.ipynb'})

    def test_explicit_output_writes_only_executed_copy(self):
        for module in (reports, experiments):
            with self.subTest(runner=module.__name__):
                original, current, files = self.run_case(module, ['--output-dir', 'COPIES'])
                self.assertEqual(current, original)
                self.assertEqual(files['copies/tiny.ipynb'].cells[0].execution_count, 1)
                self.assertEqual(files['copies/tiny.ipynb'].cells[0].outputs[0].text, 'computed\n')

    def test_in_place_explicitly_updates_outputs(self):
        for module in (reports, experiments):
            with self.subTest(runner=module.__name__):
                original, current, files = self.run_case(module, ['--in-place'])
                self.assertNotEqual(current, original)
                self.assertEqual(files['notebooks/tiny.ipynb'].cells[0].execution_count, 1)

    def test_recompute_keeps_separate_default_destination(self):
        original, current, files = self.run_case(experiments, ['--mode', 'recompute'])
        self.assertEqual(current, original)
        self.assertIn('results/reproduction/repeat_01/notebooks/tiny.ipynb', files)

    def test_recompute_rejects_in_place_before_execution(self):
        with patch.object(experiments, 'execute_notebook') as execute, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                experiments.main(['--mode', 'recompute', '--in-place'])
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
