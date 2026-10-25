"""Проверки изоляции запуска и зафиксированного бюджета C4 без обучения."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c4_runner_for_tests', ROOT / 'src/run_c4.py')
c4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c4)


class C4RunnerTests(unittest.TestCase):
    def test_paths_reject_escape(self):
        for name in ['../other', '/tmp/job', 'bad/name', '', 'a' * 65]:
            with self.assertRaises(ValueError):
                c4.run_paths(name)

    def test_protocol_freezes_sources_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'src').mkdir()
            source = root / 'src/a.py'
            source.write_text('value=1\n')
            args = SimpleNamespace(run_name='c4_tiny', max_training_seconds=7200)
            with patch.object(c4, 'ROOT', root):
                data, result = c4.run_paths(args.run_name)
                protocol = c4.freeze_protocol(data, result, args)
                self.assertFalse(protocol['test_used'])
                self.assertEqual(protocol['models'], ['M-t', 'C2b-t', 'D462-t'])
                self.assertEqual(protocol['max_training_seconds'], 7200)
                self.assertEqual(len(protocol['comparisons']), 3)
                c4.verify_sources(protocol)
                with self.assertRaises(ValueError):
                    c4.freeze_protocol(data, result, SimpleNamespace(run_name='c4_tiny', max_training_seconds=3600))
                source.write_text('value=2\n')
                with self.assertRaises(ValueError):
                    c4.verify_sources(protocol)

    def test_existing_unregistered_folder_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, result = root / 'data', root / 'results'
            data.mkdir()
            sentinel = data / 'existing'
            sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                c4.freeze_protocol(data, result, SimpleNamespace(run_name='c4_tiny', max_training_seconds=7200))
            self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
