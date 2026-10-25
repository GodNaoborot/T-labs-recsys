"""Лёгкие CLI читают UTF-8 и разделяют вход отчёта и папку вывода."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import check_project
import summarize_selected_features as summary

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LightCliTests(unittest.TestCase):
    def test_source_check_explicitly_reads_utf8_under_non_utf8_default(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / 'src').mkdir()
            (task / 'notebooks').mkdir()
            (task / 'src/example.py').write_text("answer = 'ответ'\n", encoding='utf-8')
            notebook = {'cells': [{'cell_type': 'code', 'source': ["name = 'имя'\n"]}]}
            (task / 'notebooks/example.ipynb').write_text(
                json.dumps(notebook, ensure_ascii=False), encoding='utf-8')
            original_open = Path.open
            encodings = []
            def non_utf8_default(path, mode='r', buffering=-1, encoding=None,
                                 errors=None, newline=None):
                if 'b' not in mode:
                    encodings.append(encoding)
                    encoding = encoding or 'cp1251'
                return original_open(path, mode, buffering, encoding, errors, newline)
            with patch.object(check_project, 'TASK_ROOT', task), patch.object(Path, 'open', non_utf8_default):
                result = check_project.check_sources()
            self.assertEqual(result, {'scripts': 1, 'notebooks': 1, 'code_cells': 1})
            self.assertEqual(encodings, ['utf-8', 'utf-8'])

    def test_standard_and_separate_output_match_without_touching_sources(self):
        canonical = ROOT / summary.OUTPUT
        expected = {path.name: path.read_bytes() for path in canonical.glob('*.csv')
                    if path.name != 'prediction_values_change.csv'}
        self.assertEqual(len(expected), 6)
        protected = {path: digest(path) for path in canonical.iterdir() if path.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project'
            run = root / summary.RUN
            sources = [summary.RUN / 'generation_dfs_train.json',
                       summary.RUN / 'features_preparation.json',
                       summary.RUN / 'models/D462-t/evaluation.json',
                       summary.RUN / 'models/D462-t/control.json',
                       Path('task_1/results/experiment-1/temporal_runs/temporal_used_20261023_v1/protocol.json'),
                       summary.OUTPUT / 'prediction_values_change.csv']
            sources += [summary.DEFINITIONS / (entity + '.json') for entity in summary.ENTITY_LABELS]
            for relative in sources:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            with redirect_stdout(StringIO()):
                summary.main(['--root', str(root)])
            default = root / summary.OUTPUT
            original_default = {path.name: digest(path) for path in default.iterdir() if path.is_file()}
            separate = Path(directory) / 'separate-report'
            self.assertFalse(separate.exists())
            with redirect_stdout(StringIO()):
                summary.main(['--root', str(root), '--output-dir', str(separate)])
            for name, content in expected.items():
                self.assertEqual((default / name).read_bytes(), content)
                self.assertEqual((separate / name).read_bytes(), content)
            self.assertEqual(original_default,
                             {path.name: digest(path) for path in default.iterdir() if path.is_file()})
        self.assertTrue(all(digest(path) == value for path, value in protected.items()))


if __name__ == '__main__':
    unittest.main()
