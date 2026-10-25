"""Сохранённые пути разрешаются в текущей копии, независимо от cwd."""
from pathlib import Path
import hashlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import replay_paths
import c4_test_evaluation
import final_test_features
import optuna_data


class PortableSavedPathsTests(unittest.TestCase):
    def test_relative_paths_do_not_depend_on_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / 'checkout/task_1'
            task.mkdir(parents=True)
            before = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(replay_paths, 'TASK_ROOT', task):
                    self.assertEqual(replay_paths.saved_path('data/history.parquet'), task / 'data/history.parquet')
                    self.assertEqual(replay_paths.saved_path('../docs/report.csv').resolve(), task.parent / 'docs/report.csv')
            finally:
                os.chdir(before)

    def test_frozen_artifact_readers_accept_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            source = task / 'data/input.bin'
            source.parent.mkdir()
            source.write_bytes(b'fixed content')
            record = {'path': 'data/input.bin', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            with patch.object(replay_paths, 'TASK_ROOT', task):
                self.assertEqual(c4_test_evaluation._artifact(record), source)
                self.assertEqual(final_test_features._assert_artifact(record), source)

    def test_relative_cache_state_keeps_content_and_stat_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            source = task / 'pool.qbin'
            source.write_bytes(b'fixed pool')
            state = optuna_data._file_state(source)
            state['path'] = 'pool.qbin'
            record = {'state': state, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            with patch.object(replay_paths, 'TASK_ROOT', task):
                optuna_data._check_artifact(record, checksum=True)
                source.write_bytes(b'changed pool')
                with self.assertRaises(ValueError):
                    optuna_data._check_artifact(record, checksum=True)


if __name__ == '__main__':
    unittest.main()
