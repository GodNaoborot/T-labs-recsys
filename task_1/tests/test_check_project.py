"""Явная проверка публичного клона не должна скрывать ошибки локальных артефактов."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import check_project


class CheckProjectTests(unittest.TestCase):
    def run_check(self, arguments):
        output = StringIO()
        with redirect_stdout(output):
            check_project.main(arguments)
        return json.loads(output.getvalue())

    def test_skip_artifacts_is_explicit_and_does_not_read_models(self):
        with patch.object(check_project, 'verify_current_artifacts') as verify, \
                patch.object(check_project, 'check_sources', return_value={'scripts': 1}) as syntax:
            result = self.run_check(['--skip-artifacts'])
        verify.assert_not_called()
        syntax.assert_called_once_with()
        self.assertEqual(result['artifacts']['status'], 'skipped')
        self.assertEqual(result['syntax']['scripts'], 1)

    def test_default_mode_still_verifies_local_artifacts(self):
        with patch.object(check_project, 'verify_current_artifacts',
                          return_value={'comparison_id': 'fixture'}) as verify, \
                patch.object(check_project, 'check_sources', return_value={}):
            result = self.run_check([])
        verify.assert_called_once_with()
        self.assertEqual(result['artifacts']['comparison_id'], 'fixture')

    def test_default_mode_does_not_hide_artifact_errors(self):
        with patch.object(check_project, 'verify_current_artifacts',
                          side_effect=ValueError('hash mismatch')):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.run_check([])


if __name__ == '__main__':
    unittest.main()
