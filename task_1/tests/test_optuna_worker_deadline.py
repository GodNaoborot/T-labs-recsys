"""Worker сам соблюдает срок, даже если управляющий процесс исчез."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src'
CODE = ('import sys,time;sys.path.insert(0,' + repr(str(SOURCE)) + ');'
        'from optuna_worker import arm_worker_deadline;'
        'arm_worker_deadline();print("numpy" in sys.modules,flush=True);time.sleep(20)')


class WorkerDeadlineTest(unittest.TestCase):
    def test_fit_cli_accepts_controller_objective(self):
        sys.path.insert(0, str(SOURCE))
        from optuna_worker import parser
        args = parser().parse_args(['--run-name', 'cli_check', '--stage', 'refit',
            '--method', 'manual', '--objective', 'pointwise',
            '--params-file', '/tmp/params.json', '--output', '/tmp/result'])
        self.assertEqual(args.objective, 'pointwise')

    def test_expired_deadline_exits_before_loading_libraries(self):
        environment = {**os.environ, 'TLAB_QUEUE_DEADLINE_UNIX': str(time.time() - 1)}
        result = subprocess.run([sys.executable, '-c', CODE], env=environment,
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(result.stdout, '')

    def test_search_deadline_stops_orphan_before_later_queue_deadline(self):
        started = time.monotonic()
        environment = {**os.environ, 'TLAB_QUEUE_DEADLINE_UNIX': str(time.time() + 10),
                       'TLAB_SEARCH_DEADLINE_UNIX': str(time.time() + .6)}
        result = subprocess.run([sys.executable, '-c', CODE], env=environment,
                                capture_output=True, text=True, timeout=3)
        self.assertIn(result.returncode, (-signal.SIGALRM, -signal.SIGKILL))
        self.assertLess(time.monotonic() - started, 2.)
        self.assertEqual(result.stdout.strip(), 'False')


if __name__ == '__main__':
    unittest.main()
