"""Проверки защитного процесса без запуска моделей и сигналов реальным PID."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import watch_temporal_controls as guard


class TestControlGuard(unittest.TestCase):
    def test_command_requires_exact_runner_and_run_name(self):
        command = ['python', '/tmp/run_temporal_controls.py', '--run-name', 'run1',
                   '--stage', 'fit', '--variant', 'D462']
        self.assertTrue(guard._matches(command, 'run1', worker=True))
        self.assertFalse(guard._matches(command, 'run', worker=True))
        self.assertFalse(guard._matches(command, 'run1-suffix', worker=True))
        command[1] = '/tmp/other_run_temporal_controls.py'
        self.assertFalse(guard._matches(command, 'run1', worker=True))

    def test_supervisor_is_not_selected_as_worker(self):
        command = ['python', 'run_temporal_controls.py', '--run-name', 'run1']
        self.assertTrue(guard._matches(command, 'run1'))
        self.assertFalse(guard._matches(command, 'run1', worker=True))

    def test_file_cache_rss_is_not_anonymous_limit(self):
        state = {'wsl_disk_free_gib': 8, 'host_disk_free_gib': 8,
                 'worker_rss_gib': 11, 'anonymous_rss_gib': 3}
        self.assertIsNone(guard._resource_reason(state, 2))
        self.assertEqual(guard._resource_reason(state, 3), 'memory_reserve')
        state['anonymous_rss_gib'] = 8.76
        self.assertEqual(guard._resource_reason(state, 0), 'memory_reserve')

    def test_both_disks_need_reserve(self):
        state = {'wsl_disk_free_gib': 4.9, 'host_disk_free_gib': 10,
                 'anonymous_rss_gib': 1}
        self.assertEqual(guard._resource_reason(state, 0), 'disk_reserve_wsl')
        state.update(wsl_disk_free_gib=10, host_disk_free_gib=4.9)
        self.assertEqual(guard._resource_reason(state, 0), 'disk_reserve_windows')

    def test_reused_pid_is_never_signaled(self):
        command = ['python', 'run_temporal_controls.py', '--run-name', 'run1',
                   '--stage', 'fit']
        record = {'pid': 123, 'create_time': 10, 'command': command}
        process = Mock()
        process.create_time.return_value = 11
        with patch.object(guard.psutil, 'Process', return_value=process):
            guard._stop_workers([record], 'run1')
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()

    def test_changed_command_is_never_signaled(self):
        record = {'pid': 123, 'create_time': 10,
                  'command': ['python', 'run_temporal_controls.py', '--run-name',
                              'run1', '--stage', 'fit']}
        process = Mock()
        process.create_time.return_value = 10
        process.cmdline.return_value = ['python', 'unrelated.py']
        with patch.object(guard.psutil, 'Process', return_value=process):
            guard._stop_workers([record], 'run1')
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
