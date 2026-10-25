"""Проверки ночной очереди и остановки собственных процессов без CatBoost fit."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import optuna
import psutil
from optuna.trial import TrialState, create_trial
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_optuna as controller
from optuna_config import baseline_parameters, fixed_parameters, SEARCH_SPACE, write_json
from optuna_selection import matched_selection
from project_artifacts import file_hash

optuna.logging.set_verbosity(optuna.logging.ERROR)
SAFE = {'worker_rss_gib': .1, 'available_gib': 8., 'host_and_guest_free_gib': 50.}


class OptunaControllerTest(unittest.TestCase):
    def paths(self, folder):
        return {'results': Path(folder), 'data': Path(folder) / 'data'}

    def args(self, seconds=3600):
        return SimpleNamespace(run_name='unit_test', budget_hours=2.,
                               deadline_unix=time.time() + seconds)

    def parameters(self, objective='pointwise'):
        return {**fixed_parameters(objective), **baseline_parameters(objective)}

    def completed_fit(self, folder, parameters):
        folder.mkdir(parents=True)
        model = folder / 'model.cbm'
        model.write_bytes(b'synthetic-model')
        config = folder / 'input_config.json'
        config.write_text('{}')
        result = {'completed': True, 'metrics': {'NDCG@10': .75},
                  'artifacts': {'model': str(model)}, 'artifact_sha256': {'model': file_hash(model)},
                  'input_configs': {'train': {'path': str(config), 'sha256': file_hash(config)}}}
        write_json(folder / 'requested_params.json', parameters)
        write_json(folder / 'result.json', result)
        return result

    def test_cli_aware_timezone_and_no_legacy_refit_flag(self):
        args = controller.parser().parse_args(['--run-name', 'unit_test'])
        self.assertEqual(args.budget_hours, 2.)
        self.assertIsNone(args.deadline)
        self.assertEqual(args.max_total_hours, 6.)
        expected = datetime(2026, 10, 24, 11, tzinfo=timezone.utc).timestamp()
        self.assertEqual(controller.parse_deadline('2026-10-24T13:00:00+02:00'), expected)
        with tempfile.TemporaryDirectory() as folder:
            paths = self.paths(folder)
            with patch.object(controller.time, 'time', return_value=100.):
                controller.configure_deadline(args, paths)
            self.assertEqual(args.deadline_unix, 21700.)
            write_json(Path(folder) / 'protocol.json', {'deadline_utc': args.deadline})
            resumed = controller.parser().parse_args(['--run-name', 'unit_test'])
            with patch.object(controller.time, 'time', return_value=1000.):
                controller.configure_deadline(resumed, paths)
            self.assertEqual(resumed.deadline_unix, 21700.)
        with self.assertRaises(ValueError):
            controller.parse_deadline('2026-10-24T13:00:00')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            controller.parser().parse_args(['--run-name', 'unit_test', '--refit'])

    def test_worker_watchdog_deadlines_and_control_env_cleanup(self):
        args = self.args()
        with patch.dict(os.environ, {'TLAB_SEARCH_DEADLINE_UNIX': '1'}):
            control = controller._worker_environment(args)
            search = controller._worker_environment(args, 5000.)
        self.assertNotIn('TLAB_SEARCH_DEADLINE_UNIX', control)
        self.assertEqual(float(control['TLAB_QUEUE_DEADLINE_UNIX']), args.deadline_unix - 4.)
        self.assertEqual(float(search['TLAB_SEARCH_DEADLINE_UNIX']), 5000.)
        with self.assertRaises(KeyboardInterrupt):
            controller._handle_sigterm(15, None)

    def test_search_clock_is_persistent_and_not_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, args = self.paths(folder), self.args()
            with patch.object(controller.time, 'time', return_value=100.):
                first = controller.search_clock(args, paths)
            before = (Path(folder) / 'search_clock.json').read_bytes()
            with patch.object(controller.time, 'time', return_value=5000.):
                second = controller.search_clock(args, paths)
            self.assertEqual(first, second)
            self.assertEqual(first['deadline_unix'], 7300.)
            self.assertEqual(before, (Path(folder) / 'search_clock.json').read_bytes())
            args.budget_hours = 1.
            with self.assertRaises(ValueError):
                controller.search_clock(args, paths)

    def test_search_clock_exists_before_cache_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, args = self.paths(folder), self.args()
            studies = {key: optuna.create_study(direction='maximize')
                       for key in [('manual', 'pointwise'), ('dfs', 'pointwise'),
                                   ('manual', 'ranker'), ('dfs', 'ranker')]}
            def prepare(*args, **kwargs):
                self.assertTrue((Path(folder) / 'search_clock.json').exists())
                return False
            with patch.object(controller, 'create_studies', return_value=studies), \
                 patch.object(controller, 'ensure_pools', side_effect=prepare), \
                 patch.object(controller, 'run_trial') as trial:
                result = controller.search(args, paths)
            trial.assert_not_called()
            self.assertEqual(result['pointwise']['k'], 0)

    def test_deadline_stops_owned_sigterm_ignoring_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, args = self.paths(folder), self.args(4.3)
            output = Path(folder) / 'worker'
            command = [sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']
            started = time.time()
            with patch.object(controller, 'worker_command', return_value=command), \
                 patch.object(controller, 'resource_sample', return_value=SAFE):
                result = controller.run_worker(args, paths, 'refit', output, 20.)
            self.assertEqual(result['reason'], 'deadline')
            self.assertLess(time.time() - started, 4.3)
            record = json.loads((Path(folder) / 'active_worker.json').read_text())
            self.assertFalse(record['running'])
            self.assertFalse(psutil.pid_exists(record['worker_pid']))

    def test_expired_deadline_never_spawns_and_returns_normal_queue_outcome(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, args = self.paths(folder), self.args(-1)
            with patch.object(controller.subprocess, 'Popen') as spawn:
                with self.assertRaises(controller.QueueDeadline):
                    controller.run_worker(args, paths, 'refit', Path(folder) / 'fit', 60.)
                spawn.assert_not_called()
            with patch.object(controller, 'guard_previous_worker'), patch.object(controller, 'freeze_protocol'), \
                 patch.object(controller, 'prepare_stage', side_effect=controller.QueueDeadline):
                controller.execute(args, paths)
            status = json.loads((Path(folder) / 'status.json').read_text())
            self.assertEqual(status['outcome'], 'deadline')
            self.assertEqual(status['remaining_fit_ids'], controller.FIT_IDS)

    def test_self_watchdog_exit_is_classified_as_budget_or_deadline(self):
        process = SimpleNamespace(poll=lambda: -14, returncode=-14)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = controller.watch_worker(process, output, output / 'status.json', 60., {},
                                             deadline_unix=time.time() + 3.)
            self.assertEqual(result['reason'], 'deadline')
            result = controller.watch_worker(process, output, output / 'status.json', 60., {},
                                             deadline_unix=time.time() + 100., budget_deadline=time.time() - 1.)
            self.assertEqual(result['reason'], 'search_budget')

    def test_resource_guard_requires_three_low_ram_samples(self):
        process = SimpleNamespace(poll=lambda: None, pid=123)
        low = {**SAFE, 'available_gib': 1.9}
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(controller, 'resource_sample', side_effect=[low, low, low]) as sample, \
             patch.object(controller.time, 'sleep'):
            result = controller.watch_worker(process, Path(folder), Path(folder) / 'status.json', 60., {})
        self.assertEqual(result['reason'], 'memory_guard')
        self.assertEqual(sample.call_count, 3)

    def test_ordinary_fit_failure_does_not_abort_next_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, args, stages = self.paths(folder), self.args(), {}
            with patch.object(controller, 'ensure_pools', return_value=True), \
                 patch.object(controller, 'run_worker', return_value={'completed': False, 'error': 'synthetic model failure'}) as worker, \
                 patch.object(controller, 'publish_if_time'):
                for method in ('manual', 'dfs'):
                    controller.fit_stage(args, paths, stages, 'control', method, 'pointwise')
            self.assertEqual(worker.call_count, 2)
            self.assertEqual(stages['control_manual_pointwise']['status'], 'failed')
            self.assertEqual(stages['control_dfs_pointwise']['status'], 'failed')
            self.assertNotEqual(stages['control_manual_pointwise']['output'], stages['control_dfs_pointwise']['output'])

    def test_fixed_queue_order_is_independent_of_control_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, args, order = self.paths(folder), self.args(), []
            selection = {objective: {m: {'params': baseline_parameters(objective)} for m in ('manual', 'dfs')}
                         for objective in ('pointwise', 'ranker')}
            def fit(args, paths, stages, phase, method, objective, selected=None):
                key = f'{phase}_{method}_{objective}'
                order.append(key)
                stages[key] = {'status': 'failed' if key == 'control_manual_pointwise' else 'completed'}
            def search(*args):
                order.append('search')
                return selection
            with patch.object(controller, 'guard_previous_worker'), patch.object(controller, 'freeze_protocol'), \
                 patch.object(controller, 'prepare_stage', return_value=True), patch.object(controller, 'fit_stage', side_effect=fit), \
                 patch.object(controller, 'search', side_effect=search), patch.object(controller, 'publish_if_time'):
                controller.execute(args, paths)
            expected = controller.FIT_IDS.copy()
            expected.insert(2, 'search')
            self.assertEqual(order, expected)

    def test_baseline_reuse_requires_verified_corresponding_control(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, args, stages = self.paths(folder), self.args(), {}
            output = Path(folder) / 'control'
            result = self.completed_fit(output, self.parameters())
            stages['control_manual_pointwise'] = {'status': 'completed', 'output': str(output), 'result': result}
            selected = {'params': baseline_parameters('pointwise')}
            with patch.object(controller, '_perform_fit') as fit:
                controller.fit_stage(args, paths, stages, 'tuned', 'manual', 'pointwise', selected)
            fit.assert_not_called()
            self.assertEqual(stages['tuned_manual_pointwise']['additional_fit_seconds'], 0.)
            self.assertEqual(stages['tuned_manual_pointwise']['reused_from'], 'control_manual_pointwise')
            stages.pop('tuned_manual_pointwise')
            stages['control_manual_pointwise']['status'] = 'failed'
            with patch.object(controller, '_perform_fit') as fit:
                controller.fit_stage(args, paths, stages, 'tuned', 'manual', 'pointwise', selected)
            fit.assert_called_once()
            stages['control_manual_pointwise']['status'] = 'completed'
            (output / 'input_config.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                controller.verified_refit(output, self.parameters())

    def test_tuned_uses_matched_prefix_not_unmatched_winner(self):
        studies = {(m, o): optuna.create_study(direction='maximize')
                   for o in ('pointwise', 'ranker') for m in ('manual', 'dfs')}
        distributions = {'depth': IntDistribution(4, 8), 'learning_rate': FloatDistribution(.03, .2, log=True),
                         'l2_leaf_reg': CategoricalDistribution(SEARCH_SPACE['l2_leaf_reg'])}
        for method, depths, values in [('manual', [6, 7, 8], [.6, .7, .99]), ('dfs', [6, 7], [.6, .7])]:
            for depth, value in zip(depths, values):
                params = {**baseline_parameters('pointwise'), 'depth': depth}
                studies[(method, 'pointwise')].add_trial(create_trial(value=value, params=params, distributions=distributions))
        selected = matched_selection(studies)['pointwise']['manual']
        with tempfile.TemporaryDirectory() as folder, patch.object(controller, '_perform_fit') as fit:
            controller.fit_stage(self.args(), self.paths(folder), {}, 'tuned', 'manual', 'pointwise', selected)
        self.assertEqual(fit.call_args.args[-1]['depth'], 7)

    def test_orphan_worker_guard_uses_create_time(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.paths(folder)
            write_json(Path(folder) / 'active_worker.json', {'worker_pid': 123, 'worker_create_time': 10.,
                                                           'run_name': 'unit_test', 'running': True})
            own = SimpleNamespace(create_time=lambda: 10., cmdline=lambda: ['python', 'optuna_worker.py', '--run-name', 'unit_test'])
            with patch.object(controller.psutil, 'Process', return_value=own), self.assertRaises(RuntimeError):
                controller.guard_previous_worker(paths)
            reused = SimpleNamespace(create_time=lambda: 11., cmdline=own.cmdline)
            with patch.object(controller.psutil, 'Process', return_value=reused):
                controller.guard_previous_worker(paths)

    def test_trial_budget_pruned_and_resource_failure_stops_all(self):
        with tempfile.TemporaryDirectory() as folder:
            for reason in ('time_budget', 'memory_guard'):
                study = optuna.create_study(direction='maximize')
                result = {'completed': False, 'reason': reason}
                with patch.object(controller, 'run_worker', return_value=result):
                    if reason == 'memory_guard':
                        with self.assertRaises(controller.ResourceStop):
                            controller.run_trial(self.args(), self.paths(folder), study, 'manual', 'ranker', 30.)
                    else:
                        controller.run_trial(self.args(), self.paths(folder), study, 'manual', 'ranker', 30.)
                state = TrialState.FAIL if reason == 'memory_guard' else TrialState.PRUNED
                self.assertEqual(study.trials[0].state, state)


if __name__ == '__main__':
    unittest.main()
