"""Ночная очередь: контроль1600, общий поиск2ч и matched refit до общего дедлайна."""
from pathlib import Path
from replay_paths import saved_path
from datetime import datetime, timezone
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

import optuna
import psutil
from optuna.trial import TrialState

from optuna_config import (SEED, SEARCH_SPACE, baseline_parameters, fixed_parameters,
                           read_json, run_paths, runtime_versions, source_hashes,
                           suggest_parameters, write_json)
from optuna_selection import (TRIAL_LIMITS, create_studies, choose_next, matched_selection,
                              save_search_summary, trial_counts)
from project_artifacts import TASK_ROOT, file_hash

STOP_GRACE_SECONDS = 4.
POLL_SECONDS = .5
FIT_IDS = [f'{phase}_{method}_{objective}'
           for objective in ('pointwise', 'ranker') for phase in ('control', 'tuned')
           for method in ('manual', 'dfs')]


class QueueDeadline(Exception):
    """Общий дедлайн: ожидаемая остановка без новых worker."""


class ResourceStop(RuntimeError):
    """Системное ограничение RAM/диска останавливает всю очередь."""


def parse_deadline(value):
    """Преобразовать ISO8601 с явным часовым поясом в Unix UTC."""
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('--deadline должен содержать часовой пояс')
    return parsed.timestamp()


def remaining_seconds(args):
    """Оставить четыре секунды на SIGTERM/SIGKILL перед общим дедлайном."""
    return args.deadline_unix - time.time() - STOP_GRACE_SECONDS


def check_deadline(args):
    """Не запускать новый шаг после окончания доступного времени."""
    if remaining_seconds(args) <= 0:
        raise QueueDeadline()


def freeze_protocol(args, paths):
    """Заморозить код, версии, очередь и matched-политику до первого fit."""
    protocol = {'run_name': args.run_name, 'budget_hours': args.budget_hours,
                'deadline_utc': datetime.fromtimestamp(args.deadline_unix, timezone.utc).isoformat(),
                'seed': SEED, 'search_space': SEARCH_SPACE, 'test_used': False,
                'train_groups_requested': 12000, 'val_groups_requested': 8000,
                'trial_limits': TRIAL_LIMITS, 'n_startup_trials': 3,
                'selection': 'best among first k COMPLETE per side; k=min COMPLETE count within objective',
                'attempts': 'COMPLETE+FAIL+PRUNED+RUNNING; WAITING excluded; baseline included',
                'schedule': 'pointwise pair then ranker pair; fewer attempted within objective',
                'search_clock': 'first search start+budget, includes preparation/load/fit/evaluation; no reset',
                'queue': ['control_pointwise', 'search', 'tuned_pointwise', 'control_ranker', 'tuned_ranker'],
                'fixed_params': {kind: fixed_parameters(kind) for kind in ('ranker', 'pointwise')},
                'baseline_params': {kind: baseline_parameters(kind) for kind in ('ranker', 'pointwise')},
                'source_sha256': source_hashes(), 'versions': runtime_versions(),
                'objective': 'external NDCG@10 with fixed seed ties',
                'limitations': ['group subsample; fulltrain borders; reused val; exploratory results']}
    path = paths['results'] / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Протокол изменился: нужен новый --run-name')
    if not path.exists():
        write_json(path, protocol)


def worker_command(args, stage, output, **options):
    """Передать worker параметры без shell; отчет вызвать отдельным коротким процессом."""
    if stage == 'report':
        code = ("import sys,os,signal,time; signal.setitimer(signal.ITIMER_REAL,max(.001,float(os.environ['TLAB_QUEUE_DEADLINE_UNIX'])-time.time())); from pathlib import Path; from optuna_config import run_paths,read_json,write_json; "
                "from optuna_report import publish_reports; p=run_paths(sys.argv[1]); "
                "publish_reports(p,read_json(p['results']/'stages.json'),float(sys.argv[2])); "
                "write_json(Path(sys.argv[3])/'result.json',{'completed':True})")
        return [sys.executable, '-c', code, args.run_name, str(args.deadline_unix), str(output)]
    command = [sys.executable, str(TASK_ROOT / 'src/optuna_worker.py'),
               '--run-name', args.run_name, '--stage', stage, '--output', str(output)]
    for name, value in options.items():
        command += ['--' + name.replace('_', '-'), str(value)]
    return command


def stop_worker(process):
    """Убрать только созданную нами session/processgroup за резерв остановки."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
    except ProcessLookupError:
        process.wait(timeout=1)


def resource_sample(process):
    """Измерить RSS единственного worker и свободную RAM/место WSL и хоста."""
    try:
        rss = psutil.Process(process.pid).memory_info().rss / 2**30
    except psutil.NoSuchProcess:
        rss = 0.
    return {'worker_rss_gib': rss, 'available_gib': psutil.virtual_memory().available / 2**30,
            'host_and_guest_free_gib': min(psutil.disk_usage(p).free for p in (str(TASK_ROOT), str(Path('/mnt/c') if Path('/mnt/c').exists() else TASK_ROOT))) / 2**30}


def watch_worker(process, output, status_path, timeout, details, deadline_unix=None, budget_deadline=None):
    """Проверять бюджет и память каждые0.5с; вернуть причину до finally-stop."""
    started, last_status, low_memory, peak = time.monotonic(), 0., 0, 0.
    while process.poll() is None:
        if deadline_unix is not None and time.time() >= deadline_unix - STOP_GRACE_SECONDS:
            return {'completed': False, 'reason': 'deadline'}
        if budget_deadline is not None and time.time() >= budget_deadline - STOP_GRACE_SECONDS:
            return {'completed': False, 'reason': 'time_budget'}
        if time.monotonic() - started >= timeout:
            return {'completed': False, 'reason': 'time_budget'}
        sample = resource_sample(process)
        peak = max(peak, sample['worker_rss_gib'])
        low_memory = low_memory + 1 if sample['available_gib'] < 2. else 0
        if sample['worker_rss_gib'] > 7.5 or low_memory >= 3:
            return {'completed': False, 'reason': 'memory_guard', 'resources': sample}
        if sample['host_and_guest_free_gib'] < .75:
            return {'completed': False, 'reason': 'disk_guard', 'resources': sample}
        if time.monotonic() - last_status >= 5:
            write_json(status_path, {'completed': False, **details, 'worker_pid': process.pid,
                       'worker_seconds': time.monotonic() - started, 'resources': sample})
            last_status = time.monotonic()
        time.sleep(POLL_SECONDS)
    result = read_json(output / 'result.json') if (output / 'result.json').exists() else {}
    if process.returncode in (-signal.SIGKILL, -signal.SIGALRM):
        if deadline_unix is not None and time.time() >= deadline_unix - STOP_GRACE_SECONDS:
            result = {**result, 'reason': 'deadline'}
        elif budget_deadline is not None and time.time() >= budget_deadline - STOP_GRACE_SECONDS:
            result = {**result, 'reason': 'search_budget'}
    return {**result, 'completed': process.returncode == 0 and result.get('completed', False),
            'returncode': process.returncode, 'peak_monitored_rss_gib': peak}


def _active_record(args, process, output):
    """Записать доказательства личности своего worker для безопасного resume."""
    try:
        created = psutil.Process(process.pid).create_time()
    except psutil.NoSuchProcess:
        created = None
    return {'worker_pid': process.pid, 'worker_create_time': created,
            'run_name': args.run_name, 'output': str(output), 'running': True}


def _worker_environment(args, budget_deadline=None):
    """Передать независимый OS watchdog и убрать устаревший search deadline для control."""
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(TASK_ROOT / 'src')}
    environment['TLAB_QUEUE_DEADLINE_UNIX'] = str(args.deadline_unix - STOP_GRACE_SECONDS)
    environment.pop('TLAB_SEARCH_DEADLINE_UNIX', None)
    if budget_deadline is not None:
        environment['TLAB_SEARCH_DEADLINE_UNIX'] = str(budget_deadline)
    return environment


def run_worker(args, paths, stage, output, timeout, budget_deadline=None, **options):
    """Один worker; в finally остановить только его processgroup и сохранить результат."""
    check_deadline(args)
    output.mkdir(parents=True, exist_ok=True)
    if budget_deadline is not None and time.time() >= budget_deadline - STOP_GRACE_SECONDS:
        result = {'completed': False, 'reason': 'time_budget'}
        write_json(output / 'controller_result.json', result)
        return result
    environment = _worker_environment(args, budget_deadline)
    result = {'completed': False, 'reason': 'controller_interrupted'}
    with (output / 'worker.log').open('a') as log:
        process = subprocess.Popen(worker_command(args, stage, output, **options), cwd=TASK_ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            record = _active_record(args, process, output)
            write_json(paths['results'] / 'active_worker.json', record)
            result = watch_worker(process, output, paths['results'] / 'status.json',
                                  min(timeout, remaining_seconds(args)),
                                  {'phase': stage, 'output': str(output)}, args.deadline_unix, budget_deadline)
        finally:
            stop_worker(process)
            write_json(paths['results'] / 'active_worker.json', {'worker_pid': process.pid, 'running': False})
            write_json(output / 'controller_result.json', result)
    return result


def guard_previous_worker(paths):
    """Не создавать второй fit, если прежний worker с тем же PID/create_time/run еще жив."""
    path = paths['results'] / 'active_worker.json'
    previous = read_json(path) if path.exists() else read_json(paths['results'] / 'status.json') if (paths['results'] / 'status.json').exists() else {}
    pid = previous.get('worker_pid')
    if not pid or previous.get('running') is False:
        return
    try:
        process = psutil.Process(pid)
        if previous.get('worker_create_time') is not None and abs(process.create_time() - previous['worker_create_time']) > .01:
            return
        command = process.cmdline()
        own_run = '--run-name' not in command or previous.get('run_name') in (None, command[command.index('--run-name') + 1])
        if any('optuna_worker.py' in part or 'optuna_report' in part for part in command) and own_run:
            raise RuntimeError(f'Предыдущий worker еще жив: PID {pid}; новый fit запрещен')
    except psutil.NoSuchProcess:
        pass


def _raise_terminal(result):
    """Отделить общий дедлайн и системные ограничения от ошибки отдельной модели."""
    if result.get('reason') == 'deadline':
        raise QueueDeadline()
    if result.get('reason') in {'memory_guard', 'disk_guard'}:
        raise ResourceStop(str(result))


def new_attempt(folder):
    """Создать новый attempt, не перезаписывая частично выполненный предыдущий."""
    folder.mkdir(parents=True, exist_ok=True)
    number = 1
    while (folder / f'attempt_{number:04d}').exists():
        number += 1
    output = folder / f'attempt_{number:04d}'
    output.mkdir()
    return output


def _prepared(paths, stage, options):
    """Проверить уже существующий небольшой cache/config перед повторным использованием."""
    from optuna_data import prepare_selection, pool_info
    try:
        if stage == 'selection':
            if not (paths['data'] / 'selection/config.json').exists():
                return False
            prepare_selection(paths['data'])
        else:
            pool_info(paths['data'], options['method'], options['split'], options['scope'])
        return True
    except (FileNotFoundError, ValueError, KeyError):
        return False


def prepare_stage(args, paths, stage, key, limit=None, budget_deadline=None, **options):
    """Подготовить зависимость в отдельном процессе; ошибка не превращается в готовый cache."""
    check_deadline(args)
    if _prepared(paths, stage, options):
        return True
    if limit is not None and limit <= 0:
        return False
    output = new_attempt(paths['results'] / 'preparation' / key)
    result = run_worker(args, paths, stage, output,
                        min(remaining_seconds(args), limit if limit is not None else remaining_seconds(args)),
                        budget_deadline=budget_deadline, **options)
    path = paths['results'] / 'preparation.json'
    journal = read_json(path) if path.exists() else {}
    journal[key] = {'status': 'completed' if result['completed'] else 'failed', 'output': str(output), 'result': result}
    write_json(path, journal)
    _raise_terminal(result)
    return result['completed']


def ensure_pools(args, paths, method, scope, search_deadline=None):
    """Готовить train перед val; поисковая подготовка входит в общие2ч."""
    for split in ('train', 'val'):
        limit = None if search_deadline is None else search_deadline - time.time() - STOP_GRACE_SECONDS
        if not prepare_stage(args, paths, 'prepare', f'{scope}_{method}_{split}', limit,
                             budget_deadline=search_deadline, method=method, split=split, scope=scope):
            return False
    return True


def verified_refit(output, parameters):
    """Повторно принять только те же requested params, все artifacts и input config hashes."""
    if read_json(output / 'requested_params.json') != parameters:
        raise ValueError(f'Параметры завершенного fit отличаются: {output}')
    result = read_json(output / 'result.json')
    artifacts, hashes = result.get('artifacts', {}), result.get('artifact_sha256', {})
    if not result.get('completed') or not artifacts or set(artifacts) != set(hashes):
        raise ValueError(f'Нет полного проверяемого результата: {output}')
    for name, path in artifacts.items():
        if file_hash(path) != hashes[name]:
            raise ValueError(f'Поврежден артефакт: {path}')
    for record in result.get('input_configs', {}).values():
        if file_hash(saved_path(record['path'])) != record['sha256']:
            raise ValueError(f'Изменился входной config: {record["path"]}')
    return result


def _verified_stage(stages, stage_id, parameters):
    """Вернуть проверенный ранее завершенный fit либо отсутствие пригодного результата."""
    stage = stages.get(stage_id, {})
    if stage.get('status') != 'completed':
        return None
    try:
        return verified_refit(saved_path(stage['output']), parameters)
    except (FileNotFoundError, ValueError, KeyError):
        return None


def _save_stage(paths, stages, stage_id, status, output, result, **extra):
    """Сразу сохранять итог каждой модели независимо от успеха соседней."""
    stages[stage_id] = {'status': status, 'output': str(output) if output is not None else None,
                        'result': result, **extra}
    write_json(paths['results'] / 'stages.json', stages)


def publish_if_time(args, paths, stages):
    """Строить отчет отдельным процессом, только при наличии запаса общего времени."""
    if remaining_seconds(args) < 60:
        return
    write_json(paths['results'] / 'stages.json', stages)
    output = new_attempt(paths['results'] / 'reports')
    result = run_worker(args, paths, 'report', output, min(30., remaining_seconds(args)))
    _raise_terminal(result)


def fit_stage(args, paths, stages, phase, method, objective, selected=None):
    """Один полный control/tuned fit; обычная ошибка не блокирует последующие модели."""
    check_deadline(args)
    stage_id = f'{phase}_{method}_{objective}'
    if phase == 'tuned' and (selected is None or selected['params'] is None):
        _save_stage(paths, stages, stage_id, 'skipped', None, {'reason': 'no_matched_trials'})
        return
    parameters = {**fixed_parameters(objective), **(baseline_parameters(objective) if phase == 'control' else selected['params'])}
    if _verified_stage(stages, stage_id, parameters) is not None:
        return
    if phase == 'tuned' and selected['params'] == baseline_parameters(objective):
        control_id = f'control_{method}_{objective}'
        result = _verified_stage(stages, control_id, parameters)
        if result is not None:
            _save_stage(paths, stages, stage_id, 'completed', stages[control_id]['output'], result,
                        reused_from=control_id, additional_fit_seconds=0.)
            return
    _perform_fit(args, paths, stages, stage_id, method, objective, parameters)


def _perform_fit(args, paths, stages, stage_id, method, objective, parameters):
    """Подготовить только нужный full Pool и сохранить отдельную попытку модели."""
    output = new_attempt(paths['results'] / 'queue' / stage_id)
    if not ensure_pools(args, paths, method, 'full'):
        _save_stage(paths, stages, stage_id, 'failed', output, {'reason': 'dependency_failed'})
        return
    write_json(output / 'requested_params.json', parameters)
    _save_stage(paths, stages, stage_id, 'running', output, {})
    try:
        result = run_worker(args, paths, 'refit', output, remaining_seconds(args),
                            method=method, objective=objective, params_file=output / 'requested_params.json')
    except QueueDeadline:
        result = {'completed': False, 'reason': 'deadline'}
    if result['completed']:
        try:
            result = {**result, **verified_refit(output, parameters)}
        except (FileNotFoundError, ValueError, KeyError) as error:
            result = {**result, 'completed': False, 'error': repr(error)}
    _save_stage(paths, stages, stage_id, 'completed' if result['completed'] else 'failed', output, result)
    _raise_terminal(result)
    publish_if_time(args, paths, stages)


def search_clock(args, paths):
    """Сохранить первое начало поиска; время подготовки и простоя resume не возвращается."""
    path = paths['results'] / 'search_clock.json'
    if path.exists():
        clock = read_json(path)
        if clock['budget_seconds'] != args.budget_hours * 3600:
            raise ValueError('Бюджет существующего search_clock изменился')
        return clock
    started = time.time()
    clock = {'started_unix': started, 'budget_seconds': args.budget_hours * 3600,
             'deadline_unix': started + args.budget_hours * 3600}
    write_json(path, clock)
    return clock


def run_trial(args, paths, study, method, objective, remaining, search_deadline=None):
    """Одна проба с внешней NDCG; FAIL не мешает очереди, guards останавливают всё."""
    attempted = trial_counts(study)['attempted']
    study.sampler = optuna.samplers.TPESampler(seed=SEED + attempted, n_startup_trials=3)
    trial = study.ask()
    parameters = {**fixed_parameters(objective), **suggest_parameters(trial)}
    output = new_attempt(paths['results'] / 'search' / f'{method}_{objective}' / f'trial_{trial.number:04d}')
    write_json(output / 'requested_params.json', parameters)
    try:
        result = run_worker(args, paths, 'trial', output, remaining, budget_deadline=search_deadline, method=method,
                            objective=objective, params_file=output / 'requested_params.json')
    except QueueDeadline:
        study.tell(trial, state=TrialState.PRUNED)
        raise
    trial.set_user_attr('worker_result', result)
    if result['completed']:
        study.tell(trial, result['metrics']['NDCG@10'])
    else:
        state = TrialState.PRUNED if result.get('reason') in ('time_budget', 'search_budget', 'deadline') else TrialState.FAIL
        study.tell(trial, state=state)
    _raise_terminal(result)


def search(args, paths):
    """Один persisted wallclock на cache prep, загрузку, fit и оценку всех четырех studies."""
    clock = search_clock(args, paths)
    studies = create_studies(paths)
    try:
        if clock['deadline_unix'] - time.time() <= STOP_GRACE_SECONDS:
            return matched_selection(studies)
        for method in ('manual', 'dfs'):
            if not ensure_pools(args, paths, method, 'search', clock['deadline_unix']):
                return matched_selection(studies)
        while True:
            check_deadline(args)
            remaining = min(clock['deadline_unix'] - time.time() - STOP_GRACE_SECONDS, remaining_seconds(args))
            choice = choose_next(studies)
            if remaining <= 0 or choice is None:
                return matched_selection(studies)
            method, objective = choice
            run_trial(args, paths, studies[choice], method, objective, remaining, clock['deadline_unix'])
            save_search_summary(paths, studies)
    finally:
        save_search_summary(paths, studies)


def _final_status(paths, stages, outcome, error=None):
    """Показать все недополученные fit ID, включая failed/skipped, без новых запусков."""
    missing = [name for name in FIT_IDS if stages.get(name, {}).get('status') != 'completed']
    value = {'completed': outcome == 'completed', 'outcome': outcome, 'phase': 'finished',
             'remaining_fit_ids': missing, 'fits_completed': len(FIT_IDS) - len(missing),
             'test_used': False, 'updated_unix': time.time()}
    if error is not None:
        value['error'] = repr(error)
    write_json(paths['results'] / 'status.json', value)


def execute(args, paths):
    """Контроль MC → поиск → tuned MC → контроль YetiRank → tuned YetiRank."""
    if not hasattr(args, 'deadline_unix'):
        configure_deadline(args, paths)
    guard_previous_worker(paths)
    freeze_protocol(args, paths)
    path = paths['results'] / 'stages.json'
    stages = read_json(path) if path.exists() else {}
    try:
        if not prepare_stage(args, paths, 'selection', 'selection'):
            raise RuntimeError('Общая selection не подготовлена')
        for method in ('manual', 'dfs'):
            fit_stage(args, paths, stages, 'control', method, 'pointwise')
        selection = search(args, paths)
        for objective in ('pointwise', 'ranker'):
            if objective == 'ranker':
                for method in ('manual', 'dfs'):
                    fit_stage(args, paths, stages, 'control', method, objective)
            for method in ('manual', 'dfs'):
                fit_stage(args, paths, stages, 'tuned', method, objective, selection[objective][method])
        publish_if_time(args, paths, stages)
        outcome = 'completed' if all(stages.get(k, {}).get('status') == 'completed' for k in FIT_IDS) else 'finished_with_failures'
        _final_status(paths, stages, outcome)
    except QueueDeadline:
        _final_status(paths, stages, 'deadline')
    except BaseException as error:
        _final_status(paths, stages, 'resource_guard' if isinstance(error, ResourceStop) else 'error', error)
        raise


def parser():
    """Бюджет поиска и общий лимит повторного запуска."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--run-name', required=True)
    command.add_argument('--budget-hours', type=float, default=2.)
    command.add_argument('--deadline', help='Необязательная абсолютная граница ISO 8601')
    command.add_argument('--max-total-hours', type=float, default=6.)
    command.add_argument('--status', action='store_true')
    return command



def configure_deadline(args, paths):
    """Сохранить общий лимит при продолжении, не продлевая его новым вызовом."""
    if args.max_total_hours <= 0:
        raise ValueError('--max-total-hours должен быть положительным')
    protocol = paths['results'] / 'protocol.json'
    if args.deadline:
        args.deadline_unix = parse_deadline(args.deadline)
    elif protocol.exists():
        args.deadline_unix = parse_deadline(read_json(protocol)['deadline_utc'])
    else:
        args.deadline_unix = time.time() + args.max_total_hours * 3600
    args.deadline = datetime.fromtimestamp(args.deadline_unix, timezone.utc).isoformat()


def _handle_sigterm(signum, frame):
    """Перевести SIGTERM controller в обычное unwinding с finally-stop своего worker."""
    raise KeyboardInterrupt('Controller получил SIGTERM')


def main():
    """Не допустить второго controller; deadline — нормальный ожидаемый исход."""
    args = parser().parse_args()
    paths = run_paths(args.run_name)
    if args.status:
        print(read_json(paths['results'] / 'status.json'))
        return
    if not 0 < args.budget_hours <= 2:
        raise ValueError('Бюджет поиска: больше нуля и не более2 часов')
    configure_deadline(args, paths)
    paths['results'].mkdir(parents=True, exist_ok=True)
    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        with (paths['results'] / 'controller.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args, paths)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    main()
