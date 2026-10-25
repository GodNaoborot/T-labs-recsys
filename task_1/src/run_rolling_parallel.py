"""Очередь до двух CatBoost fit с контролем CPU/RAM и неизменным ML-протоколом."""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time

import psutil

import run_rolling_windows as serial
from bootstrap_validation import file_sha256, read_json, write_json
from rolling_windows import ROOT, RESULTS, WINDOWS, RUN_HELP, window_spec, window_paths, fixed_model_params
from parallel_resources import peak_budget_gib, can_launch, MemoryPressureGuard

OUTPUT = RESULTS / 'parallel_execution'
RESERVE_GIB = 1.5
CPU_LIMIT = 12
MAX_ACTIVE = 2
POLL_SECONDS = 1


def event(name, **details):
    """Добавить событие очереди в последовательный журнал одного controller."""
    value = {'event': name, 'unix': time.time(), **details}
    print(json.dumps(value, ensure_ascii=False), flush=True)
    with (OUTPUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def job_identity(job):
    """Уникально назвать модель, не связывая разные окна или objective."""
    return f"{job['window']}:{job['kind']}:{job['objective']}"


def make_job(window, kind, objective):
    """Прочитать размеры Pool, нужные для переноса измеренного пика RAM."""
    folder = window_paths(window)['data'] / f'{kind}_pools'
    return {'window': window, 'kind': kind, 'objective': objective,
            'train_rows': read_json(folder / 'train_config.json')['rows_used'],
            'val_rows': read_json(folder / 'val_config.json')['rows_used'],
            'threads': fixed_model_params(objective)['thread_count'], 'serial_only': False}


def fit_marker(job):
    """Использовать прежний checkpoint завершения модели."""
    return serial.stage_marker(window_paths(job['window']), 'fit', job['kind'], None, job['objective'])


def is_completed(job):
    """Проверить модель, прогнозы и hashes по прежним правилам эксперимента."""
    return serial.reuse_completed(window_paths(job['window']), fit_marker(job), 'fit',
                                  job['kind'], None, job['objective'])


def completed_profiles():
    """Собрать реальные полные пики загрузки, fit и оценки каждого типа модели."""
    profiles = []
    for bounds in WINDOWS:
        window = window_spec(*bounds)['id']
        paths = window_paths(window)
        for kind in ('manual', 'dfs'):
            if not (paths['data'] / f'{kind}_pools/train_config.json').exists():
                continue
            for objective in ('ranker', 'pointwise'):
                job = make_job(window, kind, objective)
                if is_completed(job):
                    profiles.append({**job, 'peak_rss_gib': read_json(fit_marker(job))['max_rss_gib']})
    return profiles


def save_execution_manifest(previous):
    """Зафиксировать изменение расписания отдельно от научного протокола."""
    sources = [Path(__file__), Path(__file__).with_name('parallel_resources.py')]
    manifest = {'started_unix': time.time(), 'previous_runner': previous,
                'scientific_protocol_sha256': file_sha256(RESULTS / 'protocol.json'),
                'max_active_fit': MAX_ACTIVE, 'cpu_thread_budget': CPU_LIMIT,
                'memory_reserve_gib': RESERVE_GIB, 'poll_seconds': POLL_SECONDS,
                'first_window_calibration': 'all four variants complete alone before parallel queue',
                'feature_generation_and_quantization': 'exclusive, no concurrent fit',
                'memory_fallback': 'terminate newer own worker; retry once only after other fits finish',
                'model_parameters_changed': False,
                'source_files': {str(p): file_sha256(p) for p in sources},
                'fit_time_interpretation': 'parallel wall time includes competition; not isolated benchmark'}
    write_json(OUTPUT / 'manifest.json', manifest)
    return manifest


def preparation_phase():
    """Закончить первое окно для калибровки, затем подготовить остальные без fit."""
    event('calibration_first_window')
    serial.run_window(window_spec(*WINDOWS[0]))
    write_execution_note()
    for bounds in WINDOWS[1:]:
        window = window_spec(*bounds)['id']
        if psutil.disk_usage('/mnt/c' if Path('/mnt/c').exists() else ROOT).free < 8 * 1024**3:
            raise RuntimeError('Для подготовки следующего окна нужно минимум 8 GiB на диске Windows')
        event('exclusive_preparation', window=window)
        serial.launch_stage(window, 'targets')
        for kind in ('manual', 'dfs'):
            serial.prepare_kind(window, kind)


def pending_jobs():
    """Предпочесть первые окна, сохраняя право подобрать пару по CPU и памяти."""
    jobs = []
    for bounds in WINDOWS:
        window = window_spec(*bounds)['id']
        for objective in ('ranker', 'pointwise'):
            for kind in ('manual', 'dfs'):
                job = make_job(window, kind, objective)
                if not is_completed(job):
                    jobs.append(job)
    return jobs


def spawn_job(job, profiles):
    """Открыть уникальный лог и передать worker тот же общий lock FD."""
    marker = fit_marker(job)
    write_json(marker, {'completed': False, 'stage': 'fit', 'window': job['window']})
    log_path = marker.with_suffix('.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(serial.worker_command(job['window'], 'fit', job['kind'],
                                    objective=job['objective']), stdout=log, stderr=subprocess.STDOUT,
                                    cwd=ROOT, pass_fds=(serial.RUN_LOCK_FD,), env=serial.worker_environment())
    active = {**job, 'process': process, 'pid': process.pid, 'started_unix': time.time(),
              'log': str(log_path), 'rss_gib': 0., 'sampled_peak_rss_gib': 0.,
              'worker_overlap_seconds': 0., 'max_concurrent_fit': 1}
    event('fit_started', job=job_identity(job), pid=process.pid, threads=job['threads'],
          estimated_peak_gib=peak_budget_gib(job, profiles), serial_only=job['serial_only'])
    return active


def sample_active(active):
    """Измерить память своих работников, не считая завершившиеся процессы живыми."""
    for job in active:
        try:
            job['rss_gib'] = psutil.Process(job['pid']).memory_info().rss / 2**30
        except psutil.NoSuchProcess:
            job['rss_gib'] = 0.
        job['sampled_peak_rss_gib'] = max(job['sampled_peak_rss_gib'], job['rss_gib'])
        job['max_concurrent_fit'] = max(job['max_concurrent_fit'], len(active))


def save_timing(job, result):
    """Сохранить длительность worker и факт пересечения с другим обучением."""
    path = OUTPUT / 'job_timings.csv'
    values = {key: job[key] for key in ['window', 'kind', 'objective', 'pid', 'started_unix',
              'serial_only', 'sampled_peak_rss_gib', 'worker_overlap_seconds', 'max_concurrent_fit']}
    values.update(finished_unix=time.time(), result=result)
    fresh = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values))
        if fresh:
            writer.writeheader()
        writer.writerow(values)


def reap_jobs(active):
    """Принять успех только при корректном завершении и целых выходных файлах."""
    live, completed = [], 0
    for job in active:
        code = job['process'].poll()
        if code is None:
            live.append(job)
            continue
        if not is_completed(job):
            event('fit_failed', job=job_identity(job), code=code, log=job['log'])
            raise RuntimeError(f"Не завершена модель {job_identity(job)}; лог {job['log']}")
        save_timing(job, 'completed')
        event('fit_completed', job=job_identity(job), code=code)
        completed += 1
    return live, completed


def stop_newer_for_memory(active, pending):
    """Прервать только более новый свой fit и отложить его до одиночного повтора."""
    job = max(active, key=lambda value: value['started_unix'])
    event('memory_fallback', job=job_identity(job), pid=job['pid'])
    job['process'].terminate()
    try:
        job['process'].wait(timeout=10)
    except subprocess.TimeoutExpired:
        job['process'].kill()
        job['process'].wait(timeout=10)
    active.remove(job)
    if is_completed(job):
        save_timing(job, 'completed_before_guard')
        return
    if job['serial_only']:
        raise RuntimeError('Недостаточно памяти даже для одиночного fit; очередь остановлена')
    Path(job['log']).replace(Path(job['log']).with_suffix('.memory_attempt.log'))
    save_timing(job, 'memory_retry_serial')
    pending.append({**{k: job[k] for k in ['window', 'kind', 'objective', 'train_rows', 'val_rows', 'threads']},
                    'serial_only': True})


def choose_job(pending, active, profiles, available):
    """Выбрать допустимую пару; отложенный fit выполняется только в одиночку."""
    if any(job['serial_only'] for job in active):
        return None
    retry = next((i for i, job in enumerate(pending) if job['serial_only']), None)
    if retry is not None:
        if active:
            return None
        return retry if can_launch(pending[retry], [], profiles, available,
                                   CPU_LIMIT, RESERVE_GIB, MAX_ACTIVE) else None
    for index, job in enumerate(pending):
        if job['serial_only'] and active:
            continue
        if can_launch(job, active, profiles, available, CPU_LIMIT, RESERVE_GIB, MAX_ACTIVE):
            return index
    return None


def maybe_bootstrap(active, done):
    """Считать готовое окно при свободном слоте, не смешивая большие подготовки с fit."""
    if active or psutil.virtual_memory().available < 2 * 2**30:
        return
    for bounds in WINDOWS:
        spec = window_spec(*bounds)
        if spec['id'] in done:
            continue
        jobs = [make_job(spec['id'], kind, obj) for kind in ('manual', 'dfs') for obj in ('ranker', 'pointwise')]
        if all(is_completed(job) for job in jobs):
            serial.launch_stage(spec['id'], 'bootstrap')
            serial.summarize_completed()
            write_execution_note()
            done.add(spec['id'])
            event('window_completed', window=spec['id'])


def public_status(active, pending, available, phase='parallel_fit'):
    """Сохранить совместимый current и полный список одновременно работающих моделей."""
    visible = [{key: job[key] for key in ['window', 'kind', 'objective', 'pid', 'log',
               'rss_gib', 'started_unix', 'sampled_peak_rss_gib']} for job in active]
    current = {**visible[0], 'stage': 'fit'} if visible else None
    value = {'completed': False, 'execution_mode': 'bounded_parallel', 'phase': phase,
             'active': visible, 'pending_fits': len(pending), 'available_gib': available,
             'updated_unix': time.time()}
    if current:
        value['current'] = current
    write_json(OUTPUT / 'status.json', value)
    write_json(RESULTS / 'status.json', value)


def sample_resources(active, available):
    """Записать редкую компактную выборку RAM, CPU-бюджета и swap-out."""
    path = OUTPUT / 'resources.csv'
    fresh = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.writer(stream)
        if fresh:
            writer.writerow(['unix', 'active_fit', 'available_gib', 'worker_rss_gib', 'threads', 'swap_out_bytes'])
        writer.writerow([time.time(), len(active), available, sum(j['rss_gib'] for j in active),
                         sum(j['threads'] for j in active), psutil.swap_memory().sout])


def parallel_queue():
    """Чередовать контроль RAM, завершение, bootstrap и допуск следующей модели."""
    pending, active, done = pending_jobs(), [], set()
    profiles, guard = completed_profiles(), MemoryPressureGuard()
    state = {'previous': time.monotonic(), 'last_status': 0., 'idle_since': None}
    assert all(peak_budget_gib(job, profiles) is not None for job in pending), 'Нет RAM-профиля типа модели'
    while pending or active:
        active, profiles = monitor_cycle(active, pending, profiles, guard, state)
        maybe_bootstrap(active, done)
        state['last_status'] = advance_queue(pending, active, profiles, state['last_status'])
        state['idle_since'] = check_idle_queue(pending, active, state['idle_since'])
        time.sleep(POLL_SECONDS)
    finish_bootstraps(done)


def monitor_cycle(active, pending, profiles, guard, state):
    """Обновить метрики работников и освободить память при устойчивом давлении."""
    now = time.monotonic()
    if len(active) > 1:
        for job in active:
            job['worker_overlap_seconds'] += now - state['previous']
    state['previous'] = now
    active, completed = reap_jobs(active)
    if completed:
        profiles = completed_profiles()
    sample_active(active)
    available = psutil.virtual_memory().available / 2**30
    if guard.observe(available, psutil.swap_memory().sout, now) and active:
        event('memory_pressure_observed', available_gib=available, reason=guard.reason)
        stop_newer_for_memory(active, pending)
        guard.reset()
    return active, profiles


def check_idle_queue(pending, active, idle_since):
    """Не ждать бесконечно, если ни одно оставшееся обучение не помещается."""
    if active or not pending:
        return None
    now = time.monotonic()
    idle_since = now if idle_since is None else idle_since
    if now - idle_since > 300:
        available = psutil.virtual_memory().available / 2**30
        raise RuntimeError(f'Пять минут нет допуска по RAM: доступно {available:.2f} GiB; '
                           'результаты сохранены, очередь можно продолжить после освобождения памяти')
    return idle_since


def finish_bootstraps(done):
    """Не объявлять успех до наличия всех пяти проверенных bootstrap-отчетов."""
    deadline = time.monotonic() + 300
    while len(serial.completed_window_tables()) != len(WINDOWS):
        maybe_bootstrap([], done)
        if len(serial.completed_window_tables()) == len(WINDOWS):
            return
        if time.monotonic() > deadline:
            raise RuntimeError('Не завершены все пять bootstrap-окон; общий успех не заявляется')
        time.sleep(1)


def advance_queue(pending, active, profiles, last_status):
    """Запустить не более одного нового fit за цикл и обновлять статус раз в5секунд."""
    available = psutil.virtual_memory().available / 2**30
    index = choose_job(pending, active, profiles, available)
    if index is not None:
        active.append(spawn_job(pending.pop(index), profiles))
    now = time.monotonic()
    if now - last_status > 5:
        public_status(active, pending, available)
        sample_resources(active, available)
        return now
    return last_status


def write_execution_note():
    """Сохранить описание расписания в результатах запуска, не изменяя документацию."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    note = ('После калибровки первого окна выполняются до двух обучений одновременно: '
            f'суммарно не более {CPU_LIMIT} fit-потоков, с проверкой ожидаемых пиков RAM '
            f'и резервом {RESERVE_GIB:g} GiB. При давлении памяти выполняется возврат '
            'к одиночному fit. Генерация и квантование выполняются отдельно. '
            'Параметры моделей не меняются. Время параллельных fit включает конкуренцию '
            'за ресурсы и не используется для прямого сравнения скорости manual/DFS.\n')
    (OUTPUT / 'execution_note.md').write_text(note)


def execute(previous):
    """Проверить прежний протокол и продолжить все пять окон новым расписанием."""
    if serial.show_completed_run():
        return
    serial.freeze_protocol()
    manifest = save_execution_manifest(previous)
    write_json(RESULTS / 'runner.json', {'pid': os.getpid(), 'started_unix': previous['started_unix'],
               'parallel_started_unix': manifest['started_unix'], 'execution_mode': 'bounded_parallel'})
    try:
        preparation_phase()
        parallel_queue()
        serial.summarize_completed()
        write_execution_note()
        complete = {'completed': True, 'execution_mode': 'bounded_parallel',
                    'elapsed_seconds': time.time() - previous['started_unix']}
        write_json(RESULTS / 'status.json', complete)
        write_json(OUTPUT / 'status.json', complete)
        event('all_windows_completed')
    except Exception as error:
        save_controller_error(error)
        raise


def save_controller_error(error):
    """Сохранить явную ошибку очереди, не останавливая чужие или уже идущие fit."""
    event('controller_error', error=repr(error))
    write_json(OUTPUT / 'status.json', {'completed': False, 'error': repr(error)})
    status_path = RESULTS / 'status.json'
    status = read_json(status_path) if status_path.exists() else {}
    status.update(completed=False, error=repr(error))
    write_json(RESULTS / 'status.json', status)


def main():
    """Показать завершенный прогон либо начать/продолжить очередь после общего flock."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=RUN_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--status', action='store_true', help='Только прочитать статус и наличие отчетов')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(serial.read_run_status(), ensure_ascii=False, indent=2))
        return
    if serial.show_completed_run():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runner_path = RESULTS / 'runner.json'
    previous = read_json(runner_path) if runner_path.exists() else {'started_unix': time.time()}
    event('waiting_for_existing_lock', pid=os.getpid(), previous_runner=previous)
    write_json(OUTPUT / 'waiting.json', {'pid': os.getpid(), 'previous_runner': previous})
    with (RESULTS / 'runner.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        serial.RUN_LOCK_FD = lock.fileno()
        event('lock_acquired', pid=os.getpid())
        execute(previous)


if __name__ == '__main__':
    main()
