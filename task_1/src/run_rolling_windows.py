"""Последовательный запуск пяти окон с восстановлением завершенных этапов."""
from pathlib import Path
from replay_paths import saved_path
import argparse
import fcntl
import json
import os
import resource
import subprocess
import sys
import time

import pandas as pd
import psutil

from bootstrap_validation import file_sha256, read_json, write_json
from rolling_windows import ROOT, WINDOWS, RESULTS, RUN_NAME, RUN_HELP, window_spec, get_spec
from rolling_windows import window_paths, make_directories, prepare_targets, freeze_protocol
from rolling_catboost import prepare_pool, fit_model, bootstrap_window

RUN_LOCK_FD = None


def stage_marker(paths, stage, kind=None, split=None, objective=None):
    """Задать отдельный checkpoint завершения каждого вычислительного этапа."""
    parts = [value for value in (stage, kind, split, objective) if value]
    return paths['results'] / ('stage_' + '_'.join(parts) + '.json')


def run_feature_stage(paths, kind):
    """Передать генератору только общие целевые строки и новый путь результата."""
    targets = pd.read_parquet(paths['data'] / 'targets.parquet')
    output = paths['data'] / f'{kind}_features.parquet'
    for artifact in (output, output.with_suffix('.report.json'), output.with_suffix('.probe.parquet')):
        artifact.unlink(missing_ok=True)
    if kind == 'manual':
        from rolling_manual_features import generate_manual_features
        result = generate_manual_features(targets, output)
    elif kind == 'dfs':
        from rolling_dfs_features import generate_dfs_features
        result = generate_dfs_features(targets, output)
    else:
        raise ValueError(kind)
    return {'generator': result, 'feature_file_sha256': file_sha256(output)}


def perform_stage(args):
    """Выполнить ровно один этап отдельным процессом, затем освободить его RAM."""
    paths = window_paths(args.window)
    make_directories(paths)
    if args.stage == 'targets':
        return prepare_targets(get_spec(args.window))
    if args.stage == 'features':
        return run_feature_stage(paths, args.kind)
    if args.stage == 'pool':
        return prepare_pool(args.window, args.kind, args.split)
    if args.stage == 'fit':
        return fit_model(args.window, args.kind, args.objective)
    if args.stage == 'bootstrap':
        frame = bootstrap_window(get_spec(args.window))
        return {'comparisons': len(frame), 'all_reject_h0': bool(frame['reject_h0'].all()),
                'summary_sha256': file_sha256(paths['results'] / 'bootstrap_summary.csv'),
                'losses_sha256': file_sha256(paths['results'] / 'bootstrap_losses.npz')}
    raise ValueError(args.stage)


def execute_worker(args):
    """Сохранить успех только после результатов; ошибки оставляют диагностический отчет."""
    started = time.perf_counter()
    marker = stage_marker(window_paths(args.window), args.stage, args.kind, args.split, args.objective)
    report = {'completed': False, 'stage': args.stage, 'window': args.window}
    try:
        report['result'] = perform_stage(args)
        report['completed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started,
                      max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        write_json(marker, report)


def acquire_lock(lock):
    """Исключить конкурирующий runner или самостоятельный worker этих окон."""
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Уже выполняется другой расчет этих окон') from error


def run_worker(args):
    """Использовать lock родителя либо проверить протокол самостоятельного этапа."""
    if show_completed_run():
        return
    inherited = os.environ.get('ROLLING_RUN_LOCK_FD')
    if inherited is not None:
        os.fstat(int(inherited))
        execute_worker(args)
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / 'runner.lock').open('a+') as lock:
            acquire_lock(lock)
            if show_completed_run():
                return
            freeze_protocol()
            execute_worker(args)


def worker_environment():
    """Передать ребенку только действительный дескриптор общей блокировки."""
    environment = os.environ.copy()
    environment.pop('ROLLING_RUN_LOCK_FD', None)
    if RUN_LOCK_FD is not None:
        environment['ROLLING_RUN_LOCK_FD'] = str(RUN_LOCK_FD)
    return environment


def worker_command(window_id, stage, kind=None, split=None, objective=None):
    """Сформировать аргументы процесса без shell и подстановок строк команд."""
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', stage,
               '--window', window_id]
    for name, value in [('kind', kind), ('split', split), ('objective', objective)]:
        if value:
            command.extend([f'--{name}', value])
    return command


def reuse_completed(paths, marker, stage, kind, split, objective):
    """Повторно использовать завершенный этап только при наличии итоговых файлов."""
    if not marker.exists() or not read_json(marker)['completed']:
        return False
    try:
        check_stage_artifacts(paths, marker, stage, kind, split, objective)
        return True
    except (FileNotFoundError, AssertionError, KeyError, ValueError):
        return False


def check_stage_artifacts(paths, marker, stage, kind, split, objective):
    """Не считать сохраненный marker достаточным свидетельством готовности данных."""
    if stage == 'targets':
        config = read_json(paths['results'] / 'targets_report.json')
        assert file_sha256(paths['data'] / 'targets.parquet') == config['targets_sha256']
    elif stage == 'pool':
        check_saved_pool(paths, kind, split)
    elif stage == 'fit':
        check_saved_model(paths, kind, objective)
    elif stage == 'features':
        expected = read_json(marker)['result']['feature_file_sha256']
        assert file_sha256(paths['data'] / f'{kind}_features.parquet') == expected
    elif stage == 'bootstrap':
        checked_bootstrap_table(paths)


def check_saved_pool(paths, kind, split):
    """Проверить целостность Pool, его меток и train-порогов при продолжении."""
    folder = paths['data'] / f'{kind}_pools'
    config = read_json(folder / f'{split}_config.json')
    for name, key in [(f'{split}.qbin', 'pool_sha256'),
                      (f'{split}_targets.npz', 'targets_sha256'),
                      ('train_borders.tsv', 'borders_sha256')]:
        assert file_sha256(folder / name) == config[key]


def check_saved_model(paths, kind, objective):
    """Сверить checkpoint и результаты, без которых парное сравнение невозможно."""
    name = f'{kind}_{objective}'
    info = read_json(paths['results'] / f'{name}_report.json')
    assert file_sha256(saved_path(info['path'])) == info['sha256']
    for suffix, key in [('val_score.npy', 'score_sha256'),
                        ('group_ndcg.parquet', 'group_ndcg_sha256')]:
        assert file_sha256(paths['results'] / f'{name}_{suffix}') == info[key]


def checked_bootstrap_table(paths):
    """Включать в отчет только четыре завершенные проверки с целыми артефактами."""
    marker = read_json(stage_marker(paths, 'bootstrap'))
    assert marker['completed']
    for suffix, key in [('summary.csv', 'summary_sha256'), ('losses.npz', 'losses_sha256')]:
        assert file_sha256(paths['results'] / f'bootstrap_{suffix}') == marker['result'][key]
    frame = pd.read_csv(paths['results'] / 'bootstrap_summary.csv')
    expected = {(name, length) for name in ('ranker', 'pointwise') for length in (3, 4)}
    assert len(frame) == 4 and set(map(tuple, frame[['objective', 'block_days']].values)) == expected
    assert frame['window'].eq(paths['results'].name).all()
    return frame


def launch_stage(window_id, stage, kind=None, split=None, objective=None):
    """Запустить один процесс с отдельным логом и текущим статусом для наблюдения."""
    paths = window_paths(window_id)
    make_directories(paths)
    marker = stage_marker(paths, stage, kind, split, objective)
    if reuse_completed(paths, marker, stage, kind, split, objective):
        print('REUSE', marker.stem, window_id, flush=True)
        return
    status = dict(window=window_id, stage=stage, kind=kind, split=split,
                  objective=objective, started_unix=time.time(), log=str(marker.with_suffix('.log')))
    write_json(RESULTS / 'status.json', {'completed': False, 'current': status})
    print('STAGE_START', json.dumps(status), flush=True)
    with marker.with_suffix('.log').open('w') as log:
        subprocess.run(worker_command(window_id, stage, kind, split, objective),
                       stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT,
                       pass_fds=() if RUN_LOCK_FD is None else (RUN_LOCK_FD,),
                       env=worker_environment())
    assert read_json(marker)['completed']
    print('STAGE_DONE', window_id, marker.stem, flush=True)


def completed_pools(paths, kind):
    """Проверить готовность обоих Pool перед освобождением широкой матрицы."""
    for split in ('train', 'val'):
        marker = stage_marker(paths, 'pool', kind, split)
        if not marker.exists() or not read_json(marker)['completed']:
            return False
        if not reuse_completed(paths, marker, 'pool', kind, split, None):
            return False
    return True


def prepare_kind(window_id, kind):
    """Генерировать признаки только пока не готовы оба компактных Pool."""
    paths = window_paths(window_id)
    if not completed_pools(paths, kind):
        output = paths['data'] / f'{kind}_features.parquet'
        if not output.exists():
            marker = stage_marker(paths, 'features', kind)
            if marker.exists():
                marker.unlink()
        launch_stage(window_id, 'features', kind)
        for split in ('train', 'val'):
            launch_stage(window_id, 'pool', kind, split)
    output = paths['data'] / f'{kind}_features.parquet'
    if output.exists():
        output.unlink()


def summarize_completed():
    """Показывать все завершенные окна, не выбирая удачные результаты."""
    tables = completed_window_tables()
    if not tables:
        return
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(RESULTS / 'summary.csv', index=False)
    sensitivity = full.groupby(['window', 'objective'], as_index=False).agg(
        relative_loss=('relative_loss', 'first'), worst_upper_bound=('upper_bound', 'max'),
        passes_both_lengths=('reject_h0', 'all'), tested_lengths=('block_days', 'nunique'))
    sensitivity.to_csv(RESULTS / 'sensitivity.csv', index=False)
    overall_statistics(sensitivity).to_csv(RESULTS / 'overall.csv', index=False)


def completed_window_tables():
    """Не включать начатые, поврежденные или неполные bootstrap-окна в сводку."""
    tables = []
    for bounds in WINDOWS:
        paths = window_paths(window_spec(*bounds)['id'])
        if reuse_completed(paths, stage_marker(paths, 'bootstrap'), 'bootstrap', None, None, None):
            tables.append(checked_bootstrap_table(paths))
    return tables


def overall_statistics(sensitivity):
    """Проверить общий критерий только после получения всех пяти окон."""
    sensitivity = sensitivity.assign(passes_both_lengths=(
        sensitivity['passes_both_lengths'] & sensitivity['tested_lengths'].eq(2)))
    aggregate = sensitivity.groupby('objective', as_index=False).agg(
        windows_completed=('window', 'nunique'), windows_passed=('passes_both_lengths', 'sum'),
        minimum_loss=('relative_loss', 'min'), maximum_loss=('relative_loss', 'max'),
        worst_upper_bound=('worst_upper_bound', 'max'))
    aggregate['passes_all_requested_windows'] = (aggregate['windows_completed'].eq(len(WINDOWS))
                                                  & aggregate['windows_passed'].eq(len(WINDOWS)))
    return aggregate


def run_window(spec):
    """Сохранить порядок: общие события, признаки/Pool, четыре модели, bootstrap."""
    window_id = spec['id']
    launch_stage(window_id, 'targets')
    for kind in ('manual', 'dfs'):
        prepare_kind(window_id, kind)
    for objective in ('ranker', 'pointwise'):
        for kind in ('manual', 'dfs'):
            launch_stage(window_id, 'fit', kind, objective=objective)
    launch_stage(window_id, 'bootstrap')
    summarize_completed()


def execute_windows():
    """Продолжить пять заранее зафиксированных окон, не изменяя старый эксперимент."""
    if show_completed_run():
        return
    freeze_protocol()
    started = time.time()
    write_json(RESULTS / 'runner.json', {'pid': os.getpid(), 'started_unix': started})
    try:
        for bounds in WINDOWS:
            if psutil.disk_usage('/mnt/c' if Path('/mnt/c').exists() else ROOT).free < 8 * 1024**3:
                raise RuntimeError('На Windows-диске меньше8GiB свободного места')
            run_window(window_spec(*bounds))
        write_json(RESULTS / 'status.json', {'completed': True, 'elapsed_seconds': time.time() - started})
    except Exception as error:
        status = read_json(RESULTS / 'status.json') if (RESULTS / 'status.json').exists() else {}
        status.update(completed=False, error=repr(error))
        write_json(RESULTS / 'status.json', status)
        raise


def expected_report_paths():
    """Перечислить небольшие обязательные отчеты завершенного прогона."""
    reports = [RESULTS / name for name in ('protocol.json', 'summary.csv', 'sensitivity.csv', 'overall.csv')]
    for bounds in WINDOWS:
        folder = window_paths(window_spec(*bounds)['id'])['results']
        reports += [folder / 'bootstrap_summary.csv', folder / 'stage_bootstrap.json']
        reports += [folder / f'{kind}_{objective}_report.json'
                    for kind in ('manual', 'dfs') for objective in ('ranker', 'pointwise')]
    return reports


def read_run_status():
    """Прочитать статус и наличие отчетов без изменения файлов и проверки больших артефактов."""
    path = RESULTS / 'status.json'
    status = read_json(path) if path.exists() else {'completed': False, 'state': 'not_started'}
    reports = expected_report_paths()
    missing = [str(path.relative_to(RESULTS)) for path in reports if not path.is_file()]
    return {'run_name': RUN_NAME, 'results': str(RESULTS), 'read_only': True,
            'status': status, 'reports_present': len(reports) - len(missing),
            'reports_expected': len(reports), 'missing_reports': missing,
            'artifact_hashes_checked': False}


def show_completed_run():
    """Показать завершенный прогон и остановиться, даже если часть отчетов удалена."""
    status = read_run_status()
    if not status['status'].get('completed', False):
        return False
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return True


def run_all():
    """Показать завершенный прогон либо заблокировать и продолжить незавершенный."""
    if show_completed_run():
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    global RUN_LOCK_FD
    with (RESULTS / 'runner.lock').open('a+') as lock:
        acquire_lock(lock)
        RUN_LOCK_FD = lock.fileno()
        execute_windows()


def main():
    """Разделить просмотр статуса, управляющий запуск и отдельные процессы этапов."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=RUN_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--status', action='store_true', help='Только прочитать статус и наличие отчетов')
    parser.add_argument('--stage', choices=['all', 'targets', 'features', 'pool', 'fit', 'bootstrap'], default='all')
    parser.add_argument('--window')
    parser.add_argument('--kind', choices=['manual', 'dfs'])
    parser.add_argument('--split', choices=['train', 'val'])
    parser.add_argument('--objective', choices=['ranker', 'pointwise'])
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_run_status(), ensure_ascii=False, indent=2))
    elif args.stage == 'all':
        run_all()
    else:
        if not args.window:
            parser.error('Для отдельного этапа нужен --window')
        run_worker(args)


if __name__ == '__main__':
    main()
