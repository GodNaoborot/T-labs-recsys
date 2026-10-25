"""Полные выбранные временные DFS-признаки: расчет порциями и сопоставимый YetiRank."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import mmap
import importlib.metadata
import platform
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil

from replay_paths import saved_path, configure_legacy, legacy_featuretools_dir, project_path, run_name

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = Path('experiment-1/temporal_runs')
SELECTION = ROOT / 'results' / RELATIVE / 'temporal_used_20261023_v1/protocol.json'
BASELINE = ROOT / 'results/tuning/optuna_20261023_2h_v2/queue/control_manual_ranker/attempt_0001'
DEFINITIONS = ROOT / 'data' / RELATIVE / 'temporal_topk_20261023_v2/definitions'
STAGES = ['generate_train', 'quantize_train', 'generate_val', 'quantize_val', 'fit', 'compare']
SOURCE_FILES = ['run_temporal_full.py', 'temporal_full_data.py', 'temporal_full_training.py',
                'temporal_dfs_features.py', 'cumulative_features.py', 'temporal_sample_data.py',
                'temporal_column_quantization.py', 'watch_temporal_full.py',
                'optuna_evaluation.py', 'train_dfs_catboost.py', 'block_bootstrap.py', 'project_artifacts.py', 'replay_paths.py']



def configure_sources(args):
    """Выбрать входные повторы и передать их worker-процессам."""
    global SELECTION, BASELINE, DEFINITIONS
    configure_legacy(args.legacy_run)
    selection_run = run_name(args.selection_run)
    definitions_run = run_name(args.definitions_run)
    SELECTION = ROOT / 'results' / RELATIVE / selection_run / 'protocol.json'
    DEFINITIONS = ROOT / 'data' / RELATIVE / definitions_run / 'definitions'
    BASELINE = project_path(args.baseline_dir)
    os.environ['TLAB_TEMPORAL_SELECTION_RUN'] = selection_run
    os.environ['TLAB_TEMPORAL_DEFINITIONS_RUN'] = definitions_run
    os.environ['TLAB_TEMPORAL_BASELINE_DIR'] = str(BASELINE)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Атомарный статус: читатель не увидит половину JSON."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    state = path.stat()
    return {'path': str(path), 'bytes': state.st_size, 'mtime_ns': state.st_mtime_ns,
            'sha256': sha256(path)}


def resource_state():
    """Проверять и диск Windows, на котором физически расположен WSL."""
    status = Path('/proc/self/status').read_text().splitlines()
    anonymous = next(int(line.split()[1]) for line in status if line.startswith('RssAnon:')) / 2**20
    return {'rss_gib': psutil.Process().memory_info().rss / 2**30,
            'anonymous_rss_gib': anonymous,
            'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            'available_gib': psutil.virtual_memory().available / 2**30,
            'host_disk_free_gib': psutil.disk_usage('/mnt/c' if Path('/mnt/c').exists() else str(Path(__file__).resolve().parents[1])).free / 2**30}


def resource_reason(state, low_samples=0):
    """Небольшой резерв предотвращает падение всего WSL."""
    if state['host_disk_free_gib'] < 5:
        return 'disk_reserve'
    if state.get('anonymous_rss_gib', state['rss_gib']) > 8.75 or low_samples >= 3:
        return 'memory_reserve'
    return None


def arm_guard(folder, stage):
    """Worker защищен даже при закрытии управляющего чата."""
    stopped = threading.Event()

    def watch():
        low_samples = 0
        ticks = 0
        while not stopped.wait(1):
            state = resource_state()
            ticks += 1
            if ticks % 5 == 0:
                write_json(folder / 'resources.json', {'stage': stage, 'updated_utc':
                           datetime.now(timezone.utc).isoformat(), **state})
            low_samples = low_samples + 1 if state['available_gib'] < 1.25 else 0
            reason = resource_reason(state, low_samples)
            if reason:
                write_json(folder / 'stopped.json', {'stage': stage, 'reason': reason, **state})
                os._exit(2)

    threading.Thread(target=watch, daemon=True).start()
    return stopped


def make_protocol(args):
    """Заморозить выбор признаков, исходники и прежний ручной контроль."""
    selected = read_json(SELECTION)
    if not selected['features'] or len(set(selected['features'])) != len(selected['features']):
        raise ValueError('Нужен непустой набор разных выбранных признаков')
    sources = {name: file_record(ROOT / 'src' / name) for name in SOURCE_FILES}
    inputs = {'selection': file_record(SELECTION), 'listens': file_record(ROOT / 'data/listens.parquet')}
    for split in ['train', 'val']:
        base = legacy_featuretools_dir()
        inputs[split + '_targets'] = file_record(base / 'catboost_pools' / f'{split}_targets.npz')
        inputs[split + '_metadata'] = file_record(
            base / f'full_{split}_cumulative' / f'{split}_all_cumulative_metadata.parquet')
    for name in ['Users', 'Items', 'UserItem', 'UserArtist']:
        inputs[name + '_definitions'] = file_record(DEFINITIONS / (name + '.json'))
    baseline = {name: file_record(BASELINE / name) for name in
                ['requested_params.json', 'metrics.json', 'model.cbm', 'group_ndcg.parquet', 'val_score.npy']}
    if args.reuse_train_run:
        inputs['reused_train_report'] = file_record(ROOT / 'results' / RELATIVE / args.reuse_train_run / 'generation_train.json')
        inputs['reused_train_protocol'] = file_record(ROOT / 'results' / RELATIVE / args.reuse_train_run / 'protocol.json')
    return {'run_name': args.run_name, 'reuse_train_run': args.reuse_train_run,
            'features': selected['features'],
            'indices': selected['indices'], 'feature_count': len(selected['features']),
            'history_rule': 'end_ts < prediction_start_ts; [t-window,t)',
            'test_used': False, 'selection_frozen_before_full_fit': True,
            'validation_reused': True, 'parameters': read_json(BASELINE / 'requested_params.json'),
            'history_chunk': args.history_chunk, 'query_chunk': args.query_chunk,
            'versions': {'Python': platform.python_version(), **{name: importlib.metadata.version(name)
                         for name in ['numpy', 'pandas', 'pyarrow', 'featuretools', 'catboost', 'numba']}},
            'sources': sources, 'inputs': inputs, 'baseline': baseline}


def initialize(args, folder, data):
    requested = make_protocol(args)
    if (folder / 'protocol.json').exists():
        if read_json(folder / 'protocol.json') != requested:
            raise ValueError('Протокол или исходники изменились: нужно новое имя запуска')
    else:
        if any(folder.iterdir()) or data.exists():
            raise FileExistsError('Новая папка уже содержит данные')
        data.mkdir(parents=True)
        write_json(folder / 'protocol.json', requested)
    return requested


def progress_writer(folder, stage):
    def update(details):
        write_json(folder / 'progress.json', {'stage': stage, 'updated_utc':
                   datetime.now(timezone.utc).isoformat(), **resource_state(), **details})
    return update



def verify_sample_values(split, generation):
    """Сверить полную матрицу с прежними запросами подвыборки по log_id.

    Для STD допустима малая погрешность: порядок промежуточных запросов
    меняет округление Welford, но не состав доступной истории.
    """
    import numpy as np
    source = DEFINITIONS.parent
    selected = read_json(SELECTION)
    with np.load(saved_path(generation['targets_path'])) as saved:
        ids = saved['log_id']
    with np.load(source / (split + '_targets.npz')) as saved:
        sample_ids = saved['log_id']
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], sample_ids)
    np.testing.assert_array_equal(ids[order[positions]], sample_ids)
    full = np.load(saved_path(generation['matrix_path']), mmap_mode='r')
    previous = np.load(source / (split + '_X.npy'), mmap_mode='r')
    expected = np.asarray(previous[:, selected['indices']])
    rows = order[positions]
    actual = np.empty(expected.shape, dtype='float32')
    for column in range(full.shape[1]):
        actual[:, column] = full[rows, column]
        full._mmap.madvise(mmap.MADV_DONTNEED)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5, equal_nan=True)
    finite = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual[finite].astype('float64') - expected[finite])
    return {'rows': len(sample_ids), 'values': int(actual.size), 'passed': True,
            'rtol': 1e-5, 'atol': 2e-5,
            'max_absolute_difference': float(delta.max(initial=0)),
            'nonidentical_finite_values': int(np.count_nonzero(delta))}



def reuse_train(source_run):
    """Переиспользовать проверенный train после сбоя только на этапе квантования."""
    source = ROOT / 'results' / RELATIVE / source_run
    report = read_json(source / 'generation_train.json')
    protocol = read_json(source / 'protocol.json')
    if not report.get('completed') or report['features'] != read_json(SELECTION)['features']:
        raise ValueError('Переиспользуемый train не готов или содержит другие признаки')
    required = ['temporal_full_data.py', 'temporal_dfs_features.py', 'cumulative_features.py',
                'temporal_sample_data.py', 'project_artifacts.py']
    for name in ['listens', 'train_targets', 'train_metadata', 'selection']:
        record = protocol['inputs'][name]
        if sha256(saved_path(record['path'])) != record['sha256']:
            raise ValueError('Исходные данные переиспользуемого train изменились: ' + name)
    for name in required:
        if protocol['sources'][name]['sha256'] != sha256(ROOT / 'src' / name):
            raise ValueError('Генератор переиспользуемого train изменился: ' + name)
    if sha256(saved_path(report['matrix_path'])) != report['matrix_sha256']:
        raise ValueError('Переиспользуемая матрица повреждена')
    return {**report, 'reused_from_run': source_run}


def generate(args, folder, data, split):
    from temporal_full_data import generate_full_split
    if split == 'train' and args.reuse_train_run:
        report = reuse_train(args.reuse_train_run)
        write_json(folder / 'generation_train.json', report)
        return report
    report = generate_full_split(split, data / split, selection_path=SELECTION,
                                 definition_dir=DEFINITIONS, history_chunk=args.history_chunk,
                                 query_chunk=args.query_chunk,
                                 progress=progress_writer(folder, 'generate_' + split), resume=True)
    report['sample_verification'] = verify_sample_values(split, report)
    report['matrix_sha256'] = sha256(saved_path(report['matrix_path']))
    write_json(folder / ('generation_' + split + '.json'), report)
    return report


def quantize(folder, data, split):
    from temporal_column_quantization import prepare_quantized_split
    generation = read_json(folder / ('generation_' + split + '.json'))
    borders = None if split == 'train' else saved_path(read_json(folder / 'quantization_train.json')['borders_path'])
    report = prepare_quantized_split(saved_path(generation['matrix_path']), saved_path(generation['targets_path']),
                                    generation['features'], data / 'quantized', split,
                                    input_borders=borders,
                                    progress=progress_writer(folder, 'quantize_' + split),
                                    matrix_sha256=generation['matrix_sha256'])
    write_json(folder / ('quantization_' + split + '.json'), report)
    return report


def fit(folder):
    from temporal_full_training import fit_ranker
    protocol = read_json(folder / 'protocol.json')
    train = read_json(folder / 'quantization_train.json')
    val = read_json(folder / 'quantization_val.json')
    return fit_ranker(saved_path(train['pool_path']), saved_path(val['pool_path']), saved_path(val['targets_path']),
                      protocol['features'], folder / 'model', protocol['parameters'])


def compare(folder):
    """Сопоставить готовую модель с тем же full manual 1600/80."""
    import pandas as pd
    protocol = read_json(folder / 'protocol.json')
    result = read_json(folder / 'stages/fit.json')
    new = read_json(folder / 'model/metrics.json')
    old = read_json(BASELINE / 'metrics.json')
    baseline_groups = pd.read_parquet(BASELINE / 'group_ndcg.parquet')
    actual_groups = pd.read_parquet(folder / 'model/group_ndcg.parquet')
    for field in ['group_id', 'day', 'group_size']:
        if not baseline_groups[field].equals(actual_groups[field]):
            raise ValueError('Группы validation отличаются от manual: ' + field)
    rows = []
    for name in ['NDCG@10', 'QueryAUC']:
        rows.append({'metric': name, 'manual': old[name], 'temporal_dfs_462': new[name],
                     'difference': new[name] - old[name],
                     'relative_loss_percent': 100 * (old[name] - new[name]) / old[name]})
    pd.DataFrame(rows).to_csv(folder / 'comparison.csv', index=False)
    report = {'completed': True, 'comparison': rows, 'validation_rows': int(actual_groups['group_size'].sum()),
              'validation_groups': len(actual_groups), 'parameters': protocol['parameters'],
              'feature_count': len(protocol['features']), 'test_used': False, 'validation_reused': True,
              'fit_seconds': result.get('fit_seconds'), 'new_artifacts': {
                  name: file_record(folder / 'model' / name) for name in
                  ['model.cbm', 'metrics.json', 'evaluation.json', 'val_score.npy']}}
    write_json(folder / 'result.json', report)
    text = '\n'.join([f"# Полный временной DFS: {len(protocol['features'])} признаков", '',
        'Ручной контроль и новый ранкер используют одинаковые строки и параметры 1600/80.', '',
        f"| Метрика | Ручные признаки | DFS {len(protocol['features'])} | Потеря DFS, % |",
        '| --- | ---: | ---: | ---: |'] + [
        f"| {row['metric']} | {row['manual']:.6f} | {row['temporal_dfs_462']:.6f} | {row['relative_loss_percent']:.3f} |"
        for row in rows] + ['', 'Признаки выбраны и обучение остановлено по validation; использован восстановленный start_ts.', ''])
    (folder / 'REPORT.md').write_text(text)
    return report



def verify_frozen(folder):
    """Не смешивать версии кода и входов между отдельными worker."""
    protocol = read_json(folder / 'protocol.json')
    for group in ['sources', 'inputs', 'baseline']:
        for name, record in protocol[group].items():
            path = saved_path(record['path'])
            state = path.stat()
            if state.st_size != record['bytes'] or state.st_mtime_ns != record['mtime_ns']:
                raise ValueError('Замороженный файл изменился: ' + name)
            if group == 'sources' and sha256(path) != record['sha256']:
                raise ValueError('Исходник изменился: ' + name)
    for name, version in protocol['versions'].items():
        actual = platform.python_version() if name == 'Python' else importlib.metadata.version(name)
        if actual != version:
            raise ValueError('Версия зависимости изменилась: ' + name)


def run_worker(args, folder, data):
    verify_frozen(folder)
    stop = arm_guard(folder, args.stage)
    started = time.monotonic()
    try:
        if args.stage.startswith('generate_'):
            value = generate(args, folder, data, args.stage.split('_')[1])
        elif args.stage.startswith('quantize_'):
            value = quantize(folder, data, args.stage.split('_')[1])
        elif args.stage == 'fit':
            value = fit(folder)
        else:
            value = compare(folder)
        write_json(folder / 'stages' / (args.stage + '.json'),
                   {**value, 'completed': True, 'elapsed_seconds': time.monotonic() - started,
                    **resource_state()})
    finally:
        stop.set()


def run_queue(args, folder, data):
    """Каждый крупный этап завершается отдельным процессом и освобождает RAM."""
    protocol = initialize(args, folder, data)
    (folder / 'stages').mkdir(exist_ok=True)
    (folder / 'logs').mkdir(exist_ok=True)
    started = time.monotonic()
    monitor = subprocess.Popen([sys.executable, str(ROOT / 'src/watch_temporal_full.py'),
                               '--parent-pid', str(os.getpid()), '--run-name', args.run_name])
    try:
        _run_stages(args, folder, data, started)
    finally:
        if monitor.poll() is None:
            monitor.terminate()
        monitor.wait()


def _run_stages(args, folder, data, started):
    for stage in STAGES:
        result_path = folder / 'stages' / (stage + '.json')
        if result_path.exists() and read_json(result_path).get('completed'):
            continue
        state = resource_state()
        if state['available_gib'] < 2 or state['host_disk_free_gib'] < 8:
            raise RuntimeError('Недостаточный резерв перед этапом: ' + str(state))
        write_json(folder / 'status.json', {'completed': False, 'stage': stage,
                   'elapsed_seconds': time.monotonic() - started, **state})
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-name', args.run_name,
                   '--stage', stage, '--history-chunk', str(args.history_chunk),
                   '--query-chunk', str(args.query_chunk)]
        if args.reuse_train_run:
            command += ['--reuse-train-run', args.reuse_train_run]
        print('START', stage, datetime.now(timezone.utc).isoformat(), flush=True)
        with (folder / 'logs' / (stage + '.log')).open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait()
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
                raise
        if code:
            write_json(folder / 'status.json', {'completed': False, 'stage': stage,
                       'exit_code': code, 'elapsed_seconds': time.monotonic() - started,
                       **resource_state()})
            raise RuntimeError(f'Этап {stage} завершился с кодом {code}; подробности в logs')
        print('DONE', stage, flush=True)
    write_json(folder / 'status.json', {'completed': True, 'stage': 'finished',
               'elapsed_seconds': time.monotonic() - started, **resource_state()})
    print('FINISHED', folder / 'REPORT.md', flush=True)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--run-name', required=True)
    command.add_argument('--legacy-run', default=os.environ.get('TLAB_EXPERIMENT_RUN'))
    command.add_argument('--selection-run', default=os.environ.get('TLAB_TEMPORAL_SELECTION_RUN', 'temporal_used_20261023_v1'))
    command.add_argument('--definitions-run', default=os.environ.get('TLAB_TEMPORAL_DEFINITIONS_RUN', 'temporal_topk_20261023_v2'))
    command.add_argument('--baseline-dir', default=os.environ.get('TLAB_TEMPORAL_BASELINE_DIR',
        'results/tuning/optuna_20261023_2h_v2/queue/control_manual_ranker/attempt_0001'))
    command.add_argument('--stage', choices=['all'] + STAGES, default='all')
    command.add_argument('--status', action='store_true')
    command.add_argument('--reuse-train-run')
    command.add_argument('--history-chunk', type=int, default=500_000)
    command.add_argument('--query-chunk', type=int, default=50_000)
    return command


def main():
    args = parser().parse_args()
    configure_sources(args)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_name):
        raise ValueError('Некорректное имя запуска')
    if args.reuse_train_run and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.reuse_train_run):
        raise ValueError('Некорректное имя источника train')
    if min(args.history_chunk, args.query_chunk) <= 0:
        raise ValueError('Размеры порций должны быть положительными')
    folder = ROOT / 'results' / RELATIVE / args.run_name
    data = ROOT / 'data' / RELATIVE / args.run_name
    if args.status:
        for name in ['status', 'progress', 'resources', 'result', 'stopped']:
            path = folder / (name + '.json')
            if path.exists():
                print(name, json.dumps(read_json(path), ensure_ascii=False))
        return
    folder.mkdir(parents=True, exist_ok=True)
    if args.stage != 'all':
        run_worker(args, folder, data)
        return
    # Lock лежит снаружи папки результатов и не мешает проверке нового namespace.
    with (folder.parent / (args.run_name + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_queue(args, folder, data)


if __name__ == '__main__':
    main()
