"""Временной DFS: широкие кандидаты на подвыборке и выбор top-k по validation."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import re
import resource
import threading
import time

import featuretools as ft
import numpy as np
import pandas as pd
import psutil

from temporal_dfs_features import build_temporal_definitions, calculate_temporal
from temporal_feature_selection import select_top_k
from temporal_sample_data import choose_sample, load_branch_history
from replay_paths import configure_legacy, legacy_featuretools_dir

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = [('Users', 'uid'), ('Items', 'item_id'),
            ('UserItem', 'user_item_id'), ('UserArtist', 'user_artist_id')]
SOURCE_FILES = ['run_temporal_selection.py', 'temporal_dfs_features.py',
                'temporal_feature_selection.py', 'temporal_sample_data.py',
                'train_dfs_catboost.py', 'cumulative_features.py', 'replay_paths.py']


def write_json(path, value):
    """Атомарно сохранить отчет, который можно читать во время работы."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    """Посчитать контрольную сумму, читая файл небольшими порциями."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def parser():
    """Описать явные ограничения первого эксперимента на подвыборке."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-name', required=True)
    result.add_argument('--legacy-run', help='Имя входного повторного запуска DFS130')
    result.add_argument('--train-rows', type=int, default=10000)
    result.add_argument('--val-rows', type=int, default=5000)
    result.add_argument('--k-values', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    result.add_argument('--iterations', type=int, default=300)
    result.add_argument('--budget-minutes', type=float, default=30)
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--prepare-only', action='store_true')
    result.add_argument('--selection-only', action='store_true')
    return result


def run_paths(args):
    """Разместить данные и результаты отдельно от старых экспериментов."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_name):
        raise ValueError('Имя запуска: 1–64 буквы, цифры, дефис или подчеркивание')
    if min(args.train_rows, args.val_rows, args.iterations, args.threads) <= 0:
        raise ValueError('Размеры выборок, число деревьев и потоков должны быть положительными')
    if args.budget_minutes <= 0 or not np.isfinite(args.budget_minutes):
        raise ValueError('Нужен конечный положительный бюджет времени')
    if args.prepare_only and args.selection_only:
        raise ValueError('Нужно выбрать один этап')
    relative = Path('experiment-1/temporal_runs') / args.run_name
    return ROOT / 'data' / relative, ROOT / 'results' / relative


def check_resources():
    """Оставить резерв WSL, прежде чем начинать следующий крупный этап."""
    if psutil.virtual_memory().available < 2 * 2**30:
        raise MemoryError('Осталось менее 2 GiB свободной памяти WSL')


def start_guard(folder, started, budget_minutes):
    """Остановить только свой процесс при превышении времени или нехватке RAM."""
    stop = threading.Event()
    process = psutil.Process()

    def watch():
        low_memory_count = 0
        while not stop.wait(1):
            low_memory = psutil.virtual_memory().available < 1.5 * 2**30
            low_memory_count = low_memory_count + 1 if low_memory else 0
            reason = None
            if time.monotonic() - started > budget_minutes * 60:
                reason = 'time_budget'
            elif low_memory_count >= 3 or process.memory_info().rss > 8.5 * 2**30:
                reason = 'memory_budget'
            if reason:
                # Отдельный файл не конфликтует с атомарным основным статусом.
                write_json(folder / 'stopped.json', {'completed': False, 'reason': reason,
                           'elapsed_seconds': time.monotonic() - started})
                os._exit(2)

    threading.Thread(target=watch, daemon=True).start()
    return stop


def update_status(folder, phase, started, **details):
    """Сохранить прогресс и затраты без подмены научных результатов."""
    value = {'completed': False, 'phase': phase,
             'elapsed_seconds': time.monotonic() - started,
             'rss_gib': psutil.Process().memory_info().rss / 2**30,
             'max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
             'available_gib': psutil.virtual_memory().available / 2**30,
             'test_used': False, **details}
    write_json(folder / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def save_sample(data_dir, split, queries, targets):
    """Сохранить точные запросы и цели, чтобы продолжение использовало те же строки."""
    queries.to_parquet(data_dir / f'{split}_queries.parquet', index=False)
    np.savez_compressed(data_dir / f'{split}_targets.npz', **targets)


def build_matrices(args, data_dir, folder, started):
    """Выбрать группы и последовательно заполнить широкие матрицы по четырем ветвям."""
    samples = {}
    sources = {}
    for offset, split in enumerate(['train', 'val']):
        queries, targets, provenance = choose_sample(split, getattr(args, f'{split}_rows'),
                                                      args.seed + offset)
        samples[split] = queries
        sources[split] = provenance
        save_sample(data_dir, split, queries, targets)
    queries = pd.concat([samples['train'], samples['val']], ignore_index=True)
    train_rows = len(samples['train'])
    definitions = {}
    feature_names = ['is_organic', 'track_length_seconds']
    definition_dir = data_dir / 'definitions'
    definition_dir.mkdir()
    for parent, key in BRANCHES:
        features = build_temporal_definitions(parent, key)
        definitions[parent] = features
        ft.save_features(features, str(definition_dir / f'{parent}.json'))
        feature_names.extend(parent + '__' + feature.get_name() for feature in features)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('Имена кандидатов должны быть уникальными')
    arrays = {}
    for split in ['train', 'val']:
        arrays[split] = np.lib.format.open_memmap(
            data_dir / f'{split}_X.npy', mode='w+', dtype='float32',
            shape=(len(samples[split]), len(feature_names)), fortran_order=True)
        arrays[split][:, :2] = samples[split][feature_names[:2]].to_numpy(dtype='float32')
    timings = {}
    left = 2
    for parent, key in BRANCHES:
        check_resources()
        update_status(folder, 'features_' + parent, started, candidates=len(feature_names),
                      train_rows=train_rows, val_rows=len(samples['val']))
        branch_started = time.monotonic()
        history = load_branch_history(queries, key)
        history_rows = len(history)
        width = len(definitions[parent])
        path = data_dir / f'{parent}_scratch.npy'
        matrix = np.lib.format.open_memmap(path, mode='w+', dtype='float32',
                                          shape=(len(queries), width), fortran_order=True)
        calculate_temporal(history, queries, parent, key, definitions[parent], output=matrix)
        for column in range(width):
            arrays['train'][:, left + column] = matrix[:train_rows, column]
            arrays['val'][:, left + column] = matrix[train_rows:, column]
        for array in arrays.values():
            array.flush()
        del matrix, history
        gc.collect()
        path.unlink()
        timings[parent] = {'seconds': time.monotonic() - branch_started,
                           'history_rows': history_rows, 'features': width}
        write_json(folder / 'generation_timings.json', timings)
        left += width
    for array in arrays.values():
        array.flush()
    del arrays
    gc.collect()
    prepared = {'completed': True, 'candidate_count': len(feature_names),
                'train_rows': train_rows, 'val_rows': len(samples['val']),
                'sources': sources, 'feature_names': feature_names,
                'data_artifacts': {name: sha256(data_dir / name) for name in
                    ['train_X.npy', 'val_X.npy', 'train_targets.npz', 'val_targets.npz',
                     'train_queries.parquet', 'val_queries.parquet']}}
    write_json(folder / 'preparation.json', prepared)
    return prepared


def protocol(args):
    """Зафиксировать пространство кандидатов, отбор, версии и исследовательские ограничения."""
    return {'run_name': args.run_name, 'legacy_inputs_dir': str(legacy_featuretools_dir()),
            'windows_days': list(range(1, 31)),
            'budget_minutes': args.budget_minutes,
            'all_history': True, 'freshness': ['time_since_last', 'time_since_first'],
            'freshness_unit': 'days', 'feature_cap': None, 'dfs_depth': 2,
            'candidate_grammar': 'all original DFS formulas; products within one window',
            'cross_window_products': False, 'time_rule': 'end_ts < prediction.time',
            'window_rule': 'prediction.time - window <= end_ts < prediction.time',
            'selection': {'kind': 'YetiRank', 'importance': 'LossFunctionChange on train',
                          'k_values': args.k_values, 'all_features_reference': True,
                          'iterations': args.iterations, 'seed': args.seed,
                          'train_row_cap': args.train_rows, 'val_row_cap': args.val_rows,
                          'threads': args.threads},
            'versions': {'featuretools': ft.__version__, 'numpy': np.__version__,
                         'pandas': pd.__version__},
            'source_sha256': {name: sha256(ROOT / 'src' / name) for name in SOURCE_FILES},
            'history_source': {'path': str(ROOT / 'data/listens.parquet'),
                               'bytes': (ROOT / 'data/listens.parquet').stat().st_size,
                               'sha256': sha256(ROOT / 'data/listens.parquet')},
            'test_used': False, 'limitations': [
                'Subsample whole ranking groups; full past history retained.',
                'Validation reused for early stopping and selection of k.',
                'One seed and a finite training cap; exploratory selection.',
                'Selected k has not been validated after full-data refit.',
                'Timestamp-as-end interpretation remains a project assumption.']}


def main(argv=None):
    """Подготовить данные и выполнить ограниченный первый отбор без финального test."""
    args = parser().parse_args(argv)
    configure_legacy(args.legacy_run)
    data_dir, folder = run_paths(args)
    requested = protocol(args)
    if args.selection_only:
        saved = json.loads((folder / 'protocol.json').read_text())
        if saved != requested:
            raise ValueError('Конфигурация или исходники изменились: нужно новое имя запуска')
        prepared = json.loads((folder / 'preparation.json').read_text())
        for name, checksum in prepared['data_artifacts'].items():
            if sha256(data_dir / name) != checksum:
                raise ValueError('Подготовленные данные изменились: ' + name)
    else:
        if folder.exists() or data_dir.exists():
            raise FileExistsError('Используйте новое имя запуска; готовые файлы не перезаписываются')
        folder.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        write_json(folder / 'protocol.json', requested)
    elapsed_before = 0.0
    if args.selection_only:
        previous_status = json.loads((folder / 'status.json').read_text())
        if previous_status['phase'] != 'prepared':
            raise ValueError('selection-only допустим только после успешного prepare-only')
        elapsed_before = previous_status['elapsed_seconds']
    started = time.monotonic() - elapsed_before
    stop = start_guard(folder, started, args.budget_minutes)
    try:
        if not args.selection_only:
            update_status(folder, 'preparation', started)
            prepared = build_matrices(args, data_dir, folder, started)
        if args.prepare_only:
            update_status(folder, 'prepared', started, candidates=prepared['candidate_count'])
            return
        check_resources()
        update_status(folder, 'selection', started, candidates=prepared['candidate_count'])
        train = np.load(data_dir / 'train_X.npy', mmap_mode='r')
        val = np.load(data_dir / 'val_X.npy', mmap_mode='r')
        targets = {}
        for split in ['train', 'val']:
            with np.load(data_dir / f'{split}_targets.npz') as saved:
                targets[split] = {name: saved[name] for name in saved.files}
        result = select_top_k(train, val, targets['train'], targets['val'],
                              prepared['feature_names'], folder / 'selection',
                              k_values=args.k_values, iterations=args.iterations,
                              seed=args.seed, threads=args.threads)
        write_json(folder / 'result.json', result)
        update_status(folder, 'finished', started, completed=True,
                      candidates=prepared['candidate_count'], result_path=str(folder / 'result.json'))
    except Exception as error:
        update_status(folder, 'failed', started, error=repr(error))
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
