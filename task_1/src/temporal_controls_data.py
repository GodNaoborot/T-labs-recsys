"""Общие строки и данные пяти контролей temporal DFS; исходные артефакты не меняются."""
from pathlib import Path
import os
import gc
import json
import mmap
import re
import time

import numpy as np
from catboost import Pool

from project_artifacts import TASK_ROOT
from replay_paths import project_path, saved_path, run_name
from optuna_data import (_load_canonical_targets, _group_positions, _save_targets,
                         _manual_metadata, _fill_manual_matrix, _feature_names)
from temporal_full_data import _read_full_metadata
from temporal_sample_data import _source_paths, _validate_queries
from temporal_full_training import _check_pool, _sha256
from user_session_features import SESSION_FEATURES, write_session_feature_matrix

ROOT = TASK_ROOT
FULL = ROOT / 'results/experiment-1/temporal_runs/temporal_full462_20261023_v5'
BASELINE = ROOT / 'results/tuning/optuna_20261023_2h_v2/queue/control_manual_ranker/attempt_0001'
BASE_BORDERS = ROOT / 'results/baselines/timing-20261020/train_borders.tsv'
VARIANTS = ('M-sub', 'D462-sub', 'C1', 'C2a', 'C2b')



def configure_sources(full_run=None, baseline_dir=None, base_borders=None):
    """Подключить новый full-DFS, ручной контроль и его пороги."""
    global FULL, BASELINE, BASE_BORDERS
    if full_run:
        FULL = ROOT / 'results/experiment-1/temporal_runs' / run_name(full_run)
        os.environ['TLAB_CONTROLS_FULL_RUN'] = full_run
    if baseline_dir:
        BASELINE = project_path(baseline_dir)
        os.environ['TLAB_CONTROLS_BASELINE_DIR'] = str(BASELINE)
    if base_borders:
        BASE_BORDERS = project_path(base_borders)
        os.environ['TLAB_CONTROLS_BASE_BORDERS'] = str(BASE_BORDERS)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.incomplete')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_paths(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', name):
        raise ValueError('Нужно короткое отдельное имя запуска')
    return (ROOT / 'results/experiment-1/temporal_controls' / name,
            ROOT / 'data/experiment-1/temporal_controls' / name)


def initialize(name, train_groups=40_000, seed=42):
    """Зафиксировать параметры, прежние источники и новую реализацию до расчета."""
    result, data = run_paths(name)
    result.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    scripts = ['temporal_controls_data.py', 'run_temporal_controls.py',
               'user_session_features.py', 'controls_quantization.py',
               'watch_temporal_controls.py', 'optuna_data.py', 'optuna_evaluation.py', 'replay_paths.py']
    protocol = {
        'run_name': name, 'train_groups': train_groups, 'seed': seed,
        'variants': list(VARIANTS), 'validation': 'all canonical validation rows',
        'parameters': read_json(BASELINE / 'requested_params.json'),
        'manual_features': _feature_names('manual'),
        'dfs_features': read_json(FULL / 'protocol.json')['features'],
        'session_features': list(SESSION_FEATURES),
        'history_rule': 'end_ts < start_ts; windows [start_ts-window,start_ts)',
        'quantization': {'existing_features': 'frozen original full-train borders',
                         'new_session_features': 'selected train only; 254 borders'},
        'test_used': False, 'c4_deferred': True, 'validation_reused': True,
        'source_sha256': {n: _sha256(ROOT / 'src' / n) for n in scripts},
        'source_reports_sha256': {n: _sha256(FULL / n) for n in
                                 ['protocol.json', 'generation_train.json', 'generation_val.json',
                                  'quantization_train.json', 'quantization_val.json']},
        'canonical_targets_sha256': {s: _sha256(_source_paths(s)[0]) for s in ['train', 'val']},
        'manual_borders_sha256': _sha256(BASE_BORDERS),
    }
    path = result / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Протокол или исходники изменились: используйте новое имя запуска')
    if not path.exists():
        write_json(path, protocol)
    return protocol


def load_selection(name, split):
    _, data = run_paths(name)
    with np.load(data / 'selection' / f'{split}.npz') as stored:
        arrays = {key: stored[key] for key in ['label', 'group_id', 'log_id', 'day']}
        positions = stored['positions']
    return arrays, positions


def prepare_selection(name):
    """Выбрать ровно 40 тысяч целых train-групп; validation сохранить полностью."""
    result, data = run_paths(name)
    protocol = read_json(result / 'protocol.json')
    report_path = result / 'selection.json'
    if report_path.exists():
        report = read_json(report_path)
        for split in ['train', 'val']:
            if _sha256(data / 'selection' / f'{split}.npz') != report[split]['sha256']:
                raise ValueError('Selection изменился')
        return report
    folder = data / 'selection'
    folder.mkdir(exist_ok=True)
    report = {}
    for split in ['train', 'val']:
        arrays = _load_canonical_targets(split)
        positions = (_group_positions(arrays['group_id'], protocol['train_groups'],
                                      len(arrays['label']), protocol['seed'])
                     if split == 'train' else np.arange(len(arrays['label']), dtype='int64'))
        selected = {key: value[positions] for key, value in arrays.items()}
        groups = np.unique(selected['group_id'])
        if split == 'train' and len(groups) != protocol['train_groups']:
            raise AssertionError('Не набрано заданное число целых групп')
        path = folder / f'{split}.npz'
        if path.exists():
            raise FileExistsError('Неполный selection: нужно новое имя запуска')
        _save_targets(path, selected, positions)
        np.save(folder / f'{split}_groups.npy', groups)
        report[split] = {'rows': len(positions), 'groups': len(groups), 'sha256': _sha256(path)}
        del arrays, selected, positions
        gc.collect()
    write_json(report_path, report)
    return report


def prepare_dfs(name):
    """Срезать готовый train qbin целыми группами, сохранив прежние пороги."""
    result, data = run_paths(name)
    report_path = result / 'dfs_preparation.json'
    if report_path.exists():
        return read_json(report_path)
    protocol = read_json(result / 'protocol.json')
    arrays, positions = load_selection(name, 'train')
    folder = data / 'dfs'
    folder.mkdir(exist_ok=True)
    destination = folder / 'train.qbin'
    if destination.exists():
        raise FileExistsError('Неполная подготовка DFS')
    started = time.perf_counter()
    source = read_json(FULL / 'quantization_train.json')
    pool = Pool('quantized://' + str(saved_path(source['pool_path'])), thread_count=4)
    if pool.get_feature_names() != protocol['dfs_features']:
        raise AssertionError('Изменился порядок DFS-признаков')
    selected = pool.slice(positions)
    _check_pool(selected, arrays, protocol['dfs_features'])
    selected.save(str(destination))
    selected.save_quantization_borders(str(folder / 'train_borders.tsv'))
    if _sha256(folder / 'train_borders.tsv') != source['borders_sha256']:
        raise AssertionError('Slice изменил квантование DFS')
    val_source = read_json(FULL / 'quantization_val.json')
    report = {'completed': True, 'train_pool': str(destination),
              'val_pool': str(saved_path(val_source['pool_path'])), 'features': protocol['dfs_features'],
              'rows': len(positions), 'elapsed_seconds': time.perf_counter() - started,
              'quantization_uses_original_full_train': True,
              'train_pool_sha256': _sha256(destination), 'test_used': False}
    write_json(report_path, report)
    return report


def _verify_session_probe(queries, stored, history_path):
    """Независимый pandas-oracle для 40 строк, без вызова сессионного движка."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    probe = np.unique(np.linspace(0, len(queries) - 1, min(40, len(queries)), dtype=int))
    users = set(queries.iloc[probe]['uid'].tolist())
    pieces = []
    for batch in pq.ParquetFile(history_path).iter_batches(
            batch_size=500_000, columns=['uid', 'timestamp', 'played_ratio_pct']):
        frame = pa.Table.from_batches([batch]).to_pandas()
        pieces.append(frame[frame['uid'].isin(users)])
    history = pd.concat(pieces, ignore_index=True)
    for position in probe:
        query = queries.iloc[position]
        past = history[(history.uid == query.uid) & (history.timestamp < query.start_ts)]
        past = past.sort_values('timestamp', kind='stable')
        recent = past[past.timestamp >= query.start_ts - 86400]
        if len(past):
            last = past.iloc[-1]
            values = [min(float(last.played_ratio_pct), 100), float(last.played_ratio_pct < 35),
                      float((past.tail(5).played_ratio_pct < 35).mean()),
                      float((past.tail(20).played_ratio_pct < 35).mean()),
                      float((recent.played_ratio_pct < 35).mean()) if len(recent) else np.nan,
                      len(recent), query.start_ts - last.timestamp]
        else:
            values = [np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan]
        np.testing.assert_array_equal(stored[position], np.asarray(values, dtype='float32'))
    return {'rows': len(probe), 'values': len(probe) * 7, 'exact_float32': True}


def prepare_manual_sessions(name, split):
    """Соединить прежние 42 ручных признака с семью новыми по общим log_id."""
    result, data = run_paths(name)
    report_path = result / f'manual_sessions_{split}.json'
    if report_path.exists():
        return read_json(report_path)
    protocol = read_json(result / 'protocol.json')
    arrays, _ = load_selection(name, split)
    folder = data / 'manual_sessions'
    folder.mkdir(exist_ok=True)
    output = folder / f'{split}_X.npy'
    partial = folder / f'{split}_X.incomplete.npy'
    if output.exists() or partial.exists():
        raise FileExistsError('Неполная ручная матрица: нужно новое имя запуска')
    names = protocol['manual_features'] + list(SESSION_FEATURES)
    started = time.perf_counter()
    matrix = np.lib.format.open_memmap(partial, mode='w+', dtype='float32',
                                      shape=(len(arrays['label']), len(names)), fortran_order=True)
    metadata, destination = _manual_metadata(split, arrays)
    _fill_manual_matrix(matrix[:, :42], names[:42], split, metadata, destination)
    matrix.flush()
    matrix._mmap.madvise(mmap.MADV_DONTNEED)
    del metadata, destination
    gc.collect()
    _, metadata_path = _source_paths(split)
    queries = _read_full_metadata(metadata_path, arrays['log_id'], 500_000)
    _validate_queries(queries, arrays)
    queries = queries[['uid', 'time']].rename(columns={'time': 'start_ts'})
    session_path = folder / f'{split}_session_X.npy'
    session_report = write_session_feature_matrix(queries, ROOT / 'data/listens.parquet', session_path)
    session = np.load(session_path, mmap_mode='r')
    oracle = _verify_session_probe(queries, session, ROOT / 'data/listens.parquet')
    for j in range(7):
        matrix[:, 42 + j] = session[:, j]
    matrix.flush()
    del matrix, session, queries
    gc.collect()
    partial.replace(output)
    report = {'completed': True, 'matrix_path': str(output), 'features': names,
              'rows': len(arrays['label']), 'matrix_sha256': _sha256(output),
              'session': session_report, 'independent_oracle': oracle,
              'elapsed_seconds': time.perf_counter() - started, 'test_used': False}
    write_json(report_path, report)
    return report


def variant_spec(name, variant):
    """Физические Pool сохраняются общими; маски задают фактически активные признаки."""
    result, data = run_paths(name)
    protocol = read_json(result / 'protocol.json')
    if variant not in VARIANTS:
        raise ValueError('Неизвестный контроль')
    dfs = variant in ['D462-sub', 'C1']
    names = protocol['dfs_features'] if dfs else protocol['manual_features'] + list(SESSION_FEATURES)
    ignored = ([i for i, feature in enumerate(names) if feature.startswith('Users__')]
               if variant == 'C1' else list(range(42, 49)) if variant == 'M-sub'
               else [48] if variant == 'C2a' else [])
    if dfs:
        info = read_json(result / 'dfs_preparation.json')
        train_pool, val_pool = info['train_pool'], info['val_pool']
    else:
        train_pool = read_json(data / 'manual_quantized/train_report.json')['pool_path']
        val_pool = read_json(data / 'manual_quantized/val_report.json')['pool_path']
    return {'train_pool': train_pool, 'val_pool': val_pool, 'features': names,
            'ignored_indices': ignored, 'active_count': len(names) - len(ignored),
            'active_features': [feature for i, feature in enumerate(names) if i not in ignored]}
