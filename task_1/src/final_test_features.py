"""Признаки и фиксированное квантование финального test без обучения моделей."""
from pathlib import Path
import gc
import json
import time

import catboost
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from catboost import Pool

from project_artifacts import TASK_ROOT, file_hash
from temporal_full_data import (
    BRANCHES, _branch_matrix, _export_branch, _release_heap,
    load_selected_definitions, release_mapping,
)
from temporal_sample_data import load_branch_history, _validate_queries
from temporal_full_training import load_targets, _check_pool, _prepare_request
from temporal_column_quantization import (
    SUPPORTED_CATBOOST, _border_columns, _column_record,
    _merge_native_columns, _release,
)
from user_session_features import SESSION_FEATURES, write_session_feature_matrix

ROOT = TASK_ROOT
ALIGN_COLUMNS = ('uid', 'item_id', 'start_ts', 'label', 'group_id',
                 'is_organic', 'track_length_seconds')


def _write_json(path, record):
    """Атомарно записать отчёт в новой папке test."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.incomplete')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _assert_artifact(record):
    path = Path(record['path'])
    if file_hash(path) != record['sha256']:
        raise ValueError('Изменился зафиксированный источник: ' + str(path))
    return path


def _read_queries(data_dir):
    """Сверить canonical-запросы с едиными целями и восстановить nullable artist key."""
    data_dir = Path(data_dir)
    targets = load_targets(data_dir / 'targets.npz')
    queries = pd.read_parquet(data_dir / 'queries.parquet')
    if 'source_position' not in queries:
        raise ValueError('Нужна позиция среди всех исходных test-строк')
    source = queries['source_position'].to_numpy()
    if source.dtype.kind not in 'iu' or (source < 0).any() or len(np.unique(source)) != len(source):
        raise ValueError('Некорректные source_position')
    if 'time' not in queries and 'start_ts' in queries:
        queries['time'] = queries['start_ts']
    _validate_queries(queries, targets)
    if 'start_ts' in queries:
        np.testing.assert_array_equal(queries['start_ts'], queries['time'])
    artist_keys = pd.array(queries['user_artist_id'], dtype='Int64')
    artist_keys[queries['artist_id'].to_numpy() < 0] = pd.NA
    queries['user_artist_id'] = artist_keys
    return queries, targets


def _manual_columns(matrices, names, queries, raw_path, feature_path, progress=None):
    """Сопоставить manual со всеми split-строками, затем выбрать canonical позиции.

    Каждая исходная test-строка проверяется до удаления постоянных групп.
    Совпадающие uid/item/time не используются как ключ: сохраняется номер строки.
    """
    raw = pd.read_parquet(raw_path, columns=list(ALIGN_COLUMNS))
    positions = queries['source_position'].to_numpy(dtype='int64')
    if positions.max() >= len(raw):
        raise ValueError('source_position выходит за исходные test-строки')
    for name in ALIGN_COLUMNS:
        column = 'time' if name == 'start_ts' else name
        np.testing.assert_array_equal(raw[name].to_numpy()[positions], queries[column].to_numpy(),
                                      err_msg='canonical ' + name)
    order = np.argsort(positions, kind='stable')
    sorted_positions = positions[order]
    offset = 0
    seen = np.zeros(len(queries), dtype='bool')
    columns = list(dict.fromkeys(list(names) + list(ALIGN_COLUMNS) + ['split']))
    for batch in pq.ParquetFile(feature_path).iter_batches(batch_size=100_000, columns=columns):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.equal(table['split'], 'test'))
        count = len(table)
        if offset + count > len(raw):
            raise ValueError('В manual больше test-строк, чем в samples')
        for name in ALIGN_COLUMNS:
            np.testing.assert_array_equal(table[name].to_numpy(), raw[name].to_numpy()[offset:offset+count],
                                          err_msg='manual all rows ' + name)
        left, right = np.searchsorted(sorted_positions, [offset, offset + count])
        destination = order[left:right]
        local = sorted_positions[left:right] - offset
        if len(destination):
            for column, name in enumerate(names):
                values = table[name].to_numpy().astype('float32', copy=False)
                if np.isinf(values).any():
                    raise ValueError('Бесконечные manual значения: ' + name)
                for matrix in matrices:
                    matrix[destination, column] = values[local]
            seen[destination] = True
            for matrix in matrices:
                release_mapping(matrix)
        offset += count
        if progress and count:
            progress({'stage': 'manual42', 'rows': offset, 'total_rows': len(raw)})
    if offset != len(raw) or not seen.all():
        raise ValueError('Manual и исходные test-строки не совпали')
    return {'all_split_rows': offset, 'retained_rows': len(queries),
            'all_alignment_columns_checked': list(ALIGN_COLUMNS)}


def prepare_manual(data_dir, result_dir, plan, progress=None):
    """Записать общие manual42 и manual49, сохранив порядок всех canonical строк."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    report_path = result_dir / 'generation_manual.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['completed']:
            for schema in ['manual42', 'manual49']:
                _assert_artifact(report['matrices'][schema])
            if report['targets_sha256'] != file_hash(data_dir / 'targets.npz'):
                raise ValueError('Изменились canonical targets')
            return report
        raise FileExistsError('Незавершённая manual генерация требует разбора')
    queries, targets = _read_queries(data_dir)
    base_names, names = plan['schemas']['manual42'], plan['schemas']['manual49']
    if names != base_names + list(SESSION_FEATURES):
        raise ValueError('Неверная схема 42+7')
    paths = {schema: data_dir / schema / 'test_X.npy' for schema in ['manual42', 'manual49']}
    matrices = {}
    for schema, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        matrices[schema] = np.lib.format.open_memmap(
            path, mode='w+', dtype='float32', shape=(len(queries), len(plan['schemas'][schema])),
            fortran_order=True)
    started = time.perf_counter()
    alignment = _manual_columns(list(matrices.values()), base_names, queries,
                                data_dir / 'raw_queries.parquet', ROOT / 'data/features.parquet', progress)
    session_path = data_dir / 'manual49/test_session_X.npy'
    session = write_session_feature_matrix(
        queries[['uid', 'time']].rename(columns={'time': 'start_ts'}),
        ROOT / 'data/listens.parquet', session_path)
    values = np.load(session_path, mmap_mode='r', allow_pickle=False)
    for column in range(len(SESSION_FEATURES)):
        matrices['manual49'][:, len(base_names) + column] = values[:, column]
        release_mapping(matrices['manual49'])
    del values
    for matrix in matrices.values():
        release_mapping(matrix)
    matrices.clear()
    gc.collect()
    report = {'completed': True, 'split': 'test', 'test_used': True,
              'rows': len(queries), 'matrices': {
                  schema: {'path': str(path), 'sha256': file_hash(path)}
                  for schema, path in paths.items()},
              'targets_sha256': file_hash(data_dir / 'targets.npz'),
              'queries_sha256': file_hash(data_dir / 'queries.parquet'),
              'alignment': alignment, 'session': session,
              'elapsed_seconds': time.perf_counter() - started}
    _write_json(report_path, report)
    return report


def prepare_dfs(data_dir, result_dir, plan, progress=None,
                query_chunk=50_000, history_chunk=500_000):
    """Посчитать выбранные формулы прежним движком, порциями целых сущностей."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    folder = data_dir / 'dfs462'
    folder.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / 'generation_dfs462.json'
    matrix_path = folder / 'test_X.npy'
    queries, targets = _read_queries(data_dir)
    provenance = plan['upstream_data_provenance_from_full_run']
    selection = _assert_artifact(provenance['selection'])
    definition_paths = [_assert_artifact(provenance[parent + '_definitions']) for parent, _ in BRANCHES]
    if len({path.parent for path in definition_paths}) != 1:
        raise ValueError('Definitions должны лежать вместе')
    names, definitions = load_selected_definitions(selection, definition_paths[0].parent)
    if names != plan['schemas']['dfs462']:
        raise ValueError('Порядок DFS-признаков изменился')
    expected = {'targets_sha256': file_hash(data_dir / 'targets.npz'),
                'queries_sha256': file_hash(data_dir / 'queries.parquet'),
                'features': names, 'rows': len(queries)}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for name, value in expected.items():
            if report[name] != value:
                raise ValueError('Изменились входы DFS checkpoint')
        if report['completed']:
            _assert_artifact(report['matrix'])
            return report
    else:
        if matrix_path.exists():
            raise FileExistsError(matrix_path)
        report = {**expected, 'completed': False, 'split': 'test', 'test_used': True, 'branches': {}}
        _write_json(report_path, report)
        initial = np.lib.format.open_memmap(matrix_path, mode='w+', dtype='float32',
                    shape=(len(queries), len(names)), fortran_order=True)
        initial.flush()
        del initial
    started = time.perf_counter()
    final = np.load(matrix_path, mmap_mode='r+', allow_pickle=False)
    if final.dtype != np.dtype('float32') or not final.flags.f_contiguous or final.shape != (len(queries), len(names)):
        raise ValueError('Неверная матрица DFS checkpoint')
    for column, name in enumerate(names):
        if '__' not in name:
            final[:, column] = queries[name].to_numpy(dtype='float32')
            release_mapping(final)
    for parent, key in BRANCHES:
        chosen = definitions[parent]
        if not chosen or report['branches'].get(parent, {}).get('completed'):
            continue
        before = time.perf_counter()
        requests = queries[['log_id', 'time', key]].copy()
        if progress:
            progress({'stage': 'history', 'branch': parent})
        history = load_branch_history(requests, key)
        _release_heap()
        scratch = folder / ('.test_' + parent + '_scratch.npy')
        branch, inverse = _branch_matrix(requests, history, parent, key, chosen,
                                         scratch, query_chunk, history_chunk, progress)
        history_rows = len(history)
        del requests, history
        _release_heap()
        _export_branch(branch, inverse, final, [column for column, _ in chosen], query_chunk)
        del branch, inverse
        scratch.unlink()
        report['branches'][parent] = {'completed': True, 'history_rows': history_rows,
            'feature_count': len(chosen), 'elapsed_seconds': time.perf_counter() - before}
        _write_json(report_path, report)
        _release_heap()
    release_mapping(final)
    del final
    report.update(completed=True, matrix={'path': str(matrix_path), 'sha256': file_hash(matrix_path)},
                  this_attempt_seconds=time.perf_counter() - started)
    _write_json(report_path, report)
    return report


def quantize_schema(data_dir, result_dir, plan, schema, progress=None):
    """Применить только фиксированные train-пороги по одному столбцу.

    Приватные column-операции уже проверены для CatBoost 1.2.10. Здесь они
    получают готовые пороги каждого столбца; test не выдаётся за train или val.
    """
    if catboost.__version__ != SUPPORTED_CATBOOST:
        raise RuntimeError('Нужна проверенная версия CatBoost ' + SUPPORTED_CATBOOST)
    if schema not in ('manual42', 'manual49', 'dfs462'):
        raise ValueError('Неизвестная физическая схема')
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    folder = data_dir / schema
    column_dir = folder / 'test_columns'
    column_dir.mkdir(parents=True, exist_ok=True)
    targets_path = data_dir / 'targets.npz'
    matrix_path = folder / 'test_X.npy'
    names = plan['schemas'][schema]
    borders_path = _assert_artifact(plan['train_borders'][schema])
    request = _prepare_request(matrix_path, targets_path, names, borders_path)
    request.update(split='test', method='frozen_native_columns_test_v1')
    request_path = result_dir / ('quantization_' + schema + '_request.json')
    report_path = result_dir / ('quantization_' + schema + '.json')
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Изменились входы квантования test')
    else:
        _write_json(request_path, request)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for kind in ['pool', 'targets', 'applied_borders']:
            if file_hash(report[kind + '_path']) != report[kind + '_sha256']:
                raise ValueError('Изменился quantized test checkpoint')
        return report
    started = time.perf_counter()
    targets = load_targets(targets_path)
    matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
    if matrix.dtype != np.dtype('float32') or not matrix.flags.f_contiguous or matrix.shape != (len(targets['label']), len(names)):
        raise ValueError('Нужна F-order float32 матрица в canonical-порядке')
    inputs = _border_columns(borders_path, len(names))
    paths, records = [], []
    for index, name in enumerate(names):
        column_paths, record = _column_record(matrix, index, name, targets, column_dir, inputs[index])
        paths.append(column_paths['pool'])
        records.append(record)
        if progress:
            progress({'stage': 'quantize_column', 'schema': schema, 'feature_index': index,
                      'rows': index + 1, 'total_rows': len(names)})
    del matrix
    _release()
    pool_path = folder / 'test.qbin'
    temporary = folder / 'test.qbin.partial'
    _merge_native_columns(paths, names, len(targets['label']), temporary)
    pool = Pool('quantized://' + str(temporary), thread_count=1)
    _check_pool(pool, targets, names)
    applied = folder / 'test_applied_borders.tsv'
    pool.save_quantization_borders(str(applied))
    if applied.read_bytes() != borders_path.read_bytes():
        raise AssertionError('Test изменил фиксированные train-пороги')
    del pool
    _release()
    temporary.replace(pool_path)
    saved_targets = folder / 'test_targets.npz'
    np.savez_compressed(saved_targets, **targets)
    report = {'completed': True, 'split': 'test', 'test_used': True, 'schema': schema,
              'features': names, 'rows': len(targets['label']),
              'pool_path': str(pool_path), 'pool_sha256': file_hash(pool_path),
              'targets_path': str(saved_targets), 'targets_sha256': file_hash(saved_targets),
              'source_targets_sha256': file_hash(targets_path),
              'applied_borders_path': str(applied), 'applied_borders_sha256': file_hash(applied),
              'frozen_train_borders': plan['train_borders'][schema],
              'quantize_seconds': sum(record['seconds'] for record in records),
              'elapsed_seconds': time.perf_counter() - started, 'training_performed': False}
    _write_json(report_path, report)
    for path in paths:
        path.unlink()
    return report
