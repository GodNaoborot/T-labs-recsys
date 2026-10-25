"""Полные выбранные temporal-DFS признаки: история по ветвям, расчет порциями."""
from pathlib import Path
import gc
import ctypes
import json
import mmap
import time

import featuretools as ft
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from project_artifacts import TASK_ROOT, file_hash
from temporal_dfs_features import calculate_temporal
from temporal_sample_data import (_source_paths, _branch_keys, _array_hash,
                                  _validate_queries, load_branch_history)

ROOT = TASK_ROOT
BRANCHES = [('Users', 'uid'), ('Items', 'item_id'),
            ('UserItem', 'user_item_id'), ('UserArtist', 'user_artist_id')]
METADATA_COLUMNS = ['log_id', 'uid', 'item_id', 'artist_id', 'time', 'end_ts',
                    'label', 'track_length_seconds', 'is_organic', 'group_id',
                    'user_item_id', 'user_artist_id']
DEFAULT_SELECTION = ROOT / 'results/experiment-1/temporal_runs/temporal_used_20261023_v1/protocol.json'
DEFAULT_DEFINITIONS = ROOT / 'data/experiment-1/temporal_runs/temporal_topk_20261023_v2/definitions'


def release_mapping(matrix):
    """Сбросить запись и освободить резидентные страницы дисковой матрицы."""
    matrix.flush()
    if hasattr(matrix._mmap, 'madvise'):
        matrix._mmap.madvise(mmap.MADV_DONTNEED)


def _release_heap():
    gc.collect()
    # Вернуть освобожденные pandas/Arrow блоки системе перед следующей ветвью.
    allocator = ctypes.CDLL(None)
    if hasattr(allocator, 'malloc_trim'):
        allocator.malloc_trim(0)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.incomplete')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _read_full_metadata(path, log_ids, batch_size):
    """Сопоставить metadata с canonical targets без большого join и смены порядка."""
    order = np.argsort(log_ids, kind='stable')
    sorted_ids = log_ids[order]
    if len(sorted_ids) > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError('Повторные canonical log_id')
    arrays, found = {}, np.zeros(len(log_ids), dtype='bool')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=METADATA_COLUMNS):
        table = pa.Table.from_batches([batch])
        ids = table['log_id'].to_numpy()
        positions = np.searchsorted(sorted_ids, ids)
        mask = positions < len(sorted_ids)
        mask[mask] &= sorted_ids[positions[mask]] == ids[mask]
        destination = order[positions[mask]]
        if found[destination].any():
            raise ValueError('Повторные log_id в metadata')
        found[destination] = True
        for name in METADATA_COLUMNS:
            column = pc.fill_null(table[name], -1) if name == 'user_artist_id' else table[name]
            values = column.to_numpy(zero_copy_only=False)
            if name not in arrays:
                arrays[name] = np.empty(len(log_ids), dtype=values.dtype)
            arrays[name][destination] = values[mask]
    if not found.all():
        raise ValueError('Не все canonical log_id найдены в metadata')
    queries = pd.DataFrame(arrays, copy=False)
    # Составной int64 ключ не проходит через float даже при пропусках артиста.
    artist_keys = pd.array(queries['user_artist_id'], dtype='Int64')
    artist_keys[queries['artist_id'].to_numpy() < 0] = pd.NA
    queries['user_artist_id'] = artist_keys
    return queries


def load_full_queries(split, batch_size=500_000):
    """Взять все train/val canonical-строки; test этим путем не открывается."""
    targets_path, metadata_path = _source_paths(split)
    with np.load(targets_path) as source:
        targets = {name: source[name] for name in ['label', 'group_id', 'log_id']}
    targets['day'] = targets['group_id'] % 1000
    if not len(targets['log_id']) or np.any(np.diff(targets['group_id']) < 0):
        raise ValueError('Нужны непустые canonical targets в порядке групп')
    queries = _read_full_metadata(metadata_path, targets['log_id'], batch_size)
    _validate_queries(queries, targets)
    return queries, targets


def load_selected_definitions(selection_path, definition_dir):
    """Прочитать уже выбранные формулы, сохранив точный порядок столбцов."""
    selection_path, definition_dir = Path(selection_path), Path(definition_dir)
    selection = json.loads(selection_path.read_text())
    names = selection['features']
    if not names or len(set(names)) != len(names):
        raise ValueError('Список выбранных признаков пуст или содержит повторы')
    definitions, covered = {}, set()
    for parent, key in BRANCHES:
        all_features = ft.load_features(str(definition_dir / f'{parent}.json'))
        lookup = {parent + '__' + feature.get_name(): feature for feature in all_features}
        chosen = [(column, lookup[name]) for column, name in enumerate(names) if name in lookup]
        definitions[parent] = chosen
        covered.update(column for column, _ in chosen)
    raw = {'is_organic', 'track_length_seconds'}
    covered.update(column for column, name in enumerate(names) if name in raw)
    if covered != set(range(len(names))):
        raise ValueError('Выбранный признак не найден в зафиксированных definitions')
    return names, definitions


def entity_blocks(query_keys, history_keys, query_chunk, history_chunk):
    """Порции ограничены запросами и историей; одна сущность не разрывается."""
    if query_chunk <= 0 or history_chunk <= 0:
        raise ValueError('Размеры порций должны быть положительными')
    left = 0
    while left < len(query_keys):
        proposed = min(left + query_chunk, len(query_keys))
        right = int(np.searchsorted(query_keys, query_keys[proposed - 1], side='right'))
        history_left = int(np.searchsorted(history_keys, query_keys[left], side='left'))
        history_right = int(np.searchsorted(history_keys, query_keys[right - 1], side='right'))
        if history_right - history_left > history_chunk:
            last_key = history_keys[history_left + history_chunk - 1]
            narrowed = int(np.searchsorted(query_keys, last_key, side='right'))
            first_entity_end = int(np.searchsorted(query_keys, query_keys[left], side='right'))
            right = max(first_entity_end, min(right, narrowed))
            history_right = int(np.searchsorted(history_keys, query_keys[right - 1], side='right'))
        yield left, right, history_left, history_right
        left = right


def _query_layout(queries, key):
    valid = queries[key].notna().to_numpy()
    valid_rows = np.flatnonzero(valid)
    values = queries.loc[valid, key].to_numpy(dtype='int64')
    order = np.argsort(values, kind='stable')
    positions = np.r_[valid_rows[order], np.flatnonzero(~valid)]
    inverse = np.empty(len(queries), dtype='int64')
    inverse[positions] = np.arange(len(queries), dtype='int64')
    return positions, values[order], inverse


def _branch_matrix(queries, history, parent, key, chosen, scratch_path,
                   query_chunk, history_chunk, progress):
    """Считать целые сущности в дисковую ветвь, упорядоченную по ключу."""
    positions, query_keys, inverse = _query_layout(queries, key)
    history_values = history[key].to_numpy(dtype='int64')
    history_order = np.argsort(history_values, kind='stable')
    history_keys = history_values[history_order]
    shape = (len(queries), len(chosen))
    matrix = np.lib.format.open_memmap(scratch_path, mode='w+', dtype='float32', shape=shape)
    for left in range(0, len(queries), query_chunk):
        matrix[left:left + query_chunk] = np.nan
        release_mapping(matrix)
    for left, right, hleft, hright in entity_blocks(
            query_keys, history_keys, query_chunk, history_chunk):
        requests = queries.iloc[positions[left:right]]
        child = history.iloc[history_order[hleft:hright]]
        calculate_temporal(child, requests, parent, key,
                           [feature for _, feature in chosen], output=matrix[left:right])
        release_mapping(matrix)
        if progress:
            progress({'stage': 'features', 'branch': parent, 'rows': right,
                      'total_rows': len(query_keys), 'history_rows': hright - hleft})
    release_mapping(matrix)
    return matrix, inverse


def _export_branch(matrix, inverse, final, columns, query_chunk):
    """Вернуть canonical-порядок малыми порциями и сбрасывать страницы memmap."""
    for left in range(0, len(final), query_chunk):
        right = min(left + query_chunk, len(final))
        final[left:right, columns] = matrix[inverse[left:right]]
        release_mapping(final)
        # Случайное чтение scratch тоже не должно оставаться в RSS до конца ветви.
        if hasattr(matrix._mmap, 'madvise'):
            matrix._mmap.madvise(mmap.MADV_DONTNEED)


def _input_record(split, selection_path, definition_dir, names, targets):
    targets_path, metadata_path = _source_paths(split)
    files = {'selection': Path(selection_path), 'targets': targets_path,
             'metadata': metadata_path, 'history': ROOT / 'data/listens.parquet'}
    files.update({parent + '_definitions': Path(definition_dir) / f'{parent}.json'
                  for parent, key in BRANCHES})
    return {'split': split, 'features': names, 'feature_count': len(names),
            'rows': len(targets['log_id']), 'test_used': False,
            'time_rule': 'history.end_ts < query.time; window=[time-days,time)',
            'source_sha256': {name: file_hash(path) for name, path in files.items()},
            'sources': {name: str(path) for name, path in files.items()},
            'target_arrays': {name: _array_hash(value) for name, value in targets.items()}}


def generate_full_split(split, output_dir, *, selection_path=DEFAULT_SELECTION,
                        definition_dir=DEFAULT_DEFINITIONS, history_chunk=500_000,
                        query_chunk=50_000, progress=None, resume=False):
    """Сохранить выбранные признаки всех canonical train/val строк в F-order npy.

    История читается один раз на ветвь. После ветви остается checkpoint, поэтому
    повторный запуск resume=True не пересчитывает завершенные ветви. Незавершенная
    ветвь считается заново. Одновременно на диске: итоговая матрица и одна ветвь.
    """
    if history_chunk <= 0 or query_chunk <= 0:
        raise ValueError('Размеры порций должны быть положительными')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, report_path = output_dir / f'{split}_X.npy', output_dir / f'{split}_preparation.json'
    names, definitions = load_selected_definitions(selection_path, definition_dir)
    queries, targets = load_full_queries(split)
    expected = _input_record(split, selection_path, definition_dir, names, targets)
    if report_path.exists():
        if not resume:
            raise FileExistsError(report_path)
        report = json.loads(report_path.read_text())
        for name, value in expected.items():
            if report.get(name) != value:
                raise ValueError(f'Источники изменились после checkpoint: {name}')
        if not matrix_path.exists():
            raise FileNotFoundError(matrix_path)
    else:
        if matrix_path.exists():
            raise FileExistsError(matrix_path)
        report = {**expected, 'completed': False, 'branches': {}, 'matrix_path': str(matrix_path),
                  'targets_path': str(_source_paths(split)[0]), 'layout': 'float32 F-order',
                  'query_chunk': query_chunk, 'history_chunk': history_chunk}
        _write_json(report_path, report)
        initial = np.lib.format.open_memmap(matrix_path, mode='w+', dtype='float32',
                                            shape=(len(queries), len(names)), fortran_order=True)
        initial.flush()
        del initial
    started = time.perf_counter()
    final = np.load(matrix_path, mmap_mode='r+')
    if final.shape != (len(queries), len(names)) or final.dtype != np.dtype('float32') or not final.flags.f_contiguous:
        raise ValueError('Неверная форма, тип или порядок дисковой матрицы')
    if report['completed']:
        del final
        return report
    for column, name in enumerate(names):
        if '__' not in name:
            final[:, column] = queries[name].to_numpy(dtype='float32')
            release_mapping(final)
    for parent, key in BRANCHES:
        chosen = definitions[parent]
        if not chosen or report['branches'].get(parent, {}).get('completed'):
            continue
        branch_started = time.perf_counter()
        if progress:
            progress({'stage': 'history', 'branch': parent, 'split': split})
        branch_queries = queries[['log_id', 'time']].copy()
        branch_queries[key] = queries[key]
        history = load_branch_history(branch_queries, key)
        _release_heap()
        scratch_path = output_dir / f'.{split}_{parent}_scratch.npy'
        matrix, inverse = _branch_matrix(branch_queries, history, parent, key, chosen,
            scratch_path, query_chunk, history_chunk, progress)
        history_rows = len(history)
        del history, branch_queries
        _release_heap()
        _export_branch(matrix, inverse, final, [column for column, _ in chosen], query_chunk)
        del matrix, inverse
        scratch_path.unlink()
        report['branches'][parent] = {'completed': True, 'history_rows': history_rows,
            'feature_count': len(chosen), 'seconds': time.perf_counter() - branch_started}
        _write_json(report_path, report)
        _release_heap()
    release_mapping(final)
    del final
    report.update(completed=True, matrix_bytes=matrix_path.stat().st_size,
                  this_attempt_seconds=time.perf_counter() - started)
    _write_json(report_path, report)
    return report
