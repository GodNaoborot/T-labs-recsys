"""Подвыборка целых ranking-групп и доступная история для временного DFS."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from project_artifacts import TASK_ROOT, file_hash
from replay_paths import legacy_featuretools_dir

ROOT = TASK_ROOT
DAY = 86400
TARGET_FIELDS = ('label', 'group_id', 'log_id', 'day')
QUERY_COLUMNS = ['log_id', 'uid', 'item_id', 'artist_id', 'time', 'end_ts',
                 'label', 'track_length_seconds', 'is_organic', 'group_id',
                 'user_item_id', 'user_artist_id']
KEY_COLUMNS = {'uid': ['uid'], 'item_id': ['item_id'],
               'user_item_id': ['uid', 'item_id'],
               'user_artist_id': ['uid', 'artist_id']}


def _whole_group_positions(groups, row_cap, seed):
    """Случайно выбрать целые группы в лимит строк, сохранив исходный порядок."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or groups.dtype.kind not in 'iu':
        raise ValueError('Нужен непустой целочисленный массив групп')
    if row_cap <= 0 or int(row_cap) != row_cap:
        raise ValueError('Лимит строк должен быть положительным целым')
    if np.any(groups[1:] < groups[:-1]):
        raise ValueError('Исходные группы должны идти по возрастанию')
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(groups)]
    chosen, rows = [], 0
    for index in np.random.default_rng(seed).permutation(len(starts)):
        size = int(ends[index] - starts[index])
        if rows + size <= row_cap:
            chosen.append(index)
            rows += size
        if rows == row_cap:
            break
    if not chosen:
        raise ValueError('В лимит строк не поместилась ни одна целая группа')
    return np.concatenate([np.arange(starts[index], ends[index], dtype='int64')
                           for index in sorted(chosen)])


def _source_paths(split):
    """Разрешить только train/val исходного сравнения; test здесь недоступен."""
    if split not in ('train', 'val'):
        raise ValueError('Поддержаны только train и val')
    folder = legacy_featuretools_dir(ROOT)
    return (folder / 'catboost_pools' / f'{split}_targets.npz',
            folder / f'full_{split}_cumulative' / f'{split}_all_cumulative_metadata.parquet')


def _array_hash(array):
    """Зафиксировать значения вместе с типом массива."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(value.dtype.str.encode())
    digest.update(value.view('uint8'))
    return digest.hexdigest()


def _file_record(path):
    """Сохранить источник и SHA256 без загрузки файла целиком."""
    path = Path(path).resolve()
    state = path.stat()
    return {'path': str(path), 'bytes': state.st_size,
            'mtime_ns': state.st_mtime_ns, 'sha256': file_hash(path)}


def _read_selected_metadata(path, log_ids, batch_size=500_000):
    """Прочитать нужные log_id порциями, затем вернуть канонический порядок."""
    wanted = pa.array(np.asarray(log_ids, dtype='int64'))
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=QUERY_COLUMNS):
        table = pa.Table.from_batches([batch])
        selected = table.filter(pc.is_in(table['log_id'], value_set=wanted))
        if len(selected):
            parts.append(selected)
    if not parts:
        raise ValueError('Выбранные log_id отсутствуют в metadata')
    found = pa.concat_tables(parts).to_pandas()
    if not found['log_id'].is_unique:
        raise ValueError('В metadata повторяются log_id')
    positions = pd.Index(found['log_id']).get_indexer(log_ids)
    if (positions < 0).any() or len(found) != len(log_ids):
        raise ValueError('Не все целевые log_id найдены в metadata')
    return found.iloc[positions].reset_index(drop=True)


def _validate_queries(queries, targets):
    """Сверить метки, группы, время и ключи пар с исходной целевой таблицей."""
    for column in ('label', 'group_id', 'log_id'):
        np.testing.assert_array_equal(queries[column].to_numpy(), targets[column], err_msg=column)
    if not np.isin(targets['label'], [0, 1, 2]).all():
        raise ValueError('Неизвестная целевая метка')
    groups = targets['group_id']
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    labels = targets['label']
    if np.any(np.minimum.reduceat(labels, starts) == np.maximum.reduceat(labels, starts)):
        raise ValueError('В canonical targets обнаружена постоянная ranking-группа')
    uid = queries['uid'].to_numpy(dtype='int64')
    day = np.maximum(queries['time'].to_numpy(dtype='int64') // DAY, 0)
    np.testing.assert_array_equal(uid * 1000 + day, groups, err_msg='user-day')
    np.testing.assert_array_equal(day, targets['day'], err_msg='day')
    pair = (uid << 32) | queries['item_id'].to_numpy(dtype='int64')
    np.testing.assert_array_equal(pair, queries['user_item_id'].to_numpy(dtype='int64'))
    artist = queries['artist_id'].to_numpy(dtype='int64')
    artist_pair = (uid << 32) | np.maximum(artist, 0)
    artist_pair[artist < 0] = -1
    actual = queries['user_artist_id'].to_numpy(dtype='int64', na_value=-1)
    np.testing.assert_array_equal(artist_pair, actual, err_msg='user_artist_id')


def choose_sample(split, row_cap, seed):
    """Выбрать целые canonical-группы и восстановить запросы в порядке targets."""
    targets_path, metadata_path = _source_paths(split)
    with np.load(targets_path) as source:
        groups = source['group_id']
        positions = _whole_group_positions(groups, row_cap, seed)
        targets = {'label': source['label'][positions], 'group_id': groups[positions],
                   'log_id': source['log_id'][positions]}
    targets['day'] = targets['group_id'] % 1000
    if len(np.unique(targets['log_id'])) != len(positions):
        raise ValueError('В canonical targets повторяются log_id')
    queries = _read_selected_metadata(metadata_path, targets['log_id'])
    _validate_queries(queries, targets)
    array_records = {name: {'dtype': value.dtype.str, 'rows': len(value),
                            'sha256': _array_hash(value)} for name, value in targets.items()}
    order_hash = hashlib.sha256()
    for name in TARGET_FIELDS:
        order_hash.update(name.encode())
        order_hash.update(array_records[name]['sha256'].encode())
    provenance = {'split': split, 'row_cap': int(row_cap), 'seed': int(seed),
                  'sampling': 'random whole canonical groups; skip groups exceeding remaining row budget',
                  'rows': len(queries), 'groups': len(np.unique(targets['group_id'])),
                  'test_used': False, 'row_order_sha256': order_hash.hexdigest(),
                  'target_arrays': array_records,
                  'sources': {'targets': _file_record(targets_path),
                              'metadata': _file_record(metadata_path)}}
    return queries, targets, provenance


def _empty_history(key):
    """Вернуть пустую историю с теми же числовыми типами."""
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in
                         [('log_id', 'int64'), (key, 'int64'), ('end_ts', 'int64'),
                          ('label', 'int8'), ('track_length_seconds', 'uint16'),
                          ('is_organic', 'uint8')]})


def _branch_keys(table, key):
    """Закодировать ключ ветви, не смешивая неизвестного артиста с реальным id 0."""
    if key in ('uid', 'item_id'):
        return table[key].to_numpy().astype('int64', copy=False)
    uid = table['uid'].to_numpy().astype('int64', copy=False)
    other_name = 'item_id' if key == 'user_item_id' else 'artist_id'
    other = table[other_name].to_numpy().astype('int64', copy=False)
    values = (uid << 32) | np.maximum(other, 0)
    if key == 'user_artist_id':
        values[other < 0] = -1
    return values


def load_branch_history(queries, key, *, batch_size=500_000):
    """Прочитать полную прошлую историю ключей ветви, включая всех пользователей Item.

    Временной предел здесь общий: timestamp < max(query.time). Индивидуальную
    строгую границу и окна применяет движок. Исходный log_id — номер listen-строки.
    """
    if key not in KEY_COLUMNS:
        raise ValueError(f'Неизвестная ветвь: {key}')
    if batch_size <= 0:
        raise ValueError('batch_size должен быть положительным')
    if queries.empty:
        return _empty_history(key)
    known = np.unique(queries[key].dropna().to_numpy(dtype='int64'))
    known = known[known >= 0]
    if not len(known):
        return _empty_history(key)
    times = queries['time'].to_numpy(dtype='float64')
    if not np.isfinite(times).all():
        raise ValueError('Время запросов должно быть конечным')
    limit = times.max()
    value_set = pa.array(known)
    columns = list(dict.fromkeys(KEY_COLUMNS[key] +
                   ['timestamp', 'label', 'track_length_seconds', 'is_organic']))
    parts, offset = [], 0
    for batch in pq.ParquetFile(ROOT / 'data/listens.parquet').iter_batches(
            batch_size=batch_size, columns=columns):
        table = pa.Table.from_batches([batch])
        keys = _branch_keys(table, key)
        mask = pc.is_in(pa.array(keys), value_set=value_set).to_numpy(zero_copy_only=False)
        mask = mask & (keys >= 0) & (table['timestamp'].to_numpy() < limit)
        if mask.any():
            values = {'log_id': offset + np.flatnonzero(mask), key: keys[mask],
                      'end_ts': table['timestamp'].to_numpy()[mask]}
            for name in ['label', 'track_length_seconds', 'is_organic']:
                values[name] = table[name].to_numpy()[mask]
            parts.append(pd.DataFrame(values))
        offset += len(batch)
    return pd.concat(parts, ignore_index=True) if parts else _empty_history(key)
