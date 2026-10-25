"""Ручные признаки прежнего baseline для новых временных окон.

Метки и GAP задает вызывающий код. Здесь исходная история не обрезается по split:
для каждой целевой строки используются только события с timestamp < start_ts.
"""

from pathlib import Path
import ctypes
import gc
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
DAY = 86400
SHIFT = 2 ** 26
UI_WINDOWS = (7, 14, 30, 90)
ITEM_WINDOWS = (1, 7, 30, 90)
UA_WINDOWS = (7, 14, 30, 90)
HALF_LIVES = (1, 7, 30)
CHUNK_USERS = 1000
METADATA = [
    'log_id', 'uid', 'item_id', 'artist_id', 'start_ts', 'timestamp', 'label',
    'group_id', 'is_organic', 'track_length_seconds', 'split',
]


def _read_columns(path, columns):
    """Прочитать только необходимые столбцы, не копируя блоки pandas целиком."""
    table = pq.read_table(path, columns=columns)
    return table.to_pandas(split_blocks=True, self_destruct=True)


def _listen_arrays():
    """Загрузить полную историю listen и поля для оконных статистик."""
    columns = ['uid', 'item_id', 'artist_id', 'timestamp', 'played_ratio_pct']
    data = _read_columns(PROJECT / 'data/listens.parquet', columns)
    values = {name: data[name].to_numpy() for name in columns[:-1]}
    values['skip'] = (data['played_ratio_pct'] < 35).to_numpy('int8')
    values['pr'] = data['played_ratio_pct'].clip(upper=100).to_numpy('float32')
    if not data['uid'].is_monotonic_increasing:
        raise ValueError('listens.parquet должен быть упорядочен по uid')
    return values


def _like_arrays():
    """Загрузить like/unlike с собственным временем события."""
    columns = ['uid', 'item_id', 'timestamp', 'event_type']
    data = _read_columns(PROJECT / 'data/events.parquet', columns)
    data = data.loc[data['event_type'].isin(['like', 'unlike'])]
    values = {name: data[name].to_numpy() for name in columns[:-1]}
    values['is_like'] = (data['event_type'] == 'like').to_numpy('int8')
    if not data['uid'].is_monotonic_increasing:
        raise ValueError('events.parquet должен быть упорядочен по uid')
    return values


def _pair_key(first, second):
    """Закодировать пару id тем же множителем, что в dataset-analysis."""
    return first.astype('int64') * 10 ** 7 + second.astype('int64')


def _dense_keys(history_keys, query_keys):
    """Присвоить истории и запросам общие компактные неотрицательные коды."""
    codes, _ = pd.factorize(np.concatenate([history_keys, query_keys]))
    size = len(history_keys)
    return codes[:size].astype('int64'), codes[size:].astype('int64')


def _add_prefix_sums(index, order, values):
    """Добавить суммы с начальным нулем для вычитания границ окна."""
    for name, value in (values or {}).items():
        sums = np.cumsum(value[order], dtype='float64')
        index['cs_' + name] = np.concatenate([[0], sums])


def _add_decay_sums(index, key, order, half_lives, reference_time):
    """Накопить экспоненциальные веса отдельно внутри каждого ключа."""
    for half_life in half_lives:
        weights = np.exp2((index['ts'] - reference_time) / (half_life * DAY))
        sums = pd.Series(weights).groupby(key[order]).cumsum().to_numpy()
        index[f'dec_{half_life}'] = sums


def _build_index(key, ts, values=None, half_lives=(), keep=None, reference_time=0):
    """Подготовить отсортированный индекс событий, суммы и последние значения."""
    code = key * SHIFT + ts
    order = np.argsort(code, kind='stable')
    index = {'code': code[order], 'ts': ts[order]}
    del code
    _add_prefix_sums(index, order, values)
    for name, value in (keep or {}).items():
        index[name] = value[order]
    _add_decay_sums(index, key, order, half_lives, reference_time)
    return index


def _window_aggregates(index, base, ts, upper, windows, sums):
    """Посчитать счетчики и суммы на полуинтервале [ts-window, ts)."""
    values = {}
    for window in windows:
        lower_time = np.maximum(ts - window * DAY, 0)
        lower = np.searchsorted(index['code'], base + lower_time, 'left')
        values[f'cnt_{window}'] = upper - lower
        for name in sums:
            values[f'{name}_{window}'] = index['cs_' + name][upper] - index['cs_' + name][lower]
    return values


def _past_values(index, name, previous, has_history, missing=np.nan):
    """Получить последнее доступное значение, включая случай пустой истории."""
    if len(index['code']) == 0:
        return np.full(len(previous), missing)
    return np.where(has_history, index[name][previous], missing)


def _add_last_values(values, index, base, upper, last):
    """Дополнить результат полями последнего события и вернуть маску истории."""
    has_history = upper > np.searchsorted(index['code'], base, 'left')
    previous = np.maximum(upper - 1, 0)
    for name in last:
        values['last_' + name] = _past_values(index, name, previous, has_history)
    return has_history, previous


def _add_query_decays(values, index, ts, previous, has_history, half_lives, reference_time):
    """Перенести накопленные веса на время запроса; без истории вернуть ноль."""
    for half_life in half_lives:
        scale = np.exp2((reference_time - ts) / (half_life * DAY))
        weights = _past_values(index, f'dec_{half_life}', previous, has_history, 0)
        values[f'dec_{half_life}'] = weights * scale


def _query(index, key, ts, windows=(), sums=(), half_lives=(), last=(), reference_time=0):
    """Считать признаки строго до ts: события с timestamp == ts исключены."""
    base = key * SHIFT
    upper = np.searchsorted(index['code'], base + ts, 'left')
    values = _window_aggregates(index, base, ts, upper, windows, sums)
    if half_lives or last:
        has_history, previous = _add_last_values(values, index, base, upper, last)
        _add_query_decays(values, index, ts, previous, has_history, half_lives, reference_time)
    return values


def _rate(numerator, denominator):
    """Разделить сумму на счетчик; без наблюдений вернуть NaN."""
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1), np.nan)


def _float32_features(values):
    """Привести исторические признаки к прежнему компактному типу float32."""
    return {name: np.asarray(value, dtype='float32') for name, value in values.items()}


def _count_skip_features(stats, prefix, windows, skip_rate=False):
    """Собрать счетчики прослушиваний и количества либо доли пропусков."""
    values = {}
    for window in windows:
        count, skip = stats[f'cnt_{window}'], stats[f'skip_{window}']
        values[f'{prefix}_lst_amt_{window}'] = count
        suffix = 'rate' if skip_rate else 'cnt'
        values[f'{prefix}_skip_{suffix}_{window}'] = _rate(skip, count) if skip_rate else skip
    return values


def _decay_features(stats, prefix):
    """Назвать суммы затухания в прежнем порядке 1, 7, 30 дней."""
    return {f'{prefix}_lst_decay_{half}': stats[f'dec_{half}'] for half in HALF_LIVES}


def _item_window_features(listen_stats, like_stats, window):
    """Собрать четыре Item-признака одного окна в фиксированном порядке."""
    listen_count = listen_stats[f'cnt_{window}']
    return {
        f'item_lst_amt_{window}': listen_count,
        f'item_skip_rate_{window}': _rate(listen_stats[f'skip_{window}'], listen_count),
        f'item_like_rate_{window}': _rate(like_stats[f'cnt_{window}'], listen_count),
        f'item_avg_played_ratio_{window}': _rate(listen_stats[f'pr_{window}'], listen_count),
    }


def _user_slice(uids, first_uid, last_uid):
    """Найти сплошной срез отсортированной истории для порции пользователей."""
    return slice(np.searchsorted(uids, first_uid, 'left'), np.searchsorted(uids, last_uid, 'right'))


class _ManualHistory:
    """Держать общие Item-индексы и считать User–Item/User–Artist по порциям."""

    def __init__(self):
        """Подготовить общую историю и глобальные Item-индексы."""
        self.listens = _listen_arrays()
        self.likes = _like_arrays()
        self.reference_time = int(self.listens['timestamp'].max())
        self._build_item_indices()

    def _build_item_indices(self):
        """Построить глобальную популярность по всем пользователям один раз."""
        history = self.listens
        values = {'skip': history['skip'], 'pr': history['pr']}
        self.item_index = _build_index(history['item_id'].astype('int64'), history['timestamp'], values)
        history, liked = self.likes, self.likes['is_like'] == 1
        self.item_like_index = _build_index(history['item_id'][liked].astype('int64'), history['timestamp'][liked])

    def _pair_keys(self, queries, history_slice, attribute):
        """Подготовить общие ключи истории и запросов пары user–attribute."""
        history = self.listens
        old = _pair_key(history['uid'][history_slice], history[attribute][history_slice])
        new = _pair_key(queries['uid'].to_numpy(), queries[attribute].to_numpy())
        return _dense_keys(old, new)

    def _pair_listen_statistics(self, queries, history_slice, attribute, windows, last=()):
        """Посчитать окна и затухания истории пары user–item либо user–artist."""
        history = self.listens
        history_keys, query_keys = self._pair_keys(queries, history_slice, attribute)
        values = {'skip': history['skip'][history_slice]}
        index = _build_index(
            history_keys, history['timestamp'][history_slice], values, HALF_LIVES,
            reference_time=self.reference_time,
        )
        query_times = queries['start_ts'].to_numpy('int64')
        return _query(index, query_keys, query_times, windows, ['skip'], HALF_LIVES,
                      last, self.reference_time)

    def user_item_features(self, queries, history_slice):
        """Собрать счетчики, затухания и давность последнего listen пары user–item."""
        stats = self._pair_listen_statistics(
            queries, history_slice, 'item_id', UI_WINDOWS, last=['ts'],
        )
        result = _count_skip_features(stats, 'ui', UI_WINDOWS)
        result.update(_decay_features(stats, 'ui'))
        query_times = queries['start_ts'].to_numpy('int64')
        result['ui_time_since_last_lst'] = query_times - stats['last_ts']
        return _float32_features(result)

    def user_item_like_features(self, queries, history_slice):
        """Восстановить состояние like/unlike непосредственно до start_ts."""
        history = self.likes
        old = _pair_key(history['uid'][history_slice], history['item_id'][history_slice])
        new = _pair_key(queries['uid'].to_numpy(), queries['item_id'].to_numpy())
        old, new = _dense_keys(old, new)
        index = _build_index(old, history['timestamp'][history_slice], keep={'is_like': history['is_like'][history_slice]})
        stats = _query(index, new, queries['start_ts'].to_numpy('int64'), last=['is_like'])
        return _float32_features({'ui_liked_now': np.nan_to_num(stats['last_is_like'])})

    def item_features(self, queries):
        """Собрать популярность, доли пропусков/лайков и среднюю долю listen."""
        item_keys = queries['item_id'].to_numpy('int64')
        query_times = queries['start_ts'].to_numpy('int64')
        listen_stats = _query(self.item_index, item_keys, query_times, ITEM_WINDOWS,
                              ['skip', 'pr'])
        like_stats = _query(self.item_like_index, item_keys, query_times, ITEM_WINDOWS)
        values = {}
        for window in ITEM_WINDOWS:
            values.update(_item_window_features(listen_stats, like_stats, window))
        return _float32_features(values)

    def user_artist_features(self, queries, history_slice):
        """Собрать историю user–artist; неизвестный artist_id=-1 дает NaN."""
        stats = self._pair_listen_statistics(
            queries, history_slice, 'artist_id', UA_WINDOWS,
        )
        result = _count_skip_features(stats, 'ua', UA_WINDOWS, skip_rate=True)
        result.update(_decay_features(stats, 'ua'))
        missing_artist = queries['artist_id'].to_numpy() == -1
        return _float32_features({
            name: np.where(missing_artist, np.nan, value)
            for name, value in result.items()
        })

    def chunk_features(self, queries):
        """Объединить четыре блока, сохранив metadata и порядок целевых строк."""
        first_uid, last_uid = queries['uid'].iloc[0], queries['uid'].iloc[-1]
        listens = _user_slice(self.listens['uid'], first_uid, last_uid)
        likes = _user_slice(self.likes['uid'], first_uid, last_uid)
        values = self.user_item_features(queries, listens)
        values.update(self.user_item_like_features(queries, likes))
        values.update(self.item_features(queries))
        values.update(self.user_artist_features(queries, listens))
        metadata = queries[METADATA].reset_index(drop=True)
        return pd.concat([metadata, pd.DataFrame(values)], axis=1)


def _validate_target_order(targets):
    """Проверить возрастание uid, а внутри пользователя — строгое возрастание log_id."""
    user_ids = targets['uid'].to_numpy()
    log_ids = targets['log_id'].to_numpy()
    same_user = user_ids[1:] == user_ids[:-1]
    disorder = (user_ids[1:] < user_ids[:-1]) | (same_user & (log_ids[1:] <= log_ids[:-1]))
    if disorder.any():
        raise ValueError('targets должны быть отсортированы по uid, log_id')


def _validate_targets(targets):
    """Проверить схему, уникальность, порядок событий и границы времени."""
    missing = set(METADATA).difference(targets.columns)
    if missing:
        raise ValueError(f'В targets отсутствуют столбцы: {sorted(missing)}')
    if len(targets) == 0 or not targets['log_id'].is_unique:
        raise ValueError('Нужны непустые targets с уникальными log_id')
    _validate_target_order(targets)
    if (targets['start_ts'].to_numpy() > targets['timestamp'].to_numpy()).any():
        raise ValueError('start_ts не может быть позже окончания listen')


def _target_chunks(targets):
    """Выдать порции до 1000 пользователей без группировки всей матрицы признаков."""
    uids = targets['uid'].to_numpy()
    users = targets['uid'].drop_duplicates().to_numpy()
    for first in range(0, len(users), CHUNK_USERS):
        last = min(first + CHUNK_USERS, len(users)) - 1
        yield targets.iloc[_user_slice(uids, users[first], users[last])]


def _write_tables(history, targets, temporary_path):
    """Записывать признаки порциями, удерживая только одну порцию в памяти."""
    chunks = iter(_target_chunks(targets))
    first = pa.Table.from_pandas(history.chunk_features(next(chunks)), preserve_index=False)
    rows, chunk_count = len(first), 1
    with pq.ParquetWriter(temporary_path, first.schema) as writer:
        writer.write_table(first)
        del first
        for chunk in chunks:
            table = pa.Table.from_pandas(history.chunk_features(chunk), preserve_index=False)
            writer.write_table(table)
            rows, chunk_count = rows + len(table), chunk_count + 1
            del table
    return rows, chunk_count


def _prepare_output(output_path):
    """Разрешить отдельный Parquet-артефакт, защитив исходные файлы данных."""
    path = Path(output_path).resolve()
    protected = {PROJECT / 'data' / f'{name}.parquet' for name in ('listens', 'events', 'samples', 'features')}
    if path in protected or path.suffix != '.parquet':
        raise ValueError('Нужен отдельный новый путь *.parquet, исходные файлы защищены')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_output(history, targets, output_path):
    """Заменить только целевой артефакт после успешной полной записи."""
    temporary_path = output_path.with_name(output_path.name + f'.{os.getpid()}.tmp')
    try:
        rows, chunks = _write_tables(history, targets, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return rows, chunks


def _generation_report(history, output_path, rows, chunks, started):
    """Вернуть размеры и параметры генерации для отчета вызывающего эксперимента."""
    columns = pq.read_schema(output_path).names
    feature_names = ['is_organic', 'track_length_seconds'] + [name for name in columns if name not in METADATA]
    return {
        'output_path': str(output_path), 'rows': rows, 'columns': columns,
        'feature_names': feature_names, 'feature_count': len(feature_names),
        'history_listen_rows': len(history.listens['uid']),
        'history_like_unlike_rows': len(history.likes['uid']),
        'history_cutoff': 'event.timestamp < target.start_ts',
        'history_paths': [str(PROJECT / 'data' / f'{name}.parquet') for name in ('listens', 'events')],
        'chunk_users': CHUNK_USERS, 'chunks': chunks, 'bytes': output_path.stat().st_size,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
    }


def _release_memory():
    """Освободить нативные буферы между подготовкой признаков и следующим этапом."""
    gc.collect()
    pa.default_memory_pool().release_unused()
    if sys.platform.startswith('linux'):
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            trim(0)


def generate_manual_features(targets: pd.DataFrame, output_path: Path) -> dict:
    """Построить те же 42 признака baseline; targets уже содержат split и GAP.

    log_id должен быть уникальным, строки — упорядоченными по uid/log_id.
    История берется из полного listens/events; исходные артефакты не меняются.
    """
    started = time.perf_counter()
    _validate_targets(targets)
    output_path = _prepare_output(output_path)
    history = _ManualHistory()
    rows, chunks = _write_output(history, targets, output_path)
    report = _generation_report(history, output_path, rows, chunks, started)
    del history
    _release_memory()
    return report
