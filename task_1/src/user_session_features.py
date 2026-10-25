"""Семь ручных признаков пользователя для контроля C2.

История доступна только при timestamp < start_ts запроса. Все времена заданы
в секундах; сутки — полуинтервал [start_ts - 86400, start_ts).
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DAY = 86400
HISTORY_COLUMNS = ('uid', 'timestamp', 'played_ratio_pct')
SESSION_FEATURES = (
    'user_prev_played_ratio',
    'user_prev_skip',
    'user_skip_rate_last5',
    'user_skip_rate_last20',
    'user_skip_rate_1d',
    'user_lst_cnt_1d',
    'user_time_since_last',
)


def _numeric_columns(frame, names):
    """Проверить нужные поля, не копируя остальные столбцы таблицы."""
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f'Отсутствуют столбцы: {sorted(missing)}')
    values = {}
    for name in names:
        array = frame[name].to_numpy(copy=False)
        if len(array) and (array.dtype.kind not in 'iuf' or not np.isfinite(array).all()):
            raise ValueError(f'{name}: нужны конечные числовые значения')
        values[name] = array
    return values


def _empty_features(rows):
    """Без истории счетчик равен нулю; доли, последние значения и давность — NaN."""
    values = np.full((rows, len(SESSION_FEATURES)), np.nan, dtype='float32', order='F')
    values[:, 5] = 0
    return values


def _skip_rate(prefix_skip, lower, upper):
    """Вычесть накопленные суммы и поделить на число доступных событий."""
    count = upper - lower
    result = np.full(len(count), np.nan, dtype='float32')
    np.divide(prefix_skip[upper] - prefix_skip[lower], count,
              out=result, where=count > 0)
    return result


def _user_features(end_times, played_ratios, query_times):
    """Посчитать все запросы пользователя одним поиском границ и prefix-суммами."""
    result = _empty_features(len(query_times))
    if not len(end_times):
        return result
    if np.any(end_times[1:] < end_times[:-1]):
        order = np.argsort(end_times, kind='stable')
        end_times, played_ratios = end_times[order], played_ratios[order]
    skip = played_ratios < 35
    prefix_skip = np.r_[0, np.cumsum(skip, dtype='int64')]
    # side='left' исключает все события с временем, равным времени запроса.
    upper = np.searchsorted(end_times, query_times, side='left')
    present = upper > 0
    previous = upper[present] - 1
    result[present, 0] = np.minimum(played_ratios[previous], 100)
    result[present, 1] = skip[previous]
    result[present, 6] = query_times[present] - end_times[previous]
    for column, last_n in [(2, 5), (3, 20)]:
        result[:, column] = _skip_rate(prefix_skip, np.maximum(upper - last_n, 0), upper)
    lower = np.searchsorted(end_times, query_times - DAY, side='left')
    result[:, 4] = _skip_rate(prefix_skip, lower, upper)
    result[:, 5] = upper - lower
    return result


def calculate_session_features(queries: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Вернуть C2-признаки в исходном порядке и с исходным индексом queries.

    queries содержит uid и start_ts; history — uid, timestamp, played_ratio_pct.
    History должна идти блоками uid по возрастанию. Внутри пользователя времена
    сортируются стабильно: при равных timestamp последним остается более поздний
    ряд исходного лога. Текущие и будущие события исключаются строгой границей.
    """
    query = _numeric_columns(queries, ('uid', 'start_ts'))
    old = _numeric_columns(history, HISTORY_COLUMNS)
    user_ids = old['uid']
    if np.any(user_ids[1:] < user_ids[:-1]):
        raise ValueError('history должна быть отсортирована по uid')
    if np.any(old['played_ratio_pct'] < 0):
        raise ValueError('played_ratio_pct не может быть отрицательным')
    values = _empty_features(len(queries))
    if len(queries) and len(history):
        order = np.argsort(query['uid'], kind='stable')
        ordered_users = query['uid'][order]
        starts = np.flatnonzero(np.r_[True, ordered_users[1:] != ordered_users[:-1]])
        for left, right in zip(starts, np.r_[starts[1:], len(order)]):
            positions = order[left:right]
            uid = ordered_users[left]
            hleft, hright = np.searchsorted(user_ids, uid, side='left'), np.searchsorted(user_ids, uid, side='right')
            values[positions] = _user_features(
                old['timestamp'][hleft:hright], old['played_ratio_pct'][hleft:hright],
                query['start_ts'][positions].astype('float64', copy=False),
            )
    return pd.DataFrame(values, index=queries.index.copy(), columns=SESSION_FEATURES)


def _read_session_history(path, batch_size):
    """Загрузить только три нужных поля; остальные столбцы журнала не читаются."""
    batches = pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(HISTORY_COLUMNS))
    table = pa.Table.from_batches(batches)
    return table.to_pandas(split_blocks=True, self_destruct=True)


def write_session_feature_matrix(queries: pd.DataFrame, history_path, output_path,
                                 batch_size=500_000) -> dict:
    """Сохранить семь float32-столбцов F-order в новый npy без изменения источников.

    Runner передает уже выбранные canonical queries. Здесь нет собственного
    split, отбора групп или чтения test. Готовый файл никогда не перезаписывается.
    """
    if batch_size <= 0 or int(batch_size) != batch_size:
        raise ValueError('batch_size должен быть положительным целым')
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + '.incomplete')
    if output_path.suffix != '.npy':
        raise ValueError('Нужен новый файл с расширением .npy')
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f'Выходной файл уже существует: {output_path}')
    started = time.perf_counter()
    history = _read_session_history(history_path, int(batch_size))
    features = calculate_session_features(queries, history)
    history_rows = len(history)
    del history
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        matrix = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32',
            shape=(len(queries), len(SESSION_FEATURES)), fortran_order=True)
        for column, name in enumerate(SESSION_FEATURES):
            matrix[:, column] = features[name].to_numpy(copy=False)
        matrix.flush()
        del matrix
        if output_path.exists():
            raise FileExistsError(f'Выходной файл уже существует: {output_path}')
        temporary.rename(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'matrix_path': str(output_path.resolve()), 'rows': len(queries),
            'features': list(SESSION_FEATURES), 'feature_count': len(SESSION_FEATURES),
            'dtype': 'float32', 'layout': 'F-order', 'history_rows': history_rows,
            'history_cutoff': 'history.timestamp < query.start_ts',
            'day_window': '[start_ts - 86400, start_ts)',
            'elapsed_seconds': time.perf_counter() - started}
