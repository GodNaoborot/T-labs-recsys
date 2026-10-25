"""DFS с окнами 1–30 дней и свежестью; расчет без повторной фильтрации истории.

Синтез: все 103 формулы прежней грамматики на полной истории и в каждом окне.
Произведения соединяют агрегаты одного окна; произведений разных окон нет.
Свежесть считается по полной доступной истории. Отбор по качеству идет отдельно.
"""
import numpy as np
import pandas as pd
import featuretools as ft
from featuretools.primitives import TimeSinceFirst, TimeSinceLast
from numba import njit
from woodwork.logical_types import Datetime, Double, Integer

from cumulative_features import VALUE_COLUMNS, event_values

DAY = 86400.0


def _empty_entityset(parent_name, key):
    """Создать только схему: время Datetime нужно оконным примитивам Featuretools."""
    parent = pd.DataFrame({key: pd.Series(dtype='int64')})
    child = pd.DataFrame({column: pd.Series(dtype='float64')
                          for column in ['log_id', key] + VALUE_COLUMNS})
    child['end_ts'] = pd.Series(dtype='datetime64[ns]')
    entityset = ft.EntitySet(id=parent_name)
    entityset.add_dataframe(dataframe_name=parent_name, dataframe=parent, index=key,
                            logical_types={key: Integer})
    entityset.add_dataframe(
        dataframe_name='Interactions', dataframe=child, index='log_id', time_index='end_ts',
        logical_types={'log_id': Integer, key: Integer, 'end_ts': Datetime,
                       **{column: Double for column in VALUE_COLUMNS}})
    entityset.add_relationship(parent_name, key, 'Interactions', key)
    return entityset


def _with_window(feature, days, parent_name):
    """Прикрепить одно окно ко всем агрегациям существующей формулы DFS."""
    if isinstance(feature, ft.AggregationFeature):
        return ft.AggregationFeature(feature.base_features, parent_dataframe_name=parent_name,
                                     primitive=feature.primitive, use_previous=f'{days} days')
    if isinstance(feature, ft.TransformFeature) and feature.primitive.name == 'multiply_numeric':
        return ft.TransformFeature([_with_window(base, days, parent_name)
                                    for base in feature.base_features], primitive=feature.primitive)
    raise NotImplementedError(feature.get_name())


def build_temporal_definitions(parent_name, key, windows=range(1, 31)):
    """Получить весь набор кандидатов без лимита 32; окна задаются целыми днями."""
    windows = list(windows)
    if any(not isinstance(day, (int, np.integer)) or day <= 0 for day in windows):
        raise ValueError('Окна должны быть положительными целыми числами дней')
    if len(set(windows)) != len(windows):
        raise ValueError('Окна не должны повторяться')
    entityset = _empty_entityset(parent_name, key)
    base = ft.dfs(
        entityset=entityset, target_dataframe_name=parent_name, features_only=True,
        agg_primitives=['count', 'mean', 'std', 'min', 'max'],
        trans_primitives=['multiply_numeric'], max_depth=2, max_features=-1,
        ignore_columns={'Interactions': ['end_ts']})
    definitions = list(base)
    for days in windows:
        definitions.extend(_with_window(feature, days, parent_name) for feature in base)
    definitions.extend(ft.dfs(
        entityset=entityset, target_dataframe_name=parent_name, features_only=True,
        agg_primitives=[TimeSinceLast(unit='days'), TimeSinceFirst(unit='days')],
        trans_primitives=[], max_depth=1, max_features=-1,
        ignore_columns={'Interactions': VALUE_COLUMNS}))
    names = [feature.get_name() for feature in definitions]
    if len(set(names)) != len(names):
        raise ValueError('Повторяющиеся формулы')
    return definitions


def _feature_aggregations(feature, parent_name):
    """Извлечь листья-агрегаты и отклонить неподдержанную грамматику."""
    if isinstance(feature, ft.AggregationFeature):
        if (feature.dataframe_name != parent_name or feature.child_dataframe_name != 'Interactions'
                or feature.where is not None):
            raise NotImplementedError(feature.get_name())
        supported = {'count', 'mean', 'std', 'min', 'max', 'time_since_first', 'time_since_last'}
        if feature.primitive.name not in supported:
            raise NotImplementedError(feature.get_name())
        return [feature]
    if (isinstance(feature, ft.TransformFeature) and feature.dataframe_name == parent_name
            and feature.primitive.name == 'multiply_numeric'):
        return [aggregate for base in feature.base_features
                for aggregate in _feature_aggregations(base, parent_name)]
    raise NotImplementedError(feature.get_name())


def _window_seconds(feature):
    """Перевести native use_previous Featuretools в секунды числового журнала."""
    previous = feature.use_previous
    return None if previous is None else float(previous.delta_obj.total_seconds())


def _group_definitions(definitions, parent_name):
    """Сгруппировать формулы по окну, чтобы не хранить всю широкую матрицу дважды."""
    groups = {}
    for position, feature in enumerate(definitions):
        aggregates = _feature_aggregations(feature, parent_name)
        windows = {_window_seconds(aggregate) for aggregate in aggregates}
        if len(windows) != 1:
            raise NotImplementedError('Произведения разных окон пока не поддержаны')
        window = windows.pop()
        groups.setdefault(window, []).append((position, feature, aggregates))
    return groups


@njit
def _lower_bound(values, start, end, threshold):
    """Первый индекс со значением не меньше threshold внутри одной сущности."""
    while start < end:
        middle = (start + end) // 2
        if values[middle] < threshold:
            start = middle + 1
        else:
            end = middle
    return start


@njit
def _query_bounds(end_times, group_starts, group_ends, query_times, window):
    """Границы [t-window, t): одинаковые timestamp верхней границы исключаются все."""
    left = np.empty(len(query_times), dtype=np.int64)
    right = np.empty(len(query_times), dtype=np.int64)
    for row in range(len(query_times)):
        start, end = group_starts[row], group_ends[row]
        right[row] = _lower_bound(end_times, start, end, query_times[row])
        left[row] = start if window < 0 else _lower_bound(
            end_times, start, right[row], query_times[row] - window)
    return left, right


@njit
def _rolling_column(values, group_starts, left_bounds, right_bounds, queue_size):
    """Считать mean/std/min/max: добавлять новые события и удалять вышедшие из окна.

    Среднее и M2 обновляются формулами Уэлфорда с удалением. Монотонные очереди
    хранят кандидатов на минимум/максимум и требуют память одной сущности.
    """
    result = np.full((len(left_bounds), 4), np.nan)
    minimum_queue = np.empty(queue_size, dtype=np.int64)
    maximum_queue = np.empty(queue_size, dtype=np.int64)
    count, mean, m2 = 0, 0.0, 0.0
    left, right, min_head, min_tail, max_head, max_tail = 0, 0, 0, 0, 0, 0
    for row in range(len(left_bounds)):
        if row == 0 or group_starts[row] != group_starts[row - 1]:
            count, mean, m2 = 0, 0.0, 0.0
            left = right = left_bounds[row]
            min_head = min_tail = max_head = max_tail = 0
        new_left, new_right = left_bounds[row], right_bounds[row]
        while left < min(new_left, right):
            value = values[left]
            if not np.isnan(value):
                if count == 1:
                    count, mean, m2 = 0, 0.0, 0.0
                else:
                    new_mean = mean - (value - mean) / (count - 1)
                    m2 = max(m2 - (value - mean) * (value - new_mean), 0.0)
                    mean = new_mean
                    count -= 1
            left += 1
        left = new_left
        right = max(right, new_left)
        while right < new_right:
            value = values[right]
            if not np.isnan(value):
                count += 1
                delta = value - mean
                mean += delta / count
                m2 += delta * (value - mean)
                while min_tail > min_head and values[minimum_queue[min_tail - 1]] >= value:
                    min_tail -= 1
                minimum_queue[min_tail] = right
                min_tail += 1
                while max_tail > max_head and values[maximum_queue[max_tail - 1]] <= value:
                    max_tail -= 1
                maximum_queue[max_tail] = right
                max_tail += 1
            right += 1
        while min_head < min_tail and minimum_queue[min_head] < new_left:
            min_head += 1
        while max_head < max_tail and maximum_queue[max_head] < new_left:
            max_head += 1
        if count > 0:
            minimum = values[minimum_queue[min_head]]
            maximum = values[maximum_queue[max_head]]
            # После удаления событий M2 может сохранять остаток округления.
            # Для постоянного окна его точное значение и std равны нулю.
            if minimum == maximum:
                mean, m2 = minimum, 0.0
            result[row, 0] = mean
            result[row, 2] = minimum
            result[row, 3] = maximum
        if count > 1:
            result[row, 1] = np.sqrt(max(m2, 0.0) / (count - 1))
    return result


def _ordered_inputs(history, queries, key):
    """Оставить нужные сущности и отсортировать события/запросы один раз."""
    valid = queries[key].notna().to_numpy()
    requests = queries.loc[valid, [key, 'time']]
    child = history.loc[history[key].isin(requests[key]), [key, 'end_ts'] + VALUE_COLUMNS]
    history_keys = child[key].to_numpy(dtype='int64')
    end_times = child['end_ts'].to_numpy(dtype='float64')
    query_keys = requests[key].to_numpy(dtype='int64')
    query_times = requests['time'].to_numpy(dtype='float64')
    if not np.isfinite(end_times).all() or not np.isfinite(query_times).all():
        raise ValueError('Время должно быть конечным числом')
    history_order = np.lexsort((end_times, history_keys))
    query_order = np.lexsort((query_times, query_keys))
    history_keys = history_keys[history_order]
    query_keys = query_keys[query_order]
    starts = np.searchsorted(history_keys, query_keys, side='left')
    ends = np.searchsorted(history_keys, query_keys, side='right')
    rows = np.flatnonzero(valid)[query_order]
    return child, history_order, end_times[history_order], query_times[query_order], starts, ends, rows


def _aggregation_vectors(aggregates, child, order, end_times, query_times, starts, ends, left, right):
    """Вычислить только уникальные агрегаты текущего окна; базовые выражения по одному."""
    calculated = {}
    bases = {}
    for feature in aggregates.values():
        name = feature.primitive.name
        if name == 'count':
            if feature.base_features[0].column_name != 'log_id':
                raise NotImplementedError(feature.get_name())
            calculated[feature.unique_name()] = right - left
        elif name.startswith('time_since_'):
            if feature.use_previous is not None or feature.base_features[0].column_name != 'end_ts':
                raise NotImplementedError(feature.get_name())
            values = np.full(len(query_times), np.nan)
            found = right > starts
            positions = right[found] - 1 if name == 'time_since_last' else starts[found]
            values[found] = (query_times[found] - end_times[positions]) / DAY
            calculated[feature.unique_name()] = values
        else:
            base = feature.base_features[0]
            bases.setdefault(base.unique_name(), (base, []))[1].append(feature)
    queue_size = max(int(np.max(ends - starts, initial=0)), 1)
    positions = {'mean': 0, 'std': 1, 'min': 2, 'max': 3}
    for base, features in bases.values():
        values = event_values(base, child, order)
        if np.isinf(values).any():
            raise ValueError('Поддержаны конечные значения и NaN')
        statistics = _rolling_column(values, starts, left, right, queue_size)
        for feature in features:
            calculated[feature.unique_name()] = statistics[:, positions[feature.primitive.name]]
    return calculated


def _evaluate(feature, calculated):
    """Прочитать готовый агрегат или вычислить произведение его подформул."""
    if isinstance(feature, ft.AggregationFeature):
        return calculated[feature.unique_name()]
    left, right = feature.base_features
    return _evaluate(left, calculated) * _evaluate(right, calculated)


def calculate_temporal(history, queries, parent_name, key, features, output=None):
    """Заполнить float32-матрицу в порядке queries; output может быть memmap.

    Неизвестный ключ дает нулевой COUNT и NaN остальных агрегатов; отсутствующий
    ключ остается NaN во всех столбцах. История события доступна только end_ts<t.
    """
    if not queries['log_id'].is_unique:
        raise ValueError('log_id запросов должны быть уникальными')
    features = list(features)
    shape = (len(queries), len(features))
    if output is None:
        output = np.empty(shape, dtype='float32')
    if output.shape != shape or output.dtype != np.dtype('float32'):
        raise ValueError('output должен иметь shape=(queries, features) и dtype float32')
    output[:] = np.nan
    groups = _group_definitions(features, parent_name)
    child, order, times, query_times, starts, ends, rows = _ordered_inputs(history, queries, key)
    if not len(rows):
        return output
    for window, grouped in groups.items():
        left, right = _query_bounds(times, starts, ends, query_times, -1.0 if window is None else window)
        aggregates = {aggregate.unique_name(): aggregate
                      for _, _, items in grouped for aggregate in items}
        calculated = _aggregation_vectors(
            aggregates, child, order, times, query_times, starts, ends, left, right)
        for column, feature, _ in grouped:
            output[rows, column] = _evaluate(feature, calculated)
    return output
