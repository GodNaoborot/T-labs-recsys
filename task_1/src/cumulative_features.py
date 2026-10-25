"""Накопительный расчет выбранных формул Featuretools без повторных groupby.

Поддерживаются COUNT(log_id), MEAN, STD, MIN, MAX и MULTIPLY_NUMERIC.
История доступна строго при end_ts < start_ts запроса. Формулы создает DFS;
этот модуль меняет только способ их вычисления. Состояния живут в одном вызове.
"""
import numpy as np
import pandas as pd
import featuretools as ft
from numba import njit
from woodwork.logical_types import Integer, Double

VALUE_COLUMNS = ['label', 'track_length_seconds', 'is_organic']


def make_definitions(parent_name, key, *, agg_primitives, trans_primitives,
                     max_depth, max_features):
    """DFS нужны схема и типы. Большую историю для составления формул не загружаем."""
    parent = pd.DataFrame({key: pd.Series(dtype='int64')})
    columns = ['log_id', key, 'end_ts'] + VALUE_COLUMNS
    child = pd.DataFrame({column: pd.Series(dtype='float64') for column in columns})
    es = ft.EntitySet(id=parent_name)
    es.add_dataframe(dataframe_name=parent_name, dataframe=parent, index=key,
                     logical_types={key: Integer})
    es.add_dataframe(dataframe_name='Interactions', dataframe=child, index='log_id',
                     time_index='end_ts',
                     logical_types={'log_id': Integer, key: Integer, 'end_ts': Integer,
                                    **{column: Double for column in VALUE_COLUMNS}})
    es.add_relationship(parent_name, key, 'Interactions', key)
    return ft.dfs(entityset=es, target_dataframe_name=parent_name, features_only=True,
                  agg_primitives=agg_primitives, trans_primitives=trans_primitives,
                  max_depth=max_depth, max_features=max_features,
                  ignore_columns={'Interactions': ['end_ts']})


@njit
def accumulate_history(history_keys, end_times, values, query_keys, query_times):
    """Оба входа отсортированы по (ключ, время). Каждое событие читаем не более раза.

    Для каждого числового столбца храним count, mean, M2, min, max.
    M2 — сумма квадратов отклонений, обновляемая устойчивой формулой Уэлфорда.
    Numba компилирует этот обычный цикл Python; приближений по времени нет.
    """
    query_count, value_count = len(query_keys), values.shape[1]
    row_counts = np.zeros(query_count)
    stats = np.full((query_count, value_count, 4), np.nan)  # mean, std, min, max
    valid_counts = np.zeros(value_count, dtype=np.int64)
    means = np.zeros(value_count)
    squared_deviations = np.zeros(value_count)
    minima = np.zeros(value_count)
    maxima = np.zeros(value_count)
    history_position = 0
    entity_row_count = 0

    # Запросы одной сущности идут по времени: ее состояние переиспользуется.
    for query_position in range(query_count):
        current_key = query_keys[query_position]
        if query_position == 0 or current_key != query_keys[query_position - 1]:
            entity_row_count = 0
            valid_counts[:] = 0
            means[:] = 0
            squared_deviations[:] = 0
            # Пропускаем остаток предыдущей группы, который ни одному запросу не нужен.
            while (history_position < len(history_keys)
                   and history_keys[history_position] < current_key):
                history_position += 1

        # События ровно на границе запроса остаются недоступными.
        while (history_position < len(history_keys)
               and history_keys[history_position] == current_key
               and end_times[history_position] < query_times[query_position]):
            entity_row_count += 1
            for value_column in range(value_count):
                value = values[history_position, value_column]
                if np.isnan(value):
                    continue
                valid_counts[value_column] += 1
                mean_delta = value - means[value_column]
                means[value_column] += mean_delta / valid_counts[value_column]
                squared_deviations[value_column] += mean_delta * (value - means[value_column])
                if valid_counts[value_column] == 1:
                    minima[value_column] = maxima[value_column] = value
                else:
                    minima[value_column] = min(minima[value_column], value)
                    maxima[value_column] = max(maxima[value_column], value)
            history_position += 1

        # COUNT учитывает строки, а остальные статистики пропускают NaN по столбцам.
        row_counts[query_position] = entity_row_count
        for value_column in range(value_count):
            if valid_counts[value_column] > 0:
                stats[query_position, value_column, 0] = means[value_column]
                stats[query_position, value_column, 2] = minima[value_column]
                stats[query_position, value_column, 3] = maxima[value_column]
            if valid_counts[value_column] > 1:
                # Совпадает с групповым STD текущих Featuretools 1.31 + pandas 2.3:
                # там вызов np.std через pandas groupby фактически использует ddof=1.
                stats[query_position, value_column, 1] = np.sqrt(
                    max(squared_deviations[value_column], 0.0)
                    / (valid_counts[value_column] - 1)
                )
    return row_counts, stats


def _validate_aggregation(feature, parent_name):
    """Отклонить неподдержанные связи, окна, фильтры и виды агрегаций."""
    if (feature.child_dataframe_name != 'Interactions'
            or feature.dataframe_name != parent_name
            or getattr(feature, 'where', None) is not None
            or getattr(feature, 'use_previous', None) is not None):
        raise NotImplementedError('Пока поддержана одна связь без where и окон')
    if feature.primitive.name not in {'count', 'mean', 'std', 'min', 'max'}:
        raise NotImplementedError(feature.get_name())
    if feature.primitive.name == 'count':
        base = feature.base_features[0]
        if not isinstance(base, ft.IdentityFeature) or base.column_name != 'log_id':
            raise NotImplementedError('COUNT пока поддержан только для log_id')


def collect_aggregations(features, parent_name):
    """Собрать уникальные агрегации из дерева допустимых формул в прежнем порядке."""
    aggregations = {}

    def visit(feature):
        """Пройти произведения и сохранить встреченные агрегации."""
        if isinstance(feature, ft.AggregationFeature):
            _validate_aggregation(feature, parent_name)
            aggregations[feature.unique_name()] = feature
        elif (isinstance(feature, ft.TransformFeature)
              and feature.dataframe_name == parent_name
              and feature.primitive.name == 'multiply_numeric'):
            for base in feature.base_features:
                visit(base)
        else:
            raise NotImplementedError(feature.get_name())

    for feature in features:
        visit(feature)
    return aggregations


def event_values(feature, frame, order):
    """Вычисляем выражение уровня события: столбец или произведение столбцов."""
    if isinstance(feature, ft.IdentityFeature):
        if feature.column_name not in VALUE_COLUMNS:
            raise NotImplementedError(feature.get_name())
        return frame[feature.column_name].to_numpy(dtype='float64', na_value=np.nan)[order]
    if isinstance(feature, ft.TransformFeature) and feature.primitive.name == 'multiply_numeric':
        left, right = feature.base_features
        return event_values(left, frame, order) * event_values(right, frame, order)
    raise NotImplementedError(feature.get_name())


def _aggregation_bases(aggregations):
    """Собрать выражения уровня события, необходимые для всех агрегаций кроме COUNT."""
    bases = {}
    for feature in aggregations.values():
        if feature.primitive.name != 'count':
            base = feature.base_features[0]
            bases[base.unique_name()] = base
    return bases


def _event_matrix(history, bases, order):
    """Вычислить нужные столбцы событий в порядке накопительного прохода."""
    values = np.empty((len(history), len(bases)), dtype='float64')
    for column_position, base in enumerate(bases.values()):
        column = event_values(base, history, order)
        if np.isinf(column).any():
            raise ValueError('Накопительный расчет поддерживает конечные значения и NaN')
        values[:, column_position] = column
    return values


def _accumulate_requests(history, requests, key, bases):
    """Отсортировать события и запросы, выполнить проход и вернуть порядок запросов."""
    child = history.loc[history[key].isin(requests[key].unique()),
                        [key, 'end_ts'] + VALUE_COLUMNS]
    history_keys = child[key].to_numpy(dtype='int64')
    end_times = child['end_ts'].to_numpy(dtype='float64')
    query_keys = requests[key].to_numpy(dtype='int64')
    query_times = requests['time'].to_numpy(dtype='float64')
    if not np.isfinite(end_times).all() or not np.isfinite(query_times).all():
        raise ValueError('Для времени нужны конечные числовые значения')
    history_order = np.lexsort((end_times, history_keys))
    query_order = np.lexsort((query_times, query_keys))
    values = _event_matrix(child, bases, history_order)
    row_counts, stats = accumulate_history(
        history_keys[history_order], end_times[history_order], values,
        query_keys[query_order], query_times[query_order])
    del child, values
    inverse_order = np.argsort(query_order)
    return row_counts[inverse_order], stats[inverse_order]


def _aggregation_values(aggregations, bases, row_counts, stats):
    """Связать каждую агрегацию с нужным столбцом рассчитанных статистик."""
    base_positions = {name: position for position, name in enumerate(bases)}
    statistic_columns = {'mean': 0, 'std': 1, 'min': 2, 'max': 3}
    calculated = {}
    for name, feature in aggregations.items():
        primitive = feature.primitive.name
        if primitive == 'count':
            calculated[name] = row_counts
        else:
            position = base_positions[feature.base_features[0].unique_name()]
            calculated[name] = stats[:, position, statistic_columns[primitive]]
    return calculated


def _evaluate_feature(feature, calculated):
    """Получить готовый агрегат либо произведение уже проверенных подформул."""
    if isinstance(feature, ft.AggregationFeature):
        return calculated[feature.unique_name()]
    left, right = feature.base_features
    return _evaluate_feature(left, calculated) * _evaluate_feature(right, calculated)


def calculate_cumulative(history, queries, parent_name, key, definitions):
    """Посчитать фиксированные формулы за один проход по истории каждой сущности.

    Между запросами переиспользуются состояния; отдельный вызов готовит их заново.
    Порядок запросов и float64 сохраняются, строки без ключа остаются с NaN.
    """
    if not queries['log_id'].is_unique:
        raise ValueError('log_id запросов должен быть уникальным')
    aggregations = collect_aggregations(definitions, parent_name)
    bases = _aggregation_bases(aggregations)
    requests = queries[['log_id', key, 'time']].dropna(subset=[key]).copy()
    output_index = pd.Index(queries['log_id'], name='log_id')
    names = [feature.get_name() for feature in definitions]
    if requests.empty:
        return pd.DataFrame(index=output_index, columns=names, dtype='float64')
    requests[key] = requests[key].astype('int64')
    row_counts, stats = _accumulate_requests(history, requests, key, bases)
    calculated = _aggregation_values(aggregations, bases, row_counts, stats)
    result = pd.DataFrame(
        {feature.get_name(): _evaluate_feature(feature, calculated) for feature in definitions},
        index=pd.Index(requests['log_id'], name='log_id'),
    )
    return result.reindex(output_index)
