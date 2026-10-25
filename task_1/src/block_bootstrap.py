"""Парный циклический bootstrap дневных сумм NDCG без обучения моделей.

Исходная метрика усредняется по ranking-группам. Ресемплирование дней
меняет кратности этих групп, сохраняя их исходный вес внутри метрики.
"""
import numpy as np
import pandas as pd


def _numeric_vector(values, name):
    """Проверить непустой одномерный массив конечных чисел."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f'{name}: нужен непустой одномерный массив')
    if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
        raise ValueError(f'{name}: нужны конечные числовые значения')
    return values


def _matched_vectors(**values):
    """Проверить согласованную длину нескольких числовых векторов."""
    arrays = [_numeric_vector(value, name) for name, value in values.items()]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError('Все векторы должны иметь одинаковую длину')
    return arrays


def _positive_integer(value, name):
    """Проверить положительный целочисленный размер."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name}: нужно положительное целое число')
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name}: нужно положительное целое число')
    return int(value)


def _whole_days(values):
    """Преобразовать целые номера календарных дней в int64."""
    values = _numeric_vector(values, 'days')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError('days: номера дней должны быть целыми')
    limits = np.iinfo(np.int64)
    if int(values.min()) < limits.min or int(values.max()) > limits.max:
        raise ValueError('days: номера дней выходят за диапазон int64')
    return values.astype('int64', copy=False)


def _one_group_ndcg(labels, scores, tie_breaker, top):
    """Посчитать линейный NDCG одной группы с фиксированными точными ties."""
    size = min(top, len(labels))
    discounts = 1.0 / np.log2(np.arange(2, size + 2))
    ideal = np.sort(labels)[::-1][:size]
    ideal_dcg = float(np.dot(ideal, discounts))
    if ideal_dcg <= 0:
        raise ValueError('IDCG группы должен быть положительным')
    order = np.lexsort((tie_breaker, scores))[::-1][:size]
    return float(np.dot(labels[order], discounts) / ideal_dcg)


def group_ndcg(labels, groups, scores, tie_breaker, top=10):
    """Вернуть NDCG каждой группы; строки должны идти по возрастанию group_id."""
    labels, groups, scores, tie_breaker = _matched_vectors(
        labels=labels, groups=groups, scores=scores, tie_breaker=tie_breaker)
    top = _positive_integer(top, 'top')
    if np.any(groups[1:] < groups[:-1]):
        raise ValueError('Строки должны быть отсортированы по group_id')
    if np.any(labels < 0):
        raise ValueError('Для линейного NDCG метки должны быть неотрицательными')
    boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    return np.array([
        _one_group_ndcg(labels[left:right], scores[left:right], tie_breaker[left:right], top)
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ], dtype='float64')


def aggregate_days(days, base, dfs):
    """Собрать дневные суммы NDCG и число групп, включая пустые дни между ними."""
    days, base, dfs = _matched_vectors(days=days, base=base, dfs=dfs)
    days = _whole_days(days)
    if any(np.any((value < 0) | (value > 1 + 1e-12)) for value in (base, dfs)):
        raise ValueError('Групповые значения NDCG должны находиться от 0 до 1')
    data = pd.DataFrame({'day': days, 'base_sum': base, 'dfs_sum': dfs})
    daily = data.groupby('day', sort=True).agg(
        group_count=('base_sum', 'size'),
        base_sum=('base_sum', 'sum'), dfs_sum=('dfs_sum', 'sum'))
    calendar = np.arange(days.min(), days.max() + 1, dtype='int64')
    daily = daily.reindex(calendar, fill_value=0)
    daily.index.name = 'day'
    return daily.reset_index()


def _daily_arrays(daily):
    """Проверить полный календарь и неотрицательные дневные sufficient statistics."""
    required = ('day', 'group_count', 'base_sum', 'dfs_sum')
    if not isinstance(daily, pd.DataFrame) or not set(required).issubset(daily.columns):
        raise ValueError(f'daily должен содержать столбцы {required}')
    days, counts, base, dfs = _matched_vectors(**{name: daily[name] for name in required})
    days = _whole_days(days)
    if not np.all(np.diff(days) == 1):
        raise ValueError('daily должен содержать последовательные календарные дни')
    if np.any(counts < 0) or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('group_count должен содержать целые неотрицательные числа')
    if np.any(base < 0) or np.any(dfs < 0):
        raise ValueError('Дневные суммы NDCG должны быть неотрицательными')
    if np.any((counts == 0) & ((base != 0) | (dfs != 0))):
        raise ValueError('У пустого дня суммы NDCG должны быть нулевыми')
    return counts.astype('float64'), base.astype('float64'), dfs.astype('float64')


def _checked_indices(day_indices, n_days):
    """Проверить индексы реплик с исходным числом дневных позиций."""
    indices = np.asarray(day_indices)
    if indices.ndim != 2 or indices.shape[0] < 1 or indices.shape[1] != n_days:
        raise ValueError('day_indices должен иметь форму (число реплик, число дней)')
    if indices.dtype.kind not in 'iu':
        raise ValueError('day_indices должен содержать целые индексы')
    if np.any(indices < 0) or np.any(indices >= n_days):
        raise ValueError('day_indices содержит индекс вне календаря')
    return indices.astype('int64', copy=False)


def circular_day_indices(n_days, block_days, n_resamples, seed=42):
    """Выбрать циклические блоки с возвращением и обрезать хвост до n_days."""
    n_days = _positive_integer(n_days, 'n_days')
    block_days = _positive_integer(block_days, 'block_days')
    n_resamples = _positive_integer(n_resamples, 'n_resamples')
    if block_days > n_days:
        raise ValueError('block_days не должен превышать число дней')
    rng = np.random.default_rng(seed)
    block_count = (n_days + block_days - 1) // block_days
    starts = rng.integers(n_days, size=(n_resamples, block_count))
    blocks = (starts[:, :, None] + np.arange(block_days)) % n_days
    return blocks.reshape(n_resamples, -1)[:, :n_days]


def _require_positive_base(values):
    """Запретить неопределенную относительную потерю при нулевом baseline."""
    values = np.atleast_1d(values)
    bad = np.flatnonzero((values <= 0) | ~np.isfinite(values))
    if len(bad):
        raise ValueError('Сумма NDCG baseline должна быть положительной и конечной; '
                         f'неверные индексы: {bad[:5].tolist()}')


def relative_loss_draws(daily, day_indices):
    """Посчитать парные потери 1 - sum(DFS) / sum(base) для выбранных дней."""
    counts, base, dfs = _daily_arrays(daily)
    indices = _checked_indices(day_indices, len(counts))
    base_sums = base[indices].sum(axis=1)
    dfs_sums = dfs[indices].sum(axis=1)
    _require_positive_base(base_sums)
    if not np.isfinite(dfs_sums).all():
        raise ValueError('Суммы NDCG DFS в репликах должны быть конечными')
    return 1.0 - dfs_sums / base_sums


def _daily_totals(daily):
    """Получить конечные полные суммы и проверить определенность точечной оценки."""
    arrays = _daily_arrays(daily)
    count, base, dfs = (float(value.sum()) for value in arrays)
    if not np.isfinite([count, base, dfs]).all() or count <= 0:
        raise ValueError('Полные дневные суммы должны быть конечными, число групп — положительным')
    _require_positive_base(base)
    return count, base, dfs


def _test_settings(margin, alpha_sig):
    """Проверить допуск потери и уровень значимости односторонней проверки."""
    margin, alpha_sig = float(margin), float(alpha_sig)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('margin должен быть конечным и неотрицательным')
    if not np.isfinite(alpha_sig) or not 0 < alpha_sig < 1:
        raise ValueError('alpha_sig должен находиться строго между 0 и 1')
    return margin, alpha_sig


def summarize_bootstrap(daily, losses, margin=0.03, alpha_sig=0.05):
    """Сравнить верхнюю одностороннюю percentile-границу с допустимой потерей."""
    margin, alpha_sig = _test_settings(margin, alpha_sig)
    count, base, dfs = _daily_totals(daily)
    losses = _numeric_vector(losses, 'losses')
    upper = float(np.quantile(losses, 1 - alpha_sig, method='linear'))
    return {
        'base_ndcg': base / count, 'dfs_ndcg': dfs / count,
        'relative_loss': 1.0 - dfs / base, 'upper_bound': upper,
        'reject_h0': bool(upper < margin), 'n_resamples': int(len(losses)),
        'margin': margin, 'alpha_sig': alpha_sig,
        'bootstrap_std': float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
    }
