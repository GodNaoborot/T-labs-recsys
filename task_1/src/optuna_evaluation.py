"""Оценка выбранной Optuna-модели на фиксированном validation без дообучения.

Строки Pool и targets должны совпадать и быть отсортированы по group_id.
Модуль сохраняет внешние метрики и sufficient statistics для парного bootstrap.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from block_bootstrap import aggregate_days, group_ndcg
from train_dfs_catboost import ranking_metrics

SEED = 42
THREADS = 4
ARTIFACT_NAMES = {
    'model': 'model.cbm', 'params': 'params.json', 'metrics': 'metrics.json',
    'score': 'val_score.npy', 'probabilities': 'val_proba.npy',
    'groups': 'group_ndcg.parquet', 'daily': 'daily_ndcg.csv',
    'importance': 'feature_importance.csv', 'learning_curve': 'learning_curve.json',
    'evaluation': 'evaluation.json',
}


def _target_vector(targets, name):
    """Проверить непустой числовой вектор targets без изменения порядка строк."""
    if name not in targets:
        raise ValueError(f'targets не содержит {name}')
    values = np.asarray(targets[name])
    if values.ndim != 1 or not len(values):
        raise ValueError(f'{name}: нужен непустой одномерный массив')
    if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
        raise ValueError(f'{name}: нужны конечные числовые значения')
    if name != 'label' and not np.equal(values, np.floor(values)).all():
        raise ValueError(f'{name}: нужны целые идентификаторы или дни')
    return values


def _validate_targets(pool, targets):
    """Проверить длины, log_id, порядок групп и постоянство дня внутри группы."""
    arrays = {name: _target_vector(targets, name) for name in ('label', 'group_id', 'log_id', 'day')}
    if len({len(value) for value in arrays.values()}) != 1 or len(arrays['label']) != pool.num_row():
        raise ValueError('Pool и все массивы targets должны иметь одинаковую длину')
    if len(np.unique(arrays['log_id'])) != len(arrays['log_id']):
        raise ValueError('log_id должен быть уникальным')
    labels, groups, days = arrays['label'], arrays['group_id'], arrays['day']
    if not np.isin(labels, [0, 1, 2]).all() or np.any(groups[1:] < groups[:-1]):
        raise ValueError('Нужны метки 0/1/2 и строки по возрастанию group_id')
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    if np.any(np.minimum.reduceat(days, starts) != np.maximum.reduceat(days, starts)):
        raise ValueError('Все строки ranking-группы должны относиться к одному дню')
    if np.any(np.minimum.reduceat(labels, starts) == np.maximum.reduceat(labels, starts)):
        raise ValueError('Постоянные по label группы должны быть исключены до оценки')
    return arrays, starts


def model_scores(model, kind, pool):
    """Получить raw score ранкера или математическое ожидание метки classifier."""
    if kind == 'ranker':
        return np.asarray(model.predict(pool, thread_count=THREADS), dtype='float64'), None
    if kind != 'pointwise':
        raise ValueError('kind должен быть ranker или pointwise')
    probabilities = np.asarray(model.predict_proba(pool, thread_count=THREADS), dtype='float64')
    classes = np.asarray(model.classes_, dtype='float64')
    if probabilities.ndim != 2 or probabilities.shape != (pool.num_row(), len(classes)):
        raise ValueError('Форма predict_proba не соответствует Pool и classes_')
    if not np.isfinite(probabilities).all() or not np.isfinite(classes).all():
        raise ValueError('Вероятности и classes_ должны быть конечными')
    return probabilities @ classes, probabilities


def _score_metrics(model, kind, pool, arrays):
    """Посчитать те же внешние NDCG/QueryAUC и log loss для classifier."""
    score, probabilities = model_scores(model, kind, pool)
    if score.shape != arrays['label'].shape or not np.isfinite(score).all():
        raise ValueError('Нужен один конечный score на строку Pool')
    metrics = {name: float(value) for name, value in ranking_metrics(
        arrays['label'], arrays['group_id'], score).items()}
    if probabilities is not None:
        metrics['logloss'] = float(log_loss(arrays['label'], probabilities, labels=model.classes_))
    return score, probabilities, metrics


def _group_scores(arrays, starts, score, expected_ndcg):
    """Сверить средний групповой NDCG с внешним CatBoost oracle при тех же ties."""
    ties = np.random.default_rng(SEED).permutation(len(score))
    values = group_ndcg(arrays['label'], arrays['group_id'], score, ties, top=10)
    np.testing.assert_allclose(values.mean(), expected_ndcg, rtol=0, atol=1e-12)
    sizes = np.diff(np.r_[starts, len(score)])
    return pd.DataFrame({'group_id': arrays['group_id'][starts], 'day': arrays['day'][starts],
                         'group_size': sizes, 'ndcg': values})


def _daily_scores(group_scores):
    """Сохранить дневные суммы с исходным весом ranking-групп и полным календарем."""
    values = group_scores['ndcg'].to_numpy()
    daily = aggregate_days(group_scores['day'].to_numpy(), values, values)
    daily = daily.drop(columns='dfs_sum').rename(columns={'base_sum': 'ndcg_sum'})
    daily['ndcg_mean'] = daily['ndcg_sum'].div(daily['group_count'].replace(0, np.nan))
    return daily


def _output_paths(output_dir, kind):
    """Выбрать собственные файлы оценки, не перезаписывая существующие артефакты."""
    if kind not in {'ranker', 'pointwise'}:
        raise ValueError('kind должен быть ranker или pointwise')
    names = {name: filename for name, filename in ARTIFACT_NAMES.items()
             if name != 'probabilities' or kind == 'pointwise'}
    paths = {name: Path(output_dir) / filename for name, filename in names.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f'Артефакты оценки уже существуют: {existing[:3]}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path, value):
    """Записать JSON через временный файл без незавершенного конечного отчета."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _verify_reload(model, kind, path, pool, groups):
    """Сверить score и вероятности после загрузки на первых целых ranking-группах."""
    end = int(np.searchsorted(groups, groups[min(2047, len(groups) - 1)], side='right'))
    probe = pool.slice(np.arange(end))
    loaded = type(model)().load_model(str(path))
    expected_score, expected_proba = model_scores(model, kind, probe)
    actual_score, actual_proba = model_scores(loaded, kind, probe)
    np.testing.assert_allclose(actual_score, expected_score, rtol=0, atol=0)
    if expected_proba is not None:
        np.testing.assert_allclose(actual_proba, expected_proba, rtol=0, atol=0)


def _save_tables(paths, model, score, probabilities, groups):
    """Сохранить прогнозы, групповые и дневные статистики и диагностические таблицы."""
    np.save(paths['score'], score)
    if probabilities is not None:
        np.save(paths['probabilities'], probabilities)
    groups.to_parquet(paths['groups'], index=False)
    _daily_scores(groups).to_csv(paths['daily'], index=False)
    model.get_feature_importance(type='PredictionValuesChange', prettified=True).to_csv(
        paths['importance'], index=False)
    _write_json(paths['params'], model.get_all_params())
    _write_json(paths['learning_curve'], model.get_evals_result())


def _evaluation_record(model, kind, paths, metrics, rows, groups):
    """Описать оценку; время fit, хеши входов и provenance добавляет вызывающий код."""
    return {'kind': kind, 'rows': int(rows), 'groups': int(groups), 'metrics': metrics,
            'model_path': str(paths['model']), 'params': model.get_all_params(),
            'tree_count': int(model.tree_count_),
            'best_iteration': None if model.get_best_iteration() is None else int(model.get_best_iteration()),
            'reload_predictions_exact': True, 'tie_seed': SEED,
            'importance_type': 'PredictionValuesChange',
            'ndcg_ties': 'fixed-seed random permutation, exact score ties only',
            'artifacts': {name: str(path) for name, path in paths.items()}}


def evaluate_and_save(model, kind, pool, targets, output_dir):
    """Оценить выбранную модель на val и сохранить проверенные артефакты без fit.

    targets: label, group_id, log_id, day — по одному значению на строку Pool.
    output_dir не должен содержать прежние собственные файлы этой оценки.
    """
    arrays, starts = _validate_targets(pool, targets)
    paths = _output_paths(output_dir, kind)
    score, probabilities, metrics = _score_metrics(model, kind, pool, arrays)
    groups = _group_scores(arrays, starts, score, metrics['NDCG@10'])
    model.save_model(str(paths['model']))
    _verify_reload(model, kind, paths['model'], pool, arrays['group_id'])
    _save_tables(paths, model, score, probabilities, groups)
    record = _evaluation_record(model, kind, paths, metrics, len(score), len(groups))
    _write_json(paths['metrics'], metrics)
    _write_json(paths['evaluation'], record)
    return record
