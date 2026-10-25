"""Отбор временных DFS-признаков по CatBoost и выбор k на validation.

Первое обучение использует все кандидаты. Важности LossFunctionChange считаются
на train, после чего каждый top-k обучается заново с теми же параметрами.
Validation выбирает k и итерацию; это исследовательская оценка, test не читается.
"""
from pathlib import Path
import gc
import hashlib
import json
import time

import catboost
from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

from train_dfs_catboost import NDCG, ranking_metrics


TIE_TOLERANCE = 1e-12


def _write_json(path, value):
    """Сохранить строгий JSON атомарно, не оставляя частично записанный отчет."""
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def _sha256(path):
    """Прочитать checksum модели небольшими блоками."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validate_targets(targets, rows, split):
    """Проверить метки, целые последовательные группы и дополнительные row ids."""
    result = {name: np.asarray(value) for name, value in targets.items()}
    if not {'label', 'group_id'} <= result.keys():
        raise ValueError(f'{split}: нужны label и group_id')
    for name, values in result.items():
        if values.ndim != 1 or len(values) != rows:
            raise ValueError(f'{split}: длина или размерность {name} не совпала с матрицей')
    labels, groups = result['label'], result['group_id']
    if labels.dtype.kind not in 'fiub' or not np.isin(labels, [0, 1, 2]).all() or not np.isfinite(labels).all():
        raise ValueError(f'{split}: label должен принимать значения 0/1/2')
    if groups.dtype.kind not in 'iu' or (groups[1:] < groups[:-1]).any():
        raise ValueError(f'{split}: group_id должны быть целыми и отсортированными')
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    if (np.minimum.reduceat(labels, starts) == np.maximum.reduceat(labels, starts)).any():
        raise ValueError(f'{split}: постоянные по label группы нужно удалить до отбора')
    if 'log_id' in result:
        ids = result['log_id']
        if ids.dtype.kind not in 'iu' or len(np.unique(ids)) != rows:
            raise ValueError(f'{split}: log_id должны быть уникальными целыми')
    if 'day' in result and not np.array_equal(result['day'], groups % 1000):
        raise ValueError(f'{split}: day не совпал с group_id % 1000')
    return result


def _validate_inputs(train_matrix, val_matrix, train_targets, val_targets, names):
    """Проверить размеры входов без полной копии широких memmap-матриц."""
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('Имена признаков должны быть непустыми уникальными строками')
    targets = []
    for split, matrix, data in [('train', train_matrix, train_targets),
                                 ('val', val_matrix, val_targets)]:
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(names):
            raise ValueError(f'{split}: неверная форма матрицы признаков')
        if matrix.dtype.kind not in 'fiu':
            raise ValueError(f'{split}: нужны числовые признаки')
        # NaN допустим, а бесконечности проверяем порциями столбцов.
        varied = False
        for left in range(0, matrix.shape[1], 64):
            block = matrix[:, left:left + 64]
            if np.isinf(block).any():
                raise ValueError(f'{split}: бесконечный признак')
            if split == 'train' and not varied:
                same_missing = np.isnan(block) & np.isnan(block[0])
                varied = ((block != block[0]) & ~same_missing).any()
        if split == 'train' and not varied:
            raise ValueError('train: все признаки постоянны')
        targets.append(_validate_targets(data, len(matrix), split))
    if 'log_id' in targets[0] and 'log_id' in targets[1]:
        if np.intersect1d(targets[0]['log_id'], targets[1]['log_id']).size:
            raise ValueError('Train и validation содержат одинаковые log_id')
    return targets


def _k_grid(k_values, feature_count):
    """Оставить допустимые k; полный набор всегда остается точкой сравнения."""
    requested = list(k_values)
    if not requested or any(isinstance(k, (bool, np.bool_)) or
                            not isinstance(k, (int, np.integer)) or k <= 0 for k in requested):
        raise ValueError('k_values должен содержать положительные целые числа')
    return sorted({int(k) for k in requested if k < feature_count} | {feature_count})


def _model_parameters(iterations, seed, threads):
    """Зафиксировать одну конфигурацию для широкого набора и всех top-k."""
    if not isinstance(iterations, int) or iterations <= 0 or not isinstance(threads, int) or threads <= 0:
        raise ValueError('iterations и threads должны быть положительными целыми')
    return dict(loss_function='YetiRank', eval_metric=NDCG, depth=6,
                learning_rate=0.1, l2_leaf_reg=1e-20, iterations=iterations,
                random_seed=seed, thread_count=threads, early_stopping_rounds=50,
                use_best_model=True, bootstrap_type='MVS', subsample=0.8,
                random_strength=1., rsm=1., used_ram_limit='3GB',
                allow_writing_files=False, verbose=False)


def _target_fingerprint(targets):
    """Связать отчет с точным порядком labels/groups и log_id, если они переданы."""
    digest = hashlib.sha256()
    for name in sorted(targets):
        values = np.ascontiguousarray(targets[name])
        digest.update(name.encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.view('uint8'))
    return digest.hexdigest()


def _pool(matrix, targets, feature_names, threads, indices=None):
    """Собрать Pool; копия выбранных столбцов нужна только для небольшого top-k."""
    values = matrix if indices is None else np.asfortranarray(matrix[:, indices])
    return Pool(values, label=targets['label'], group_id=targets['group_id'],
                feature_names=feature_names, thread_count=threads)


def _fit_and_save(train, val, targets, names, indices, folder, parameters):
    """Обучить новую модель, сохранить метрики и проверить точную повторную загрузку."""
    folder.mkdir(parents=True, exist_ok=False)
    model = CatBoostRanker(**parameters)
    started = time.perf_counter()
    model.fit(train, eval_set=val)
    fit_seconds = time.perf_counter() - started
    scores = np.asarray(model.predict(val, thread_count=parameters['thread_count']))
    metrics = {key: float(value) for key, value in
               ranking_metrics(targets['label'], targets['group_id'], scores).items()}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError('Получена нечисловая validation-метрика')
    path = folder / 'model.cbm'
    model.save_model(str(path))
    loaded = CatBoostRanker().load_model(str(path))
    np.testing.assert_array_equal(scores, loaded.predict(val, thread_count=parameters['thread_count']))
    np.save(folder / 'val_score.npy', scores)
    _write_json(folder / 'params.json', model.get_all_params())
    _write_json(folder / 'learning_curve.json', model.get_evals_result())
    record = {'k': len(names), 'features': names, 'indices': [int(i) for i in indices],
              'metrics': metrics, 'fit_seconds': float(fit_seconds),
              'best_iteration': int(model.get_best_iteration()), 'tree_count': int(model.tree_count_),
              'model_path': str(path.resolve()), 'model_sha256': _sha256(path),
              'reload_predictions_exact': True}
    _write_json(folder / 'metrics.json', record)
    return record, model


def _rank_importance(model, train_pool, names, threads):
    """Упорядочить LossFunctionChange на train; равенство разрешается по имени."""
    values = np.asarray(model.get_feature_importance(
        data=train_pool, type='LossFunctionChange', thread_count=threads), dtype=float)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError('Важности признаков должны быть конечными и совпадать с именами')
    order = np.lexsort((np.asarray(names), -values))
    table = pd.DataFrame({'rank': np.arange(1, len(names) + 1),
                          'feature_index': order, 'feature': np.asarray(names)[order],
                          'importance': values[order]})
    return order, table


def choose_best_candidate(records, tolerance=TIE_TOLERANCE):
    """Максимизировать внешний NDCG; при почти равных значениях выбрать меньше k."""
    best = max(record['metrics']['NDCG@10'] for record in records)
    eligible = [record for record in records
                if record['metrics']['NDCG@10'] >= best - tolerance]
    return min(eligible, key=lambda record: record['k'])


def _save_curve(folder, records):
    """Сохранить уже завершенные точки даже до конца всей очереди k."""
    rows = [{'k': record['k'], **record['metrics'],
             'fit_seconds': record['fit_seconds'], 'best_iteration': record['best_iteration'],
             'tree_count': record['tree_count'], 'model_path': record['model_path']}
            for record in records]
    pd.DataFrame(rows).sort_values('k').to_csv(folder / 'k_curve.csv', index=False)


def select_top_k(train_matrix, val_matrix, train_targets, val_targets, feature_names,
                 output_dir, *, k_values=(32, 64, 128, 256), iterations=300,
                 seed=42, threads=4, model_kind='ranker'):
    """Обучить широкий ranker, переобучить top-k и сохранить выбранный по val набор.

    Ожидаются полные отсортированные user-day группы без постоянных labels.
    Строки матриц должны соответствовать targets; log_id проверяются на уникальность
    и непересечение split. k выше числа кандидатов пропускаются; полный набор
    участвует в выборе всегда. Непустой output_dir защищен от перезаписи.
    """
    if model_kind != 'ranker':
        raise ValueError('В этом эксперименте отбор реализован только для YetiRank')
    names = list(feature_names)
    train_targets, val_targets = _validate_inputs(
        train_matrix, val_matrix, train_targets, val_targets, names)
    requested_k = list(k_values)
    grid = _k_grid(requested_k, len(names))
    parameters = _model_parameters(iterations, seed, threads)
    folder = Path(output_dir)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Папка отбора не пуста: для нового опыта нужен отдельный output_dir')
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {'model_kind': model_kind, 'model_parameters': parameters,
                'candidate_count': len(names), 'features': names,
                'requested_k': [int(k) for k in requested_k], 'evaluated_k': grid,
                'importance_type': 'LossFunctionChange', 'importance_split': 'train',
                'selection_metric': 'external NDCG@10', 'tie_tolerance': TIE_TOLERANCE,
                'tie_rule': 'smallest k within tolerance of maximum score',
                'train_rows': len(train_matrix), 'val_rows': len(val_matrix),
                'train_targets_sha256': _target_fingerprint(train_targets),
                'val_targets_sha256': _target_fingerprint(val_targets),
                'test_used': False, 'catboost_version': catboost.__version__,
                'limitations': ['Validation reused for early stopping and k selection.',
                                'Feature importance is conditional on the wide model and train sample.',
                                'Soft CatBoost RAM hint does not replace an external memory/time guard.']}
    _write_json(folder / 'protocol.json', protocol)
    records = []
    try:
        _write_json(folder / 'status.json', {'completed': False, 'phase': 'all_features'})
        train = _pool(train_matrix, train_targets, names, threads)
        val = _pool(val_matrix, val_targets, names, threads)
        record, model = _fit_and_save(train, val, val_targets, names, range(len(names)),
                                      folder / 'models/all_features', parameters)
        records.append(record)
        _save_curve(folder, records)
        _write_json(folder / 'status.json', {'completed': False, 'phase': 'importance'})
        started = time.perf_counter()
        order, table = _rank_importance(model, train, names, threads)
        importance_seconds = time.perf_counter() - started
        table.to_csv(folder / 'feature_importance.csv', index=False)
        del model, train, val
        gc.collect()
        for k in grid:
            if k == len(names):
                continue  # Широкая модель уже обучена; второй одинаковый fit не нужен.
            _write_json(folder / 'status.json', {'completed': False, 'phase': 'top_k', 'k': k})
            indices = order[:k]
            selected = [names[i] for i in indices]
            train = _pool(train_matrix, train_targets, selected, threads, indices)
            val = _pool(val_matrix, val_targets, selected, threads, indices)
            record, model = _fit_and_save(train, val, val_targets, selected, indices,
                                          folder / f'models/top_{k}', parameters)
            records.append(record)
            _save_curve(folder, records)
            del model, train, val
            gc.collect()
        chosen = choose_best_candidate(records)
        summary = {'completed': True, 'selected_k': chosen['k'],
                   'selected_features': chosen['features'], 'selected_indices': chosen['indices'],
                   'metrics': chosen['metrics'], 'model_path': chosen['model_path'],
                   'model_sha256': chosen['model_sha256'], 'candidate_count': len(names),
                   'importance_seconds': float(importance_seconds), 'test_used': False,
                   'curve_path': str((folder / 'k_curve.csv').resolve()),
                   'protocol_path': str((folder / 'protocol.json').resolve())}
        _write_json(folder / 'selected_features.json', summary)
        _write_json(folder / 'status.json', {**summary, 'phase': 'finished'})
        return summary
    except Exception as error:
        _write_json(folder / 'status.json', {'completed': False, 'phase': 'failed',
                                            'error': repr(error), 'completed_k': [r['k'] for r in records]})
        raise
