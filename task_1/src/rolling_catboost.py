"""Подготовка Pool, обучение и парный bootstrap одного временного окна."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from catboost import CatBoostRanker, CatBoostClassifier, Pool
from catboost.utils import eval_metric
from sklearn.metrics import log_loss

from block_bootstrap import aggregate_days, circular_day_indices, group_ndcg
from block_bootstrap import relative_loss_draws, summarize_bootstrap
from bootstrap_validation import file_sha256, read_json, write_json
from rolling_windows import ROOT, DAY, SEED, fixed_model_params, window_paths, release_memory

CHUNK = 100_000
NDCG = 'NDCG:top=10;type=Base;denominator=LogPosition'


def feature_names(kind):
    """Сохранить исходный порядок42manual или130DFS признаков."""
    suffix = 'baselines/timing-20261020' if kind == 'manual' else 'experiment-1/dfs'
    return read_json(ROOT / 'results' / suffix / 'run_config.json')['features']


def selected_rows(meta, split):
    """Упорядочить целые user-day группы и удалить группы с постоянной меткой."""
    original = np.flatnonzero(meta['split'].to_numpy() == split)
    table = pa.table({'group_id': meta['group_id'].to_numpy()[original]})
    order = pc.sort_indices(table, sort_keys=[('group_id', 'ascending')]).to_numpy()
    positions = original[order]
    groups = meta['group_id'].to_numpy()[positions]
    labels = meta['label'].to_numpy()[positions]
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    sizes = np.diff(np.r_[starts, len(groups)])
    mixed = np.minimum.reduceat(labels, starts) != np.maximum.reduceat(labels, starts)
    positions = positions[np.repeat(mixed, sizes)]
    return positions, {'rows_total': len(original), 'groups_total': len(starts),
                       'rows_used': len(positions), 'groups_used': int(mixed.sum())}


def pool_directory(paths, kind):
    """Создать отдельные Pool и пороги квантования для каждого набора признаков."""
    folder = paths['data'] / f'{kind}_pools'
    folder.mkdir(exist_ok=True)
    return folder


def copy_feature_batches(source, matrix, names, destination, source_ids):
    """Перенести только выбранные строки в дисковую float32 матрицу без полной копии."""
    offset = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=CHUNK, columns=['log_id'] + names):
        table = pa.Table.from_batches([batch])
        right = offset + len(table)
        np.testing.assert_array_equal(table['log_id'].to_numpy(), source_ids[offset:right])
        target = destination[offset:right]
        keep = target >= 0
        for index, name in enumerate(names):
            values = table[name].to_numpy().astype('float32')
            if np.isinf(values).any():
                raise ValueError(f'Бесконечные признаки: {name}')
            matrix[target[keep], index] = values[keep]
        offset = right
    assert offset == len(source_ids)
    matrix.flush()


def save_targets(folder, split, meta, positions):
    """Сохранить метки, идентификаторы и календарный день в точном порядке Pool."""
    columns = ['label', 'group_id', 'log_id', 'uid', 'item_id', 'start_ts']
    arrays = {name: meta[name].to_numpy()[positions] for name in columns}
    np.savez_compressed(folder / f'{split}_targets.npz', **arrays)
    return arrays


def quantize_and_save(matrix, arrays, names, folder, split):
    """Обучить пороги только на train и применить их к val без переквантования train."""
    matrix.flags.writeable = False
    pool = Pool(matrix, label=arrays['label'], group_id=arrays['group_id'],
                feature_names=names, thread_count=4)
    borders = folder / 'train_borders.tsv'
    pool.quantize(border_count=254, random_seed=SEED, nan_mode='Min', used_ram_limit='6GB',
                  input_borders=str(borders) if split == 'val' else None)
    if split == 'train':
        pool.save_quantization_borders(str(borders))
    pool.save(str(folder / f'{split}.qbin'))


def prepare_pool(window_id, kind, split):
    """Создать компактный Pool из новых признаков и общего списка целевых событий."""
    started = time.perf_counter()
    paths = window_paths(window_id)
    folder = pool_directory(paths, kind)
    names = feature_names(kind)
    meta = pd.read_parquet(paths['data'] / 'targets.parquet')
    positions, stats = selected_rows(meta, split)
    arrays = save_targets(folder, split, meta, positions)
    destination = np.full(len(meta), -1, dtype='int64')
    destination[positions] = np.arange(len(positions))
    matrix_path = folder / f'{split}_float32.npy'
    matrix = np.lib.format.open_memmap(matrix_path, mode='w+', dtype='float32',
                                      shape=(len(positions), len(names)), fortran_order=True)
    source = paths['data'] / f'{kind}_features.parquet'
    copy_feature_batches(source, matrix, names, destination, meta['log_id'].to_numpy())
    del meta, destination, positions
    return finish_pool(matrix, matrix_path, arrays, names, folder, split, source, stats, started)


def finish_pool(matrix, matrix_path, arrays, names, folder, split, source, stats, started):
    """Зафиксировать источники Pool и освободить временную широкую матрицу."""
    release_memory()
    quantize_and_save(matrix, arrays, names, folder, split)
    del matrix
    release_memory()
    matrix_path.unlink()
    stats.update(features=names, completed=True, pool_sha256=file_sha256(folder / f'{split}.qbin'),
                 targets_sha256=file_sha256(folder / f'{split}_targets.npz'),
                 feature_file_sha256=file_sha256(source),
                 borders_sha256=file_sha256(folder / 'train_borders.tsv'),
                 elapsed_seconds=time.perf_counter() - started)
    write_json(folder / f'{split}_config.json', stats)
    print('POOL_DONE', folder.name, split, stats['rows_used'], flush=True)
    return stats


def load_pool_pair(paths, kind):
    """Загрузить только компактные квантованные train/val и сверить источники."""
    folder = pool_directory(paths, kind)
    pools = {}
    for split in ('train', 'val'):
        info = read_json(folder / f'{split}_config.json')
        assert file_sha256(folder / f'{split}.qbin') == info['pool_sha256']
        assert file_sha256(folder / f'{split}_targets.npz') == info['targets_sha256']
        pools[split] = Pool('quantized://' + str(folder / f'{split}.qbin'), thread_count=4)
    assert pools['train'].get_feature_names() == pools['val'].get_feature_names()
    with np.load(folder / 'val_targets.npz') as data:
        targets = {name: data[name] for name in data.files}
    np.testing.assert_array_equal(np.asarray(pools['val'].get_label(), dtype=float), targets['label'])
    return pools, targets


def model_predictions(model, pool, objective):
    """Ранжировать YetiRank по score, MultiClass по ожидаемой метке."""
    if objective == 'ranker':
        return np.asarray(model.predict(pool, thread_count=4)), None
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    proba = model.predict_proba(pool, thread_count=4)
    return proba @ model.classes_.astype(float), proba


def evaluate_model(model, pool, targets, objective):
    """Проверить NDCG по группам через CatBoost и сохранить дополнительные метрики."""
    score, proba = model_predictions(model, pool, objective)
    labels, groups = targets['label'], targets['group_id']
    ties = np.random.default_rng(SEED).permutation(len(labels))
    values = group_ndcg(labels, groups, score, ties)
    ranks = np.empty(len(score), dtype='float64')
    ranks[np.lexsort((ties, score))] = np.arange(len(score))
    native = eval_metric(labels, ranks, NDCG, group_id=groups, thread_count=4)[0]
    np.testing.assert_allclose(values.mean(), native, rtol=0, atol=1e-12)
    metrics = {'NDCG@10': float(values.mean()), 'QueryAUC': eval_metric(
        labels, score, 'QueryAUC:type=Ranking', group_id=groups, thread_count=4)[0]}
    if proba is not None:
        metrics['logloss'] = float(log_loss(labels, proba, labels=[0, 1, 2]))
    return score, values, metrics


def save_group_scores(targets, values, destination):
    """Сохранить одну запись на исходную ranking-группу без потери календарного дня."""
    groups = targets['group_id']
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    sizes = np.diff(np.r_[starts, len(groups)])
    day = targets['start_ts'] // DAY
    np.testing.assert_array_equal(day, np.repeat(day[starts], sizes))
    frame = pd.DataFrame({'group_id': groups[starts], 'day': day[starts],
                          'rows': sizes, 'ndcg': values})
    frame.to_parquet(destination, index=False)


def verify_reloaded(model, path, validation, groups):
    """Проверить checkpoint на полных группах, не разрывая запрос Pool.slice."""
    end = min(2048, len(groups))
    end = int(np.searchsorted(groups, groups[end - 1], side='right'))
    probe = validation.slice(np.arange(end))
    loaded = type(model)().load_model(str(path))
    np.testing.assert_array_equal(model.predict(probe), loaded.predict(probe))


def fit_model(window_id, kind, objective):
    """Переобучить одну модель с фиксированными параметрами и своим val."""
    paths = window_paths(window_id)
    pools, targets = load_pool_pair(paths, kind)
    params = fixed_model_params(objective)
    estimator = CatBoostRanker if objective == 'ranker' else CatBoostClassifier
    model = estimator(**params)
    print('FIT_START', window_id, kind, objective, flush=True)
    started = time.perf_counter()
    model.fit(pools['train'], eval_set=pools['val'])
    fit_seconds = time.perf_counter() - started
    name = f'{kind}_{objective}'
    path = paths['models'] / f'{name}.cbm'
    model.save_model(str(path))
    verify_reloaded(model, path, pools['val'], targets['group_id'])
    score, values, metrics = evaluate_model(model, pools['val'], targets, objective)
    return save_model_results(paths, name, model, path, targets, score, values,
                              metrics, fit_seconds, params)


def save_model_results(paths, name, model, path, targets, score, values, metrics, seconds, params):
    """Сохранить метрики и прогнозы до отметки этапа как завершенного."""
    output = paths['results']
    np.save(output / f'{name}_val_score.npy', score)
    save_group_scores(targets, values, output / f'{name}_group_ndcg.parquet')
    write_json(output / f'{name}_learning_curve.json', model.get_evals_result())
    model.get_feature_importance(type='PredictionValuesChange', prettified=True).to_csv(
        output / f'{name}_feature_importance.csv', index=False)
    report = {'completed': True, 'path': str(path), 'sha256': file_sha256(path),
              'fit_seconds': seconds, 'best_iteration': model.get_best_iteration(),
              'tree_count': model.tree_count_, 'params': params,
              'actual_params': model.get_all_params(), 'metrics': metrics,
              'score_sha256': file_sha256(output / f'{name}_val_score.npy'),
              'group_ndcg_sha256': file_sha256(output / f'{name}_group_ndcg.parquet'),
              'reload_predictions_exact': True, 'test_used': False}
    write_json(output / f'{name}_report.json', report)
    print('MODEL_DONE', name, metrics, 'fit_seconds', seconds, flush=True)
    return report


def assert_matching_targets(paths):
    """Проверить точное совпадение train/val событий у manual и DFS."""
    for split in ('train', 'val'):
        with np.load(paths['data'] / f'manual_pools/{split}_targets.npz') as left:
            with np.load(paths['data'] / f'dfs_pools/{split}_targets.npz') as right:
                assert left.files == right.files
                for name in left.files:
                    np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def bootstrap_objective(paths, spec, objective):
    """Сравнить парные дневные суммы для обеих заранее заданных длин блока."""
    source = paths['results']
    base = pd.read_parquet(source / f'manual_{objective}_group_ndcg.parquet')
    dfs = pd.read_parquet(source / f'dfs_{objective}_group_ndcg.parquet')
    np.testing.assert_array_equal(base[['group_id', 'day', 'rows']], dfs[['group_id', 'day', 'rows']])
    daily = aggregate_days(base['day'], base['ndcg'], dfs['ndcg'])
    np.testing.assert_array_equal(daily['day'], np.arange(spec['val_start'] // DAY, spec['val_end'] // DAY))
    daily.to_csv(source / f'{objective}_daily_ndcg.csv', index=False)
    rows, draws = [], {}
    for length in (3, 4):
        indices = circular_day_indices(len(daily), length, 20_000, SEED)
        losses = relative_loss_draws(daily, indices)
        rows.append({'objective': objective, 'block_days': length,
                     'groups': len(base), **summarize_bootstrap(daily, losses)})
        draws[f'{objective}_L{length}'] = losses
    return rows, draws


def bootstrap_window(spec):
    """Сохранить все четыре проверки окна без выбора лучшего objective или блока."""
    paths = window_paths(spec['id'])
    assert_matching_targets(paths)
    rows, draws = [], {}
    for objective in ('ranker', 'pointwise'):
        result, values = bootstrap_objective(paths, spec, objective)
        rows.extend(result)
        draws.update(values)
    frame = pd.DataFrame(rows)
    frame.insert(0, 'window', spec['id'])
    frame.to_csv(paths['results'] / 'bootstrap_summary.csv', index=False)
    np.savez_compressed(paths['results'] / 'bootstrap_losses.npz', **draws)
    return frame
