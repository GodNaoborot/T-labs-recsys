"""Оценка сохранённых ранкеров на test без обучения и выбора моделей."""
from pathlib import Path
import gc
import hashlib
import json
import time

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

from block_bootstrap import aggregate_days, circular_day_indices
from compare_temporal_controls_pairs import absolute_delta_draws
from optuna_evaluation import _validate_targets, _group_scores, _daily_scores
from train_dfs_catboost import ranking_metrics

TARGET_KEYS = ('label', 'group_id', 'log_id', 'day')
THREADS = 4


def file_hash(path):
    """Посчитать SHA256 порциями, не читая большой Pool целиком в RAM."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Публиковать только полностью записанный JSON."""
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_targets(path):
    """Прочитать четыре общих массива, не меняя canonical-порядок."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in TARGET_KEYS}


def verify_schema(model_record, plan):
    """Проверить зафиксированные физические входы и маску активных признаков."""
    features = list(plan['schemas'][model_record['schema']])
    if not features or len(features) != len(set(features)):
        raise ValueError('Нужен непустой уникальный список входов')
    forbidden = {'label', 'uid', 'item_id', 'group_id', 'log_id', 'day',
                 'timestamp', 'start_ts', 'played_ratio_pct'}
    if forbidden.intersection(features):
        raise ValueError('Сырые метки, идентификаторы или время попали в X')
    ignored = list(model_record['ignored_indices'])
    if len(ignored) != len(set(ignored)) or any(i < 0 or i >= len(features) for i in ignored):
        raise ValueError('Некорректная маска входов')
    active = [name for index, name in enumerate(features) if index not in ignored]
    if active != model_record['active_features'] or len(active) != model_record['active_count']:
        raise ValueError('Маска не совпадает со списком активных признаков')
    return features


def check_inputs(model_record, plan, data_dir):
    """Сверить модель, пороги, все targets и схему перед predict."""
    data_dir = Path(data_dir)
    schema_dir = data_dir / model_record['schema']
    model_path = Path(model_record['model']['path'])
    if file_hash(model_path) != model_record['model']['sha256']:
        raise ValueError('Изменился hash сохранённой модели')
    border = plan['train_borders'][model_record['schema']]
    if file_hash(border['path']) != border['sha256']:
        raise ValueError('Изменились зафиксированные train-пороги')
    if file_hash(schema_dir / 'test_applied_borders.tsv') != border['sha256']:
        raise ValueError('Test получил другие пороги')
    canonical = load_targets(data_dir / 'targets.npz')
    schema_targets = load_targets(schema_dir / 'test_targets.npz')
    for name in TARGET_KEYS:
        np.testing.assert_array_equal(schema_targets[name], canonical[name])
    features = verify_schema(model_record, plan)
    return canonical, features, schema_dir / 'test.qbin'


def check_pool(pool, arrays, features):
    """Сверить реальные входы Pool, метки и границы групп с общими targets."""
    if pool.get_feature_names() != list(features):
        raise ValueError('Pool изменил порядок или число входных колонок')
    validated, starts = _validate_targets(pool, arrays)
    np.testing.assert_array_equal(np.asarray(pool.get_label(), dtype='float64'),
                                  validated['label'])
    hashes = np.asarray(pool.get_group_id_hash())
    np.testing.assert_array_equal(hashes[1:] != hashes[:-1],
                                  validated['group_id'][1:] != validated['group_id'][:-1])
    if len(np.unique(hashes[starts])) != len(starts):
        raise ValueError('Групповые hash столкнулись')
    return validated, starts


def validate_schema(plan, data_dir, schema):
    """Проверить все входы одной схемы без predict и расчёта метрик."""
    models = [model for model in plan['models'] if model['schema'] == schema]
    if not models:
        raise ValueError('У плана нет моделей для этой схемы')
    arrays, features, pool_path = check_inputs(models[0], plan, data_dir)
    for model in models:
        if file_hash(model['model']['path']) != model['model']['sha256']:
            raise ValueError('Изменился hash модели')
        if verify_schema(model, plan) != features:
            raise ValueError('Различаются схемы общих физических входов')
    pool = Pool('quantized://' + str(pool_path), thread_count=THREADS)
    arrays, starts = check_pool(pool, arrays, features)
    del pool
    gc.collect()
    return {'completed': True, 'schema': schema, 'rows': len(arrays['label']),
            'groups': len(starts), 'physical_features': len(features),
            'pool_sha256': file_hash(pool_path),
            'targets_sha256': file_hash(Path(data_dir) / 'targets.npz'),
            'schema_targets_sha256': file_hash(Path(data_dir) / schema / 'test_targets.npz'),
            'applied_borders_sha256': file_hash(Path(data_dir) / schema / 'test_applied_borders.tsv'),
            'predict_executed': False, 'metrics_computed': False}


def evaluate_model(model_record, plan, data_dir, output_dir):
    """Загрузить ровно один готовый ранкер и сохранить все test-оценки."""
    started = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    folder = output_dir / 'models' / model_record['id']
    evaluation_path = folder / 'evaluation.json'
    arrays, features, pool_path = check_inputs(model_record, plan, data_dir)
    input_record = {'plan_id': plan['plan_id'], 'model_id': model_record['id'],
                    'model_sha256': model_record['model']['sha256'],
                    'targets_sha256': file_hash(data_dir / 'targets.npz'),
                    'pool_sha256': file_hash(pool_path),
                    'schema': model_record['schema']}
    if evaluation_path.exists():
        previous = json.loads(evaluation_path.read_text())
        if previous['inputs'] != input_record or not previous['completed']:
            raise ValueError('Есть оценка с другими входами')
        for name, saved in previous['artifacts'].items():
            if file_hash(saved['path']) != saved['sha256']:
                raise ValueError('Повреждён сохранённый результат: ' + name)
        return previous
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Незавершённая оценка: сначала зарегистрировать техническую ошибку')
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'request.json', input_record)
    pool = Pool('quantized://' + str(pool_path), thread_count=THREADS)
    arrays, starts = check_pool(pool, arrays, features)
    loaded = time.perf_counter()
    model = CatBoostRanker().load_model(model_record['model']['path'])
    if model.feature_names_ != features or int(model.tree_count_) != model_record['tree_count']:
        raise ValueError('Схема или число деревьев модели отличаются от плана')
    model_loaded = time.perf_counter()
    score = np.asarray(model.predict(pool, thread_count=THREADS), dtype='float64')
    predicted = time.perf_counter()
    if score.shape != arrays['label'].shape or not np.isfinite(score).all():
        raise ValueError('Нужен один конечный score на каждую test-строку')
    del model, pool
    gc.collect()
    metrics_started = time.perf_counter()
    metrics = {name: float(value) for name, value in ranking_metrics(
        arrays['label'], arrays['group_id'], score).items()}
    group_scores = _group_scores(arrays, starts, score, metrics['NDCG@10'])
    metrics_finished = time.perf_counter()
    files = {'score': folder / 'test_score.npy',
             'groups': folder / 'group_ndcg.parquet',
             'daily': folder / 'daily_ndcg.csv',
             'metrics': folder / 'metrics.json'}
    np.save(files['score'], score)
    group_scores.to_parquet(files['groups'], index=False)
    _daily_scores(group_scores).to_csv(files['daily'], index=False)
    write_json(files['metrics'], metrics)
    if file_hash(model_record['model']['path']) != model_record['model']['sha256']:
        raise ValueError('Исходный model.cbm изменился во время оценки')
    record = {'completed': True, 'split': 'test', 'model_id': model_record['id'],
              'family': model_record['family'], 'schema': model_record['schema'],
              'active_features': model_record['active_count'],
              'physical_features': len(features), 'tree_count': model_record['tree_count'],
              'rows': len(score), 'groups': len(group_scores), 'metrics': metrics,
              'inputs': input_record, 'fit_executed': False, 'hyperparameter_search': False,
              'timings': {'verification_and_pool_seconds': loaded - started,
                          'model_load_seconds': model_loaded - loaded,
                          'predict_seconds': predicted - model_loaded,
                          'metrics_seconds': metrics_finished - metrics_started,
                          'elapsed_seconds': time.perf_counter() - started},
              'artifacts': {name: {'path': str(path), 'sha256': file_hash(path)}
                            for name, path in files.items()}}
    write_json(evaluation_path, record)
    return record


def pair_statistics(candidate, reference, indices):
    """Считать разность при общем выборе дней и исходном весе групп."""
    keys = ['group_id', 'day', 'group_size']
    pd.testing.assert_frame_equal(candidate[keys], reference[keys])
    daily = aggregate_days(reference['day'].to_numpy(), reference['ndcg'].to_numpy(),
                           candidate['ndcg'].to_numpy())
    draws = absolute_delta_draws(daily, indices)
    low, high = np.quantile(draws, [.025, .975], method='linear')
    point = float(candidate['ndcg'].mean() - reference['ndcg'].mean())
    return point, float(low), float(high)


def summarize(plan, data_dir, output_dir):
    """Опубликовать все семь моделей и заранее выбранные парные сравнения."""
    started = time.perf_counter()
    output_dir = Path(output_dir)
    arrays = load_targets(Path(data_dir) / 'targets.npz')
    rows, groups = [], {}
    for model in plan['models']:
        folder = output_dir / 'models' / model['id']
        evaluation = json.loads((folder / 'evaluation.json').read_text())
        if not evaluation['completed'] or evaluation['inputs']['model_sha256'] != model['model']['sha256']:
            raise ValueError('Нет готовой оценки зафиксированной модели')
        if evaluation['inputs']['targets_sha256'] != file_hash(Path(data_dir) / 'targets.npz'):
            raise ValueError('Модели оценены на разных targets')
        for saved in evaluation['artifacts'].values():
            if file_hash(saved['path']) != saved['sha256']:
                raise ValueError('Изменился артефакт оценки')
        frame = pd.read_parquet(folder / 'group_ndcg.parquet')
        np.testing.assert_allclose(frame['ndcg'].mean(), evaluation['metrics']['NDCG@10'],
                                   rtol=0, atol=plan['metrics']['group_oracle_atol'])
        groups[model['id']] = frame
        rows.append({'model': model['id'], 'family': model['family'],
                     'active_features': model['active_count'],
                     'NDCG@10': evaluation['metrics']['NDCG@10'],
                     'QueryAUC': evaluation['metrics']['QueryAUC'],
                     'predict_seconds': evaluation['timings']['predict_seconds'],
                     'tree_count': evaluation['tree_count']})
    reference = next(iter(groups.values()))
    for frame in groups.values():
        pd.testing.assert_frame_equal(reference[['group_id', 'day', 'group_size']],
                                      frame[['group_id', 'day', 'group_size']])
    n_days = int(reference['day'].max() - reference['day'].min() + 1)
    settings = plan['bootstrap']
    indices = {block: circular_day_indices(n_days, block, settings['replicates'],
                                           seed=settings['seed'])
               for block in settings['blocks_days']}
    intervals = []
    for comparison in plan['comparisons']:
        for block, draw_indices in indices.items():
            point, low, high = pair_statistics(groups[comparison['candidate']],
                                               groups[comparison['reference']], draw_indices)
            intervals.append({**comparison, 'block_days': block, 'delta_ndcg': point,
                              'ci_low': low, 'ci_high': high,
                              'replicates': settings['replicates'], 'seed': settings['seed'],
                              'confidence': settings['confidence'], 'two_sided': True})
    table = pd.DataFrame(rows)
    for family in ['sub', 'full']:
        table[table['family'] == family].to_csv(output_dir / ('comparison_' + family + '.csv'),
                                               index=False)
    table.to_csv(output_dir / 'comparison.csv', index=False)
    pd.DataFrame(intervals).to_csv(output_dir / 'bootstrap_summary.csv', index=False)
    record = {'completed': True, 'split': 'test', 'plan_id': plan['plan_id'],
              'rows': len(arrays['label']), 'groups': len(reference),
              'day_first': int(reference['day'].min()), 'day_last': int(reference['day'].max()),
              'calendar_days': n_days, 'models': rows, 'comparisons': intervals,
              'fit_executed': False, 'hyperparameter_search': False,
              'bootstrap_and_summary_seconds': time.perf_counter() - started,
              'limitations': plan['limitations']}
    write_json(output_dir / 'summary.json', record)
    lines = ['# Оценка семи ранкеров с восстановленным start_ts', '',
             'Сохранённые модели оценены без переобучения.',
             f"Строк: {record['rows']}; групп: {record['groups']}; "
             f"дни с нуля: {record['day_first']}–{record['day_last']}.", '']
    for family, title in [('sub', 'Общий train-срез'), ('full', 'Полный train')]:
        lines += ['## ' + title, '', '| Модель | NDCG@10 | QueryAUC | Predict, с |',
                  '|---|---:|---:|---:|']
        for row in rows:
            if row['family'] == family:
                lines.append(f"| {row['model']} | {row['NDCG@10']:.6f} | "
                             f"{row['QueryAUC']:.6f} | {row['predict_seconds']:.2f} |")
        lines.append('')
    lines += ['## Парные интервалы', '',
              '| Разность | Блок, дней | Δ NDCG | 95% интервал |',
              '|---|---:|---:|---:|']
    for row in intervals:
        lines.append(f"| {row['candidate']} − {row['reference']} | {row['block_days']} | "
                     f"{row['delta_ndcg']:+.6f} | [{row['ci_low']:+.6f}; {row['ci_high']:+.6f}] |")
    lines += ['', 'Парный циклический bootstrap по дням: 20 000 реплик, seed 42; '
              'сохранён исходный вес групп, без поправки на множественные сравнения.',
              'Восстановленный start_ts содержит обнаруженный канал утечки; '
              'это сравнение относится к прежнему протоколу.']
    (output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return record
