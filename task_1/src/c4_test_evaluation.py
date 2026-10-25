"""C4 test: фиксированные train-пороги, predict и три парных сравнения.

Здесь нет обучения, подбора признаков или новых границ квантования. Это
повторная оценка прежнего test после исправления времени запроса C4.
"""
from pathlib import Path
import gc
import json
import time

import catboost
from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

from block_bootstrap import aggregate_days, circular_day_indices
from compare_temporal_controls_pairs import absolute_delta_draws
from final_test_evaluation import check_pool, file_hash, write_json
from optuna_evaluation import _group_scores, _daily_scores
from temporal_column_quantization import (
    SUPPORTED_CATBOOST, _border_columns, _column_record, _merge_native_columns, _release,
)
from c4_training import load_targets
from train_dfs_catboost import ranking_metrics

VARIANTS = ('M-t', 'C2b-t', 'D462-t')
TARGET_KEYS = ('label', 'group_id', 'log_id', 'day')
PAIRS = (('D462-t', 'M-t'), ('C2b-t', 'M-t'), ('C2b-t', 'D462-t'))
THREADS = 4
FORBIDDEN = {'label', 'uid', 'user_id', 'item_id', 'group_id', 'log_id',
             'day', 'timestamp', 'start_ts', 'played_ratio_pct', 'end_ts'}


def _authorized(plan):
    """Остановиться, если план допускает обучение или не разрешает test."""
    if plan.get('test_evaluation_authorized') is not True:
        raise ValueError('Нет разрешённого frozen-плана test')
    if plan.get('new_training_allowed') is not False:
        raise ValueError('C4 test допускает только predict')
    variants = [record['variant'] for record in plan['models']]
    if variants != list(VARIANTS):
        raise ValueError('Нужны ровно три модели C4 в фиксированном порядке')
    actual_pairs = [tuple(pair) for pair in plan['comparisons']]
    if actual_pairs != list(PAIRS):
        raise ValueError('Изменились заранее выбранные пары')
    settings = plan['bootstrap']
    if (settings['block_days'] != [3, 4] or settings['replicates'] != 20000
            or settings['seed'] != 42 or settings['confidence'] != .95):
        raise ValueError('Изменились параметры парного bootstrap')
    if set(plan['schemas']) != {'manual49', 'dfs462'}:
        raise ValueError('Нужны две физические схемы C4')
    for schema, names in plan['schemas'].items():
        if not names or len(names) != len(set(names)) or FORBIDDEN.intersection(names):
            raise ValueError('Некорректные или запрещённые входы ' + schema)
    for model in plan['models']:
        schema = 'dfs462' if model['variant'] == 'D462-t' else 'manual49'
        if model['schema'] != schema:
            raise ValueError('Модель получила другую схему')
        expected = len(plan['schemas'][schema]) - (7 if model['variant'] == 'M-t' else 0)
        if model['active_features'] != expected:
            raise ValueError('Неверное число активных признаков')


def _artifact(record):
    path = Path(record['path'])
    if file_hash(path) != record['sha256']:
        raise ValueError('Изменился зафиксированный артефакт ' + str(path))
    return path


def _verify_saved(report):
    """Проверить сохранённые файлы перед повторным чтением checkpoint."""
    for record in report['artifacts'].values():
        _artifact(record)


def _quantize_schema(data_dir, result_dir, plan, schema, progress=None):
    """Применить пороги C4 train по одной колонке, не создавая train/val."""
    if catboost.__version__ != SUPPORTED_CATBOOST:
        raise RuntimeError('Нужна проверенная версия CatBoost ' + SUPPORTED_CATBOOST)
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    folder = result_dir / 'quantized' / schema
    columns_dir = folder / 'test_columns'
    columns_dir.mkdir(parents=True, exist_ok=True)
    names = plan['schemas'][schema]
    targets_path = data_dir / 'targets.npz'
    matrix_path = data_dir / schema / 'test_X.npy'
    borders_path = _artifact(plan['train_borders'][schema])
    targets = load_targets(targets_path)
    request = {'split': 'test', 'schema': schema, 'features': names,
               'rows': len(targets['label']), 'matrix_sha256': file_hash(matrix_path),
               'targets_sha256': file_hash(targets_path),
               'train_borders_sha256': file_hash(borders_path),
               'method': 'c4_frozen_native_columns_test_v1', 'fit_executed': False}
    request_path = folder / 'test_request.json'
    report_path = folder / 'test_quantization.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Изменились входы test quantization')
    else:
        write_json(request_path, request)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if not report['completed'] or report['request'] != request:
            raise ValueError('Неверный quantization checkpoint')
        _verify_saved(report)
        return report
    matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
    if (matrix.dtype != np.dtype('float32') or not matrix.flags.f_contiguous
            or matrix.shape != (len(targets['label']), len(names))):
        raise ValueError('Нужна F-order float32 test-матрица в canonical-порядке')
    started = time.perf_counter()
    inputs = _border_columns(borders_path, len(names))
    paths, records = [], []
    for index, name in enumerate(names):
        column_paths, record = _column_record(
            matrix, index, name, targets, columns_dir, inputs[index])
        paths.append(column_paths['pool'])
        records.append(record)
        if progress:
            progress({'stage': 'quantize_test', 'schema': schema,
                      'feature_index': index, 'features_completed': index + 1,
                      'features_total': len(names)})
    del matrix
    _release()
    pool_path = folder / 'test.qbin'
    partial = folder / 'test.qbin.partial'
    _merge_native_columns(paths, names, len(targets['label']), partial)
    pool = Pool('quantized://' + str(partial), thread_count=1)
    check_pool(pool, targets, names)
    applied = folder / 'test_applied_borders.tsv'
    pool.save_quantization_borders(str(applied))
    if applied.read_bytes() != borders_path.read_bytes():
        raise AssertionError('Test изменил fixed C4 train-пороги')
    del pool
    _release()
    partial.replace(pool_path)
    saved_targets = folder / 'test_targets.npz'
    np.savez_compressed(saved_targets, **targets)
    artifacts = {'pool': pool_path, 'targets': saved_targets, 'applied_borders': applied}
    report = {'completed': True, 'split': 'test', 'test_used': True,
              'schema': schema, 'request': request,
              'rows': len(targets['label']), 'features': names,
              'training_constant_columns': [r['feature_index'] for r in records
                                            if r['training_constant_ignored']],
              'quantize_seconds': sum(r['seconds'] for r in records),
              'elapsed_seconds': time.perf_counter() - started,
              'fit_executed': False, 'new_borders_learned': False,
              'artifacts': {name: {'path': str(path), 'sha256': file_hash(path)}
                            for name, path in artifacts.items()}}
    write_json(report_path, report)
    for path in paths:
        path.unlink()
    return report


def quantize_test(data_dir, result_dir, plan, progress=None):
    """Создать только два test-пула с порогами общей C4 train-выборки."""
    _authorized(plan)
    reports = {}
    for schema in plan['schemas']:
        reports[schema] = _quantize_schema(data_dir, result_dir, plan, schema, progress)
    record = {'completed': True, 'split': 'test', 'schemas': reports,
              'fit_executed': False, 'new_borders_learned': False}
    write_json(Path(result_dir) / 'test_quantization.json', record)
    return record


def _evaluation_inputs(data_dir, result_dir, plan, model_record):
    """Сверить одну модель с общими строками, схемой и frozen train-порогами."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    schema = model_record['schema']
    folder = result_dir / 'quantized' / schema
    model_path = _artifact(model_record['model'])
    border = _artifact(plan['train_borders'][schema])
    if (folder / 'test_applied_borders.tsv').read_bytes() != border.read_bytes():
        raise ValueError('Test получил другие пороги')
    arrays = load_targets(data_dir / 'targets.npz')
    schema_arrays = load_targets(folder / 'test_targets.npz')
    for key in TARGET_KEYS:
        np.testing.assert_array_equal(schema_arrays[key], arrays[key])
    quantized = json.loads((folder / 'test_quantization.json').read_text())
    _verify_saved(quantized)
    if quantized['request']['targets_sha256'] != file_hash(data_dir / 'targets.npz'):
        raise ValueError('Пул относится к другим targets')
    names = plan['schemas'][schema]
    return arrays, names, folder / 'test.qbin', model_path


def _evaluate_variant(data_dir, result_dir, plan, model_record, progress=None):
    """Загрузить готовую модель и выполнить единственный predict на общем test."""
    started = time.perf_counter()
    variant = model_record['variant']
    folder = Path(result_dir) / 'models' / variant
    arrays, names, pool_path, model_path = _evaluation_inputs(
        data_dir, result_dir, plan, model_record)
    inputs = {'model': model_record['model'], 'schema': model_record['schema'],
              'targets_sha256': file_hash(Path(data_dir) / 'targets.npz'),
              'pool_sha256': file_hash(pool_path),
              'train_borders': plan['train_borders'][model_record['schema']]}
    evaluation_path = folder / 'evaluation.json'
    if evaluation_path.exists():
        saved = json.loads(evaluation_path.read_text())
        if not saved['completed'] or saved['inputs'] != inputs:
            raise ValueError('Сохранённая оценка имеет другие входы')
        _verify_saved(saved)
        return saved
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Незавершённая оценка требует разбора: ' + variant)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'request.json', inputs)
    if progress:
        progress({'stage': 'predict_test', 'variant': variant})
    pool = Pool('quantized://' + str(pool_path), thread_count=THREADS)
    arrays, starts = check_pool(pool, arrays, names)
    model = CatBoostRanker().load_model(str(model_path))
    if model.feature_names_ != names or int(model.tree_count_) != model_record['tree_count']:
        raise ValueError('Модель изменила схему или число деревьев')
    expected_ignored = list(range(len(names) - 7, len(names))) if variant == 'M-t' else []
    if model.get_all_params().get('ignored_features', []) != expected_ignored:
        raise ValueError('Маска модели отличается от frozen C4')
    loaded = time.perf_counter()
    score = np.asarray(model.predict(pool, thread_count=THREADS), dtype='float64')
    predicted = time.perf_counter()
    if score.shape != arrays['label'].shape or not np.isfinite(score).all():
        raise ValueError('Нужен один конечный score на test-строку')
    del model, pool
    gc.collect()
    metrics_started = time.perf_counter()
    metrics = {key: float(value) for key, value in ranking_metrics(
        arrays['label'], arrays['group_id'], score).items()}
    groups = _group_scores(arrays, starts, score, metrics['NDCG@10'])
    files = {'score': folder / 'test_score.npy', 'groups': folder / 'group_ndcg.parquet',
             'daily': folder / 'daily_ndcg.csv', 'metrics': folder / 'metrics.json'}
    np.save(files['score'], score)
    groups.to_parquet(files['groups'], index=False)
    _daily_scores(groups).to_csv(files['daily'], index=False)
    write_json(files['metrics'], metrics)
    _artifact(model_record['model'])
    record = {'completed': True, 'split': 'test', 'test_used': True, 'variant': variant,
              'rows': len(score), 'groups': len(groups), 'metrics': metrics,
              'tree_count': model_record['tree_count'],
              'active_features': model_record['active_features'], 'physical_features': len(names),
              'inputs': inputs, 'fit_executed': False, 'hyperparameter_search': False,
              'timings': {'verification_and_load_seconds': loaded - started,
                          'predict_seconds': predicted - loaded,
                          'metrics_and_save_seconds': time.perf_counter() - metrics_started,
                          'elapsed_seconds': time.perf_counter() - started},
              'artifacts': {name: {'path': str(path), 'sha256': file_hash(path)}
                            for name, path in files.items()}}
    write_json(evaluation_path, record)
    return record


def summarize_test(result_dir, plan, evaluations):
    """Сохранить три точечных оценки и шесть marginal парных интервалов."""
    result_dir = Path(result_dir)
    groups, rows = {}, []
    for model in plan['models']:
        variant = model['variant']
        evaluation = evaluations[variant]
        _verify_saved(evaluation)
        frame = pd.read_parquet(evaluation['artifacts']['groups']['path'])
        np.testing.assert_allclose(frame['ndcg'].mean(), evaluation['metrics']['NDCG@10'],
                                   rtol=0, atol=1e-12)
        groups[variant] = frame
        rows.append({'variant': variant, 'active_features': model['active_features'],
                     'physical_features': len(plan['schemas'][model['schema']]),
                     'NDCG@10': evaluation['metrics']['NDCG@10'],
                     'QueryAUC': evaluation['metrics']['QueryAUC'],
                     'tree_count': model['tree_count'],
                     'predict_seconds': evaluation['timings']['predict_seconds'],
                     'rows': evaluation['rows'], 'groups': evaluation['groups']})
    reference = groups['M-t']
    keys = ['group_id', 'day', 'group_size']
    for frame in groups.values():
        pd.testing.assert_frame_equal(reference[keys], frame[keys])
    n_days = int(reference['day'].max() - reference['day'].min() + 1)
    settings = plan['bootstrap']
    draws = {block: circular_day_indices(n_days, block, settings['replicates'],
                                         seed=settings['seed'])
             for block in settings['block_days']}
    intervals = []
    for candidate, baseline in plan['comparisons']:
        daily = aggregate_days(reference['day'].to_numpy(), groups[baseline]['ndcg'].to_numpy(),
                               groups[candidate]['ndcg'].to_numpy())
        daily.to_csv(result_dir / ('daily_' + candidate + '_minus_' + baseline + '.csv'), index=False)
        point = float(groups[candidate]['ndcg'].mean() - groups[baseline]['ndcg'].mean())
        for block, selected in draws.items():
            low, high = np.quantile(absolute_delta_draws(daily, selected), [.025, .975], method='linear')
            intervals.append({'candidate': candidate, 'reference': baseline,
                              'delta_ndcg': point, 'ci_low': float(low), 'ci_high': float(high),
                              'block_days': block, 'replicates': settings['replicates'],
                              'seed': settings['seed'], 'confidence': .95, 'two_sided': True,
                              'multiplicity_correction': False, 'test_already_open': True})
    table = pd.DataFrame(rows)
    table['delta_ndcg_vs_M_t'] = table['NDCG@10'] - rows[0]['NDCG@10']
    table['relative_ndcg_vs_M_t'] = table['NDCG@10'] / rows[0]['NDCG@10'] - 1
    table.to_csv(result_dir / 'comparison.csv', index=False)
    pd.DataFrame(intervals).to_csv(result_dir / 'bootstrap_summary.csv', index=False)
    record = {'completed': True, 'split': 'test', 'test_used': True,
              'test_already_open': True, 'new_blind_test': False,
              'fit_executed': False, 'hyperparameter_search': False,
              'new_borders_learned': False, 'rows': int(reference['group_size'].sum()),
              'groups': len(reference), 'day_first': int(reference['day'].min()),
              'day_last': int(reference['day'].max()), 'calendar_days': n_days,
              'models': table.to_dict(orient='records'), 'comparisons': intervals,
              'limitations': plan.get('limitations', [])}
    write_json(result_dir / 'test_result.json', record)
    return record


def evaluate_test(data_dir, result_dir, plan, progress=None):
    """Оценить ровно три frozen C4 модели, не выбирая между ними по test."""
    _authorized(plan)
    evaluations = {}
    for model in plan['models']:
        evaluations[model['variant']] = _evaluate_variant(
            data_dir, result_dir, plan, model, progress)
    for model in plan['models']:
        _artifact(model['model'])
    return summarize_test(result_dir, plan, evaluations)
