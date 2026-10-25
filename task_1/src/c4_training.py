"""Новая очередь C4: train-only пороги, три ранкера и парные val-интервалы.

Матрицы уже подготовлены на одинаковых строках и cutoff. Здесь нет чтения test,
отбора признаков или подбора параметров. Два часа — общий верхний предел для
трёх последовательных worker, включая загрузку Pool и сохранение оценки.
"""
from pathlib import Path
import argparse
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from block_bootstrap import aggregate_days, circular_day_indices
from compare_temporal_controls_pairs import absolute_delta_draws
from temporal_column_quantization import prepare_quantized_split
from temporal_full_training import _sha256, _write_json, fit_ranker

VARIANTS = ('M-t', 'C2b-t', 'D462-t')
SCHEMAS = {'M-t': 'manual49', 'C2b-t': 'manual49', 'D462-t': 'dfs462'}
COUNTS = {'manual49': 49, 'dfs462': 462}
TARGET_KEYS = ('label', 'group_id', 'log_id', 'day')
PAIRS = (('D462-t', 'M-t'), ('C2b-t', 'M-t'), ('C2b-t', 'D462-t'))
PARAMETERS = {
    'iterations': 1600, 'early_stopping_rounds': 80, 'random_seed': 42,
    'use_best_model': True, 'verbose': 100, 'allow_writing_files': False,
    'nan_mode': 'Min', 'border_count': 254, 'task_type': 'CPU',
    'random_strength': 1.0, 'rsm': 1.0, 'loss_function': 'YetiRank',
    'eval_metric': 'NDCG:top=10;type=Base;denominator=LogPosition',
    'thread_count': 4, 'bootstrap_type': 'MVS', 'subsample': 0.8,
    'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 1e-20,
}
FORBIDDEN = {'label', 'uid', 'user_id', 'item_id', 'group_id', 'log_id',
             'day', 'timestamp', 'start_ts', 'played_ratio_pct'}


def read_json(path):
    return json.loads(Path(path).read_text())


def load_targets(path):
    """Проверить общие массивы, порядок групп и день нового cutoff."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in TARGET_KEYS}
    if any(value.ndim != 1 or value.dtype.kind not in 'biuf'
           or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Targets должны быть конечными одномерными массивами')
    if len({len(value) for value in arrays.values()}) != 1 or not len(arrays['label']):
        raise ValueError('Targets должны иметь одинаковую ненулевую длину')
    for name in ('group_id', 'log_id', 'day'):
        if arrays[name].dtype.kind not in 'iu':
            raise ValueError(name + ' должен быть целочисленным')
    groups, labels = arrays['group_id'], arrays['label']
    if not np.isin(labels, [0, 1, 2]).all() or np.any(groups[1:] < groups[:-1]):
        raise ValueError('Нужны метки 0/1/2 и группы по возрастанию')
    if len(np.unique(arrays['log_id'])) != len(labels):
        raise ValueError('Повторный log_id в targets')
    np.testing.assert_array_equal(arrays['day'], groups % 1000)
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    if np.any(np.minimum.reduceat(labels, starts) == np.maximum.reduceat(labels, starts)):
        raise ValueError('Постоянные по label группы нужно удалить до обучения')
    return arrays


def feature_names(data_dir, schema):
    """Разрешить только две заранее заданные числовые схемы."""
    metadata = read_json(Path(data_dir) / schema / 'features.json')
    if isinstance(metadata, dict):
        if metadata.get('feature_count') != COUNTS[schema] or metadata.get('dtype') != 'float32' or metadata.get('layout') != 'F-order':
            raise ValueError('Метаданные признаков не соответствуют схеме ' + schema)
        names = metadata.get('features')
    else:
        names = metadata
    if not isinstance(names, list) or len(names) != COUNTS[schema]:
        raise ValueError('Неверное число признаков ' + schema)
    if any(not isinstance(name, str) or not name or any(c in name for c in '\t\r\n')
           for name in names) or len(set(names)) != len(names):
        raise ValueError('Имена признаков должны быть уникальными строками')
    if FORBIDDEN.intersection(names):
        raise ValueError('Сырые идентификаторы, время или исход попали в X')
    return names


def variant_parameters(variant):
    """M-t использует первые 42 колонки; два других варианта — все свои колонки."""
    if variant not in VARIANTS:
        raise ValueError('Неизвестный вариант C4')
    parameters = dict(PARAMETERS)
    if variant == 'M-t':
        parameters['ignored_features'] = list(range(42, 49))
    return parameters


def _manifest(data_dir):
    """Зафиксировать реальные новые входы до обучения, включая hashes матриц."""
    data_dir = Path(data_dir).resolve()
    targets = {split: load_targets(data_dir / (split + '_targets.npz'))
               for split in ('train', 'val')}
    if np.intersect1d(targets['train']['log_id'], targets['val']['log_id']).size:
        raise ValueError('Train и validation имеют общие target log_id')
    schemas = {schema: feature_names(data_dir, schema) for schema in COUNTS}
    inputs = {}
    for split in ('train', 'val'):
        path = data_dir / (split + '_targets.npz')
        inputs[split + '_targets'] = {'path': str(path), 'sha256': _sha256(path),
            'rows': len(targets[split]['label']),
            'groups': int(len(np.unique(targets[split]['group_id']))),
            'day_first': int(targets[split]['day'].min()),
            'day_last': int(targets[split]['day'].max())}
        for schema, names in schemas.items():
            path = data_dir / schema / (split + '_X.npy')
            matrix = np.load(path, mmap_mode='r', allow_pickle=False)
            if matrix.dtype != np.dtype('float32') or not matrix.flags.f_contiguous:
                raise ValueError('Требуется F-order float32: ' + str(path))
            if matrix.shape != (len(targets[split]['label']), len(names)):
                raise ValueError('Матрица не соответствует общим targets')
            del matrix
            inputs[schema + '_' + split] = {'path': str(path), 'sha256': _sha256(path)}
    return {'protocol': 'C4_packet_cutoff_v1', 'data_dir': str(data_dir),
            'schemas': schemas, 'inputs': inputs,
            'parameters': {variant: variant_parameters(variant) for variant in VARIANTS},
            'fit_order': list(VARIANTS), 'test_used': False, 'pilot_refit': False,
            'quantization_origin': 'new common selected C4 train only',
            'metrics': {'ndcg': 'linear NDCG@10', 'query_auc': 'QueryAUC:type=Ranking',
                        'tie_seed': 42, 'ties': 'exact score ties only'},
            'bootstrap': {'blocks_days': [3, 4], 'replicates': 20000, 'seed': 42,
                          'confidence': .95, 'two_sided': True,
                          'calendar': 'min..max cutoff day, including internal empty days',
                          'group_weighted': True, 'multiplicity_correction': False}}


def freeze_plan(data_dir, result_dir):
    """При повторном вызове проверять все входы вместо изменения плана."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / 'training_plan.json'
    current = _manifest(data_dir)
    if path.exists():
        if read_json(path) != current:
            raise ValueError('Зафиксированные входы или параметры C4 изменились')
    else:
        _write_json(path, current)
    return current


def prepare_pools(data_dir, result_dir, progress=None):
    """Заново получить все train-пороги и применить их к val по одному столбцу."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    plan = freeze_plan(data_dir, result_dir)
    reports = {}
    for schema in COUNTS:
        folder = result_dir / 'quantized' / schema
        reports[schema] = {}
        for split in ('train', 'val'):
            borders = None if split == 'train' else folder / 'train_borders.tsv'
            report = prepare_quantized_split(
                data_dir / schema / (split + '_X.npy'),
                data_dir / (split + '_targets.npz'), plan['schemas'][schema],
                folder, split, input_borders=borders, progress=progress,
                matrix_sha256=plan['inputs'][schema + '_' + split]['sha256'])
            # Универсальный helper пишет full_train. Здесь train — новая общая
            # подвыборка C4; эти отчёты принадлежат только новому запуску.
            report['quantization']['borders_origin'] = 'c4_common_selected_train'
            _write_json(folder / (split + '_quantization.json'), report)
            actual = load_targets(report['targets_path'])
            canonical = load_targets(data_dir / (split + '_targets.npz'))
            for key in TARGET_KEYS:
                np.testing.assert_array_equal(actual[key], canonical[key])
            reports[schema][split] = report
            gc.collect()
    _write_json(result_dir / 'pools.json', {'completed': True, 'test_used': False,
                                          'schemas': reports})
    return reports


def fit_variant(data_dir, result_dir, variant):
    """Worker обучает ровно одну модель из заранее сохранённого плана."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    plan = read_json(result_dir / 'training_plan.json')
    schema = SCHEMAS[variant]
    folder = result_dir / 'quantized' / schema
    output = result_dir / 'models' / variant
    parameters = plan['parameters'][variant]
    _write_json(result_dir / ('requested_params_' + variant + '.json'), parameters)
    # Используем canonical val targets: день в нём задан новым cutoff.
    fitted = fit_ranker(folder / 'train.qbin', folder / 'val.qbin',
                        data_dir / 'val_targets.npz', plan['schemas'][schema],
                        output, parameters)
    ignored = parameters.get('ignored_features', [])
    importance = pd.read_csv(output / 'feature_importance.csv').set_index('Feature Id')
    if ignored and (importance.loc[[plan['schemas'][schema][i] for i in ignored],
                                  'Importances'] != 0).any():
        raise AssertionError('Исключённые признаки получили ненулевую важность')
    record = {**fitted, 'variant': variant, 'schema': schema,
              'physical_features': len(plan['schemas'][schema]),
              'active_features': len(plan['schemas'][schema]) - len(ignored),
              'ignored_indices': ignored, 'training_plan_sha256':
              _sha256(result_dir / 'training_plan.json'), 'test_used': False}
    _write_json(output / 'control.json', record)
    return record


def _stop_worker(process):
    """Остановить только собственную process group этой очереди."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_fit_worker(data_dir, result_dir, variant, remaining_seconds, progress=None):
    """Hard timeout включает чтение Pool, fit, predict и сохранение модели."""
    if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
        raise TimeoutError('Исчерпан общий двухчасовой бюджет обучения')
    result_dir = Path(result_dir)
    (result_dir / 'logs').mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'fit',
               '--data-dir', str(data_dir), '--result-dir', str(result_dir),
               '--variant', variant]
    started = time.monotonic()
    log_path = result_dir / 'logs' / ('fit_' + variant + '.log')
    with log_path.open('x', buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        next_update = started
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= remaining_seconds:
                    raise TimeoutError('Общий бюджет обучения исчерпан на ' + variant)
                if time.monotonic() >= next_update:
                    event = {'stage': 'fit', 'variant': variant,
                             'worker_seconds': elapsed,
                             'remaining_training_seconds': max(0., remaining_seconds - elapsed)}
                    _write_json(result_dir / 'training_status.json', event)
                    if progress is not None:
                        progress(event)
                    next_update = time.monotonic() + 10
                time.sleep(min(.5, max(.01, remaining_seconds - elapsed)))
            if time.monotonic() - started > remaining_seconds:
                raise TimeoutError('Общий бюджет обучения исчерпан на ' + variant)
            if process.returncode:
                raise RuntimeError('C4 worker завершился с кодом ' + str(process.returncode)
                                   + '; лог: ' + str(log_path))
        finally:
            _stop_worker(process)
    return time.monotonic() - started


def summarize(result_dir):
    """Три прямых сравнения на одинаковых группах, без сравнения со старой задачей."""
    result_dir = Path(result_dir)
    plan = read_json(result_dir / 'training_plan.json')
    settings = plan['bootstrap']
    groups, rows = {}, []
    for variant in VARIANTS:
        folder = result_dir / 'models' / variant
        control = read_json(folder / 'control.json')
        if not control['completed'] or control['training_plan_sha256'] != _sha256(
                result_dir / 'training_plan.json'):
            raise ValueError('Модель не соответствует общему плану')
        if _sha256(folder / 'model.cbm') != control['model_sha256']:
            raise ValueError('Сохранённая модель изменилась')
        frame = pd.read_parquet(folder / 'group_ndcg.parquet')
        if not frame['group_id'].is_unique or not frame['group_id'].is_monotonic_increasing:
            raise ValueError('Некорректные группы сохранённой оценки')
        metrics = control['evaluation']['metrics']
        np.testing.assert_allclose(frame['ndcg'].mean(), metrics['NDCG@10'], rtol=0, atol=1e-12)
        groups[variant] = frame
        rows.append({'variant': variant, 'active_features': control['active_features'],
                     'physical_features': control['physical_features'],
                     'NDCG@10': metrics['NDCG@10'], 'QueryAUC': metrics['QueryAUC'],
                     'fit_seconds': control['fit_seconds'],
                     'tree_count': control['evaluation']['tree_count'],
                     'rows': int(frame['group_size'].sum()),
                     'groups': len(frame), 'test_used': False})
    reference = groups['M-t']
    for frame in groups.values():
        pd.testing.assert_frame_equal(frame[['group_id', 'day', 'group_size']],
                                      reference[['group_id', 'day', 'group_size']])
    n_days = int(reference['day'].max() - reference['day'].min() + 1)
    indices = {block: circular_day_indices(n_days, block, settings['replicates'],
                                           seed=settings['seed'])
               for block in settings['blocks_days']}
    intervals = []
    for candidate, baseline in PAIRS:
        daily = aggregate_days(reference['day'].to_numpy(),
                               groups[baseline]['ndcg'].to_numpy(),
                               groups[candidate]['ndcg'].to_numpy())
        daily.to_csv(result_dir / ('daily_' + candidate + '_minus_' + baseline + '.csv'), index=False)
        point = float(groups[candidate]['ndcg'].mean() - groups[baseline]['ndcg'].mean())
        for block, selected in indices.items():
            low, high = np.quantile(absolute_delta_draws(daily, selected), [.025, .975], method='linear')
            intervals.append({'candidate': candidate, 'reference': baseline,
                              'delta_ndcg': point, 'ci_low': float(low), 'ci_high': float(high),
                              'block_days': block, 'replicates': settings['replicates'],
                              'seed': settings['seed'], 'confidence': .95, 'two_sided': True,
                              'validation_reused': True, 'multiplicity_correction': False})
    table = pd.DataFrame(rows)
    table['delta_ndcg_vs_M_t'] = table['NDCG@10'] - rows[0]['NDCG@10']
    table.to_csv(result_dir / 'comparison.csv', index=False)
    pd.DataFrame(intervals).to_csv(result_dir / 'bootstrap_summary.csv', index=False)
    record = {'completed': True, 'models': table.to_dict(orient='records'),
              'comparisons': intervals, 'test_used': False, 'validation_reused': True,
              'calendar_days': n_days,
              'limitations': ['fixed fitted models and previously selected DFS formulas',
                              'validation reused; marginal intervals without multiplicity correction',
                              'observed event user-day proxy; timestamp availability assumption',
                              'different C4 task and rows from original start_ts comparison']}
    _write_json(result_dir / 'training_result.json', record)
    return record


def run_training(data_dir, result_dir, max_training_seconds=7200, progress=None):
    """Подготовить пулы и выполнить три fit с одним общим жёстким лимитом."""
    if (not math.isfinite(max_training_seconds) or max_training_seconds <= 0
            or max_training_seconds > 7200):
        raise ValueError('Общий бюджет обучения должен быть в (0, 7200] секунд')
    data_dir, result_dir = Path(data_dir).resolve(), Path(result_dir).resolve()
    prepare_pools(data_dir, result_dir, progress=progress)
    budget_path = result_dir / 'training_budget.json'
    if budget_path.exists():
        raise FileExistsError('Бюджет уже начат: автоматический перезапуск C4 запрещён')
    budget = {'max_training_seconds': float(max_training_seconds),
              'started_unix': time.time(), 'consumed_worker_seconds': 0.,
              'completed_variants': [], 'completed': False, 'test_used': False,
              'scope': 'all three sequential workers, including Pool loading and evaluation'}
    _write_json(budget_path, budget)
    started = time.monotonic()
    try:
        for variant in VARIANTS:
            remaining = max_training_seconds - (time.monotonic() - started)
            elapsed = run_fit_worker(data_dir, result_dir, variant, remaining, progress=progress)
            budget['consumed_worker_seconds'] += elapsed
            budget['completed_variants'].append(variant)
            _write_json(budget_path, budget)
        budget['completed'] = True
        budget['training_stage_seconds'] = time.monotonic() - started
        _write_json(budget_path, budget)
        record = summarize(result_dir)
        _write_json(result_dir / 'training_status.json', {'completed': True, 'stage': 'completed',
                    'completed_variants': list(VARIANTS), 'test_used': False})
        return record
    except BaseException as error:
        budget['training_stage_seconds'] = time.monotonic() - started
        budget['error'] = str(error)
        _write_json(budget_path, budget)
        _write_json(result_dir / 'training_stopped.json', {
            'completed': False, 'reason': str(error),
            'completed_variants': budget['completed_variants'],
            'training_stage_seconds': budget['training_stage_seconds'], 'test_used': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'train'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--max-training-seconds', type=float, default=7200)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare_pools(args.data_dir, args.result_dir)
    elif args.stage == 'fit':
        if args.variant is None:
            parser.error('--stage fit требует --variant')
        result = fit_variant(args.data_dir, args.result_dir, args.variant)
    else:
        result = run_training(args.data_dir, args.result_dir, args.max_training_seconds)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
