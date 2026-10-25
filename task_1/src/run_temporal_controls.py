"""Последовательные контроли DFS462: общий train-срез, полный val, внешний контроль RAM."""
from pathlib import Path
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

import temporal_controls_data
from replay_paths import configure_legacy

from temporal_controls_data import (
    ROOT, BASE_BORDERS, VARIANTS, initialize, run_paths, read_json, write_json,
    prepare_selection, prepare_dfs, prepare_manual_sessions, variant_spec)
from temporal_full_training import fit_ranker
from controls_quantization import prepare_augmented_pool
from block_bootstrap import aggregate_days, circular_day_indices


def fit_variant(name, variant, pilot=False):
    """Одинаковые параметры; различается только маска входных признаков."""
    result, data = run_paths(name)
    protocol = read_json(result / 'protocol.json')
    spec = variant_spec(name, variant)
    parameters = dict(protocol['parameters'])
    parameters['iterations'] = 200 if pilot else 1600
    parameters['verbose'] = 25 if pilot else 100
    if spec['ignored_indices']:
        parameters['ignored_features'] = spec['ignored_indices']
    destination = result / ('pilot' if pilot else 'models') / variant
    fitted = fit_ranker(spec['train_pool'], spec['val_pool'],
                        data / 'selection/val.npz', spec['features'], destination, parameters)
    importance = pd.read_csv(destination / 'feature_importance.csv')
    if spec['ignored_indices']:
        inactive_names = [spec['features'][i] for i in spec['ignored_indices']]
        inactive = importance.set_index('Feature Id').loc[inactive_names, 'Importances']
        if (inactive != 0).any():
            raise AssertionError('Исключенный признак получил ненулевую важность')
    record = {**fitted, 'variant': variant, 'pilot': pilot, 'pool_feature_count': len(spec['features']),
              'active_feature_count': spec['active_count'], 'ignored_indices': spec['ignored_indices'],
              'active_features': spec['active_features'], 'train_selection': str(data / 'selection/train.npz')}
    write_json(destination / 'control.json', record)
    if pilot:
        curve = read_json(destination / 'learning_curve.json')['validation']
        iterations = len(next(iter(curve.values())))
        full_seconds = fitted['fit_seconds'] / iterations * 1600
        write_json(result / 'pilot_assessment.json', {
            'completed': True, 'iterations_measured': iterations, 'fit_seconds': fitted['fit_seconds'],
            'dfs_1600_seconds_linear_estimate': full_seconds,
            'five_fits_hours_rough_range': [full_seconds * 3.3 / 3600, full_seconds * 5 / 3600],
            'limitation': 'Пилот не измеряет все варианты; время поздних итераций может отличаться',
            'metrics_are_pilot_only': True, 'test_used': False})
    return record


def paired_delta_interval(daily, indices):
    """Разница NDCG с исходным весом групп, не среднее дневных средних."""
    count = daily['group_count'].to_numpy()[indices].sum(axis=1)
    delta = ((daily['dfs_sum'].to_numpy()[indices].sum(axis=1)
              - daily['base_sum'].to_numpy()[indices].sum(axis=1)) / count)
    return np.quantile(delta, [.025, .975]).tolist()


def compare_variants(name):
    """Сопоставить только одинаковые val-группы; общий парный bootstrap для всех пар."""
    result, _ = run_paths(name)
    baseline = pd.read_parquet(result / 'models/M-sub/group_ndcg.parquet')
    reference = baseline[['group_id', 'day', 'group_size']]
    indices = {block: circular_day_indices(15, block, 20_000, seed=42) for block in [3, 4]}
    rows, intervals = [], []
    for variant in VARIANTS:
        folder = result / 'models' / variant
        groups = pd.read_parquet(folder / 'group_ndcg.parquet')
        pd.testing.assert_frame_equal(reference, groups[['group_id', 'day', 'group_size']])
        control = read_json(folder / 'control.json')
        metrics = control['evaluation']['metrics']
        daily = aggregate_days(reference['day'].to_numpy(),
                               baseline['ndcg'].to_numpy(), groups['ndcg'].to_numpy())
        if len(daily) != 15:
            raise AssertionError('Ожидались все 15 дней validation')
        daily.to_csv(result / f'daily_{variant}.csv', index=False)
        point = float(groups['ndcg'].mean() - baseline['ndcg'].mean())
        row = {'variant': variant, 'active_features': control['active_feature_count'],
               'NDCG@10': metrics['NDCG@10'], 'QueryAUC': metrics['QueryAUC'],
               'delta_ndcg_vs_M_sub': point, 'fit_seconds': control['fit_seconds'],
               'tree_count': control['evaluation']['tree_count']}
        for block in [3, 4]:
            low, high = paired_delta_interval(daily, indices[block])
            intervals.append({'variant': variant, 'reference': 'M-sub', 'block_days': block,
                              'replicates': 20_000, 'delta_ndcg': point, 'ci_low': low, 'ci_high': high,
                              'confidence': .95, 'two_sided': True, 'validation_reused': True})
            row[f'ci_{block}d_low'], row[f'ci_{block}d_high'] = low, high
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(result / 'comparison.csv', index=False)
    pd.DataFrame(intervals).to_csv(result / 'bootstrap_summary.csv', index=False)
    report = ['# Контроли временных DFS-признаков', '',
              'Общая подвыборка целых train-групп; полная validation исходного окна.',
              'Оценка использует восстановленный start_ts и повторно использованную validation.', '',
              '| Вариант | Активных признаков | NDCG@10 | QueryAUC | Δ NDCG к M-sub | Fit, мин |',
              '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {row['variant']} | {row['active_features']} | {row['NDCG@10']:.6f} | "
                      f"{row['QueryAUC']:.6f} | {row['delta_ndcg_vs_M_sub']:+.6f} | "
                      f"{row['fit_seconds']/60:.1f} |")
    report += ['', 'Парный циклический bootstrap: блоки 3 и 4 дня, '
               '20 000 реплик, seed 42. Двусторонние 95% интервалы: bootstrap_summary.csv.',
               'Метрики относятся к наблюдавшимся user–day событиям со старым восстановленным start_ts.']
    (result / 'report.md').write_text('\n'.join(report) + '\n')
    record = {'completed': True, 'variants': rows, 'test_used': False, 'c4_deferred': True,
              'validation_reused': True, 'selection': read_json(result / 'selection.json')}
    write_json(result / 'result.json', record)
    return record


def worker(args):
    """Один этап в отдельном процессе: память освобождается до следующего этапа."""
    result, data = run_paths(args.run_name)
    if args.stage == 'select':
        return prepare_selection(args.run_name)
    if args.stage == 'dfs':
        return prepare_dfs(args.run_name)
    if args.stage == 'manual':
        return prepare_manual_sessions(args.run_name, args.split)
    if args.stage == 'quantize':
        matrix = read_json(result / f'manual_sessions_{args.split}.json')
        return prepare_augmented_pool(
            matrix['matrix_path'], data / 'selection' / f'{args.split}.npz',
            matrix['features'], data / 'manual_quantized', args.split,
            base_borders=BASE_BORDERS if args.split == 'train' else None)
    if args.stage == 'pilot':
        return fit_variant(args.run_name, 'D462-sub', pilot=True)
    if args.stage == 'fit':
        return fit_variant(args.run_name, args.variant)
    if args.stage == 'compare':
        return compare_variants(args.run_name)
    raise ValueError('Неизвестный этап')


def _worker_command(args, stage, split=None, variant=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()),
               '--run-name', args.run_name, '--stage', stage]
    if split:
        command += ['--split', split]
    if variant:
        command += ['--variant', variant]
    return command


def _stop_worker(process):
    """Остановить только дочерний процесс, созданный этой очередью."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_stage(args, guard, deadline, stage, split=None, variant=None):
    result, _ = run_paths(args.run_name)
    if guard.poll() is not None or (result / 'external_stopped.json').exists():
        raise RuntimeError('Внешний контроль ресурсов остановлен')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Исчерпан общий бюджет машинного времени')
    identity = '_'.join(str(x) for x in [stage, split, variant] if x)
    write_json(result / 'status.json', {'completed': False, 'stage': stage,
                                       'split': split, 'variant': variant, 'unix_time': time.time()})
    print(json.dumps({'stage': stage, 'split': split, 'variant': variant, 'event': 'started'}), flush=True)
    with (result / 'logs' / f'{identity}.log').open('a', buffering=1) as log:
        process = subprocess.Popen(_worker_command(args, stage, split, variant),
                                   stdout=log, stderr=subprocess.STDOUT)
        started = time.perf_counter()
        try:
            while process.poll() is None:
                if guard.poll() is not None or (result / 'external_stopped.json').exists():
                    raise RuntimeError('Сработал или завершился внешний контроль ресурсов')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Исчерпан общий бюджет машинного времени')
                time.sleep(1)
        except BaseException:
            _stop_worker(process)
            raise
    if process.returncode != 0:
        raise RuntimeError(f'Этап {identity} завершился с exit_code={process.returncode}; см. его log')
    elapsed = time.perf_counter() - started
    print(json.dumps({'stage': identity, 'event': 'finished', 'elapsed_seconds': elapsed}), flush=True)


def supervisor(args):
    """Очередь не запускает новый этап после ошибки, нехватки RAM или истечения бюджета."""
    result, data = run_paths(args.run_name)
    initialize(args.run_name)
    if (result / 'external_stopped.json').exists():
        raise RuntimeError('Предыдущая очередь остановлена guard: нужен отдельный разбор')
    if (result / 'result.json').exists():
        print('Контроли уже завершены', flush=True)
        return
    (result / 'logs').mkdir(exist_ok=True)
    started = time.perf_counter()
    policy_path = result / 'runtime_policy.json'
    if policy_path.exists():
        policy = read_json(policy_path)
        if policy['max_hours'] != args.max_hours:
            raise ValueError('Бюджет этого запуска уже зафиксирован')
    else:
        policy = {'max_hours': args.max_hours, 'started_unix': time.time(),
                  'deadline_unix': time.time() + args.max_hours * 3600,
                  'sequential': True, 'threads': 4}
        write_json(policy_path, policy)
    remaining = policy['deadline_unix'] - time.time()
    if remaining <= 0:
        raise TimeoutError('Общий бюджет этого запуска уже исчерпан')
    deadline = time.monotonic() + remaining
    guard_log = (result / 'logs/guard.log').open('a')
    guard = subprocess.Popen([sys.executable, '-u', str(ROOT / 'src/watch_temporal_controls.py'),
                              '--parent-pid', str(os.getpid()), '--run-name', args.run_name],
                             stdout=guard_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            if (result / 'external_resources.json').exists():
                acknowledgement = read_json(result / 'external_resources.json')
                if (acknowledgement.get('parent_pid') == os.getpid()
                        and acknowledgement.get('parent_create_time') == psutil.Process().create_time()):
                    break
            if guard.poll() is not None:
                raise RuntimeError('Не запустился внешний guard')
            time.sleep(.1)
        else:
            raise TimeoutError('Guard не подтвердил готовность')
        run_stage(args, guard, deadline, 'select')
        run_stage(args, guard, deadline, 'dfs')
        run_stage(args, guard, deadline, 'pilot', variant='D462-sub')
        print(json.dumps(read_json(result / 'pilot_assessment.json'), ensure_ascii=False), flush=True)
        if args.stop_after_pilot:
            write_json(result / 'status.json', {'completed': False, 'stage': 'pilot_finished',
                       'elapsed_seconds': time.perf_counter() - started})
            return
        for split in ['train', 'val']:
            run_stage(args, guard, deadline, 'manual', split=split)
        for split in ['train', 'val']:
            run_stage(args, guard, deadline, 'quantize', split=split)
        for variant in VARIANTS:
            run_stage(args, guard, deadline, 'fit', variant=variant)
        run_stage(args, guard, deadline, 'compare')
        write_json(result / 'status.json', {'completed': True, 'stage': 'finished',
                   'elapsed_seconds': time.perf_counter() - started})
    except BaseException as error:
        write_json(result / 'status.json', {'completed': False, 'stage': 'stopped',
                   'error': str(error), 'elapsed_seconds': time.perf_counter() - started})
        raise
    finally:
        if guard.poll() is None:
            guard.terminate()
            guard.wait(timeout=10)
        guard_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--legacy-run', default=os.environ.get('TLAB_EXPERIMENT_RUN'))
    parser.add_argument('--full-run', default=os.environ.get('TLAB_CONTROLS_FULL_RUN',
                        'temporal_full462_20261023_v5'))
    parser.add_argument('--baseline-dir', default=os.environ.get('TLAB_CONTROLS_BASELINE_DIR',
                        'results/tuning/optuna_20261023_2h_v2/queue/control_manual_ranker/attempt_0001'))
    parser.add_argument('--base-borders', default=os.environ.get('TLAB_CONTROLS_BASE_BORDERS',
                        'results/baselines/timing-20261020/train_borders.tsv'))
    parser.add_argument('--stage', choices=['all', 'select', 'dfs', 'manual', 'quantize',
                                           'pilot', 'fit', 'compare'], default='all')
    parser.add_argument('--split', choices=['train', 'val'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--max-hours', type=float, default=6)
    parser.add_argument('--stop-after-pilot', action='store_true')
    args = parser.parse_args()
    configure_legacy(args.legacy_run)
    temporal_controls_data.configure_sources(args.full_run, args.baseline_dir, args.base_borders)
    global BASE_BORDERS
    BASE_BORDERS = temporal_controls_data.BASE_BORDERS
    if not 0 < args.max_hours <= 24:
        parser.error('Некорректный бюджет времени')
    if args.stage in ['manual', 'quantize'] and args.split is None:
        parser.error('Нужен split')
    if args.stage == 'fit' and args.variant is None:
        parser.error('Нужен variant')
    if args.stage == 'all':
        supervisor(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
