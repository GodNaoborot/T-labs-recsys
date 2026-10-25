"""Повторить контроль ручных признаков без текущего канала взаимодействия.

По умолчанию печатается план. Обучение и test запускаются отдельными этапами.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from block_bootstrap import aggregate_days, circular_day_indices
from compare_temporal_controls_pairs import absolute_delta_draws
from c4_training import load_targets
from replay_paths import run_name
from project_artifacts import TASK_ROOT, file_hash
from temporal_full_training import fit_ranker

VARIANT = 'manual48_no_current_channel'
REFERENCES = ('C2b-t', 'D462-t')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def portable(value):
    """Сохранять пути внутри проекта относительно task_1."""
    if isinstance(value, dict):
        return {portable(key): portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith(str(TASK_ROOT) + '/'):
        return Path(value).relative_to(TASK_ROOT).as_posix()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(portable(value), ensure_ascii=False,
                                   indent=2, allow_nan=False) + '\n', encoding='utf-8')


def channel_parameters(names, original):
    """Исключить только текущий is_organic; исторические агрегаты оставить."""
    if names.count('is_organic') != 1 or original.get('ignored_features'):
        raise ValueError('Ожидается исходный набор с единственным текущим is_organic')
    return {**original, 'ignored_features': [names.index('is_organic')]}


def artifact(path):
    path = Path(path)
    return {'path': portable(str(path)), 'sha256': file_hash(path)}


def comparison_rows(reference_dir, split, metrics, groups, fit, training_dir=None):
    table = pd.read_csv(reference_dir / 'comparison.csv', float_precision='round_trip')
    rows = table.loc[table.variant.isin(REFERENCES)].copy()
    rows['split'] = split
    if 'fit_seconds' not in rows:
        rows['fit_seconds'] = np.nan
    if training_dir is not None:
        training = pd.read_csv(training_dir / 'comparison.csv', float_precision='round_trip')
        times = training.set_index('variant')['fit_seconds']
        rows['fit_seconds'] = rows['fit_seconds'].fillna(rows.variant.map(times))
    row = {'variant': VARIANT, 'split': split, 'active_features': 48,
           **metrics, 'fit_seconds': fit['fit_seconds'],
           'tree_count': fit['evaluation']['tree_count'],
           'rows': int(groups.group_size.sum()), 'groups': len(groups)}
    columns = ['variant', 'split', 'active_features', 'NDCG@10', 'QueryAUC',
               'fit_seconds', 'tree_count', 'rows', 'groups']
    return pd.concat([rows[columns], pd.DataFrame([row])[columns]], ignore_index=True)


def paired_intervals(candidate, reference, split):
    """Считать парные интервалы только после точной сверки состава групп."""
    pd.testing.assert_frame_equal(candidate[['group_id', 'day', 'group_size']],
                                  reference[['group_id', 'day', 'group_size']])
    daily = aggregate_days(candidate.day.to_numpy(), reference.ndcg.to_numpy(),
                           candidate.ndcg.to_numpy())
    point = float((candidate.ndcg - reference.ndcg).mean())
    records = []
    for block in (3, 4):
        indices = circular_day_indices(len(daily), block, 20000, seed=42)
        low, high = np.quantile(absolute_delta_draws(daily, indices), [.025, .975])
        records.append({'split': split, 'candidate': VARIANT, 'reference': 'D462-t',
                        'block_days': block, 'delta_ndcg': point, 'ci_low': float(low),
                        'ci_high': float(high), 'replicates': 20000, 'seed': 42})
    return pd.DataFrame(records)


def fit_control(training_run, control_run):
    source = TASK_ROOT / 'results/c4' / training_run
    output = TASK_ROOT / 'results/c4' / control_run
    if output.exists():
        raise FileExistsError('Для нового контроля нужна новая папка: ' + str(output))
    if read_json(source / 'status.json').get('completed') is not True:
        raise ValueError('Исходное C4-обучение не завершено')
    plan = read_json(source / 'training_plan.json')
    names = plan['schemas']['manual49']
    if len(names) != 49 or len(set(names)) != 49:
        raise ValueError('Нужна общая физическая схема из 49 ручных входов')
    parameters = channel_parameters(names, plan['parameters']['C2b-t'])
    data = TASK_ROOT / 'data/c4' / training_run
    pools = source / 'quantized/manual49'
    inputs = {name: artifact(path) for name, path in {
        'training_plan': source / 'training_plan.json', 'train_pool': pools / 'train.qbin',
        'val_pool': pools / 'val.qbin', 'train_targets': data / 'train_targets.npz',
        'val_targets': data / 'val_targets.npz', 'train_borders': pools / 'train_borders.tsv',
    }.items()}
    frozen_pools = read_json(source / 'pools.json')
    if frozen_pools.get('completed') is not True:
        raise ValueError('Исходные пулы не завершены')
    for split in ('train', 'val'):
        pool_record = frozen_pools['schemas']['manual49'][split]
        if inputs[split + '_pool']['sha256'] != pool_record['pool_sha256'] or pool_record['features'] != names:
            raise ValueError('Изменился исходный Pool: ' + split)
        if inputs[split + '_targets']['sha256'] != plan['inputs'][split + '_targets']['sha256']:
            raise ValueError('Изменились общие targets: ' + split)
    output.mkdir(parents=True)
    protocol = {'variant': VARIANT, 'training_run': training_run, 'active_features': 48,
                'physical_features': 49, 'excluded_current_features': ['is_organic'],
                'features': names, 'parameters': parameters, 'inputs': inputs,
                'new_hyperparameter_search': False, 'test_evaluated': False,
                'source_sha256': file_hash(Path(__file__))}
    write_json(output / 'protocol.json', protocol)
    fitted = fit_ranker(pools / 'train.qbin', pools / 'val.qbin', data / 'val_targets.npz',
                        names, output / 'model', parameters)
    for path in (output / 'model').glob('*.json'):
        write_json(path, read_json(path))
    groups = pd.read_parquet(output / 'model/group_ndcg.parquet')
    reference = pd.read_parquet(source / 'models/D462-t/group_ndcg.parquet')
    table = comparison_rows(source, 'validation', fitted['evaluation']['metrics'], groups, fitted)
    table.to_csv(output / 'comparison.csv', index=False)
    paired_intervals(groups, reference, 'validation').to_csv(output / 'bootstrap_summary.csv', index=False)
    result = {'completed': True, 'variant': VARIANT, 'active_features': 48,
              'physical_features': 49, 'excluded_current_features': ['is_organic'],
              'fit_seconds': fitted['fit_seconds'], 'tree_count': fitted['evaluation']['tree_count'],
              'validation_metrics': fitted['evaluation']['metrics'], 'test_metrics': None,
              'model_sha256': fitted['model_sha256'], 'training_run': training_run,
              'test_evaluated': False, 'fit': fitted}
    write_json(output / 'result.json', result)
    return result


def evaluate_control(training_run, control_run, test_run):
    """Оценить сохранённый контроль на готовом test с теми же train-порогами."""
    from catboost import CatBoostRanker, Pool
    from train_dfs_catboost import ranking_metrics
    from optuna_evaluation import _group_scores
    source = TASK_ROOT / 'results/c4' / training_run
    output = TASK_ROOT / 'results/c4' / control_run
    test = TASK_ROOT / 'results/c4_test' / test_run
    result = read_json(output / 'result.json')
    if result.get('test_evaluated'):
        raise FileExistsError('Этот контроль уже оценён на test')
    if result.get('training_run') != training_run:
        raise ValueError('Контроль обучен на другом запуске')
    protocol = read_json(output / 'protocol.json')
    plan = read_json(test / 'plan.json')
    expected = plan['c4_provenance']['training_plan']['sha256']
    if file_hash(source / 'training_plan.json') != expected:
        raise ValueError('Test относится к другому C4-обучению')
    pools = test / 'quantized/manual49'
    border = source / 'quantized/manual49/train_borders.tsv'
    if (pools / 'test_applied_borders.tsv').read_bytes() != border.read_bytes():
        raise ValueError('Test получил другие train-пороги')
    if file_hash(border) != protocol['inputs']['train_borders']['sha256']:
        raise ValueError('Train-пороги контроля изменились')
    model_path = output / 'model/model.cbm'
    if file_hash(model_path) != result['model_sha256']:
        raise ValueError('Модель контроля изменена')
    names = protocol['features']
    model = CatBoostRanker().load_model(str(model_path))
    if model.feature_names_ != names or model.get_all_params()['ignored_features'] != [names.index('is_organic')]:
        raise ValueError('Изменилась схема модели')
    pool = Pool('quantized://' + str(pools / 'test.qbin'), thread_count=4)
    arrays = load_targets(TASK_ROOT / 'data/c4_test' / test_run / 'targets.npz')
    np.testing.assert_array_equal(np.asarray(pool.get_label(), dtype='float64'), arrays['label'])
    if pool.get_feature_names() != names:
        raise ValueError('Test Pool имеет другие признаки')
    score = model.predict(pool, thread_count=4)
    metrics = ranking_metrics(arrays['label'], arrays['group_id'], score)
    starts = np.r_[0, np.flatnonzero(np.diff(arrays['group_id'])) + 1]
    groups = _group_scores(arrays, starts, score, metrics['NDCG@10'])
    reference = pd.read_parquet(test / 'models/D462-t/group_ndcg.parquet')
    intervals = paired_intervals(groups, reference, 'test')
    comparison = comparison_rows(test, 'test', metrics, groups, result['fit'], training_dir=source)
    if file_hash(model_path) != result['model_sha256']:
        raise ValueError('Модель изменилась во время оценки')
    np.save(output / 'test_score.npy', score)
    groups.to_parquet(output / 'test_group_ndcg.parquet', index=False)
    pd.concat([pd.read_csv(output / 'comparison.csv'), comparison], ignore_index=True).to_csv(output / 'comparison.csv', index=False)
    pd.concat([pd.read_csv(output / 'bootstrap_summary.csv'), intervals], ignore_index=True).to_csv(output / 'bootstrap_summary.csv', index=False)
    result.update(test_metrics=metrics, test_evaluated=True, test_run=test_run,
                  test_already_open=True, new_blind_test=False)
    write_json(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'fit', 'evaluate'], default='plan')
    parser.add_argument('--training-run', type=run_name, default='c4_packet_20261024_v1')
    parser.add_argument('--test-run', type=run_name, default='c4_test_20261024_v1')
    parser.add_argument('--run-name', type=run_name, default='channel_control_repeat')
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'evaluate' and not args.include_test:
        parser.error('Для повторной оценки test нужен --include-test')
    if args.stage == 'plan':
        print(json.dumps({'executed': False, 'training_run': args.training_run,
                          'output': 'results/c4/' + args.run_name,
                          'fit_requires': '--stage fit',
                          'test_requires': '--stage evaluate --include-test'}, ensure_ascii=False))
    elif args.stage == 'fit':
        fit_control(args.training_run, args.run_name)
    else:
        evaluate_control(args.training_run, args.run_name, args.test_run)


if __name__ == '__main__':
    main()
