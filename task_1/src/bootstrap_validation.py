"""Подготовка парных NDCG user–day для bootstrap без обучения моделей."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import re
import time

import catboost
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from catboost.utils import eval_metric
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from block_bootstrap import group_ndcg

DAY = 86_400
THREADS = 4
NDCG = 'NDCG:top=10;type=Base;denominator=LogPosition'
IDENTITY = ['uid', 'item_id', 'start_ts', 'label', 'group_id',
            'is_organic', 'track_length_seconds']


def file_sha256(path):
    """Вычислить hash файла небольшими порциями."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    """Прочитать сохраненную конфигурацию UTF-8."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Сохранить JSON атомарно, чтобы прерванная запись не стала кэшем."""
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def runtime_versions():
    """Зафиксировать версии библиотек, влияющих на прогнозы и агрегации."""
    return {'numpy': np.__version__, 'pandas': pd.__version__,
            'pyarrow': pa.__version__, 'catboost': catboost.__version__}


def analysis_paths(project):
    """Связать manual и DFS одного повторного запуска, сохранив старый default."""
    name = os.environ.get('TLAB_EXPERIMENT_RUN')
    if name is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Некорректный TLAB_EXPERIMENT_RUN')
    results = project / 'results'
    if name is None:
        manual = results / 'baselines/timing-20261020'
        dfs = results / 'experiment-1/dfs'
        data = project / 'data/experiment-1/featuretools'
    else:
        manual = results / 'baselines/runs' / name
        dfs = results / 'experiment-1/runs' / name / 'dfs'
        data = project / 'data/experiment-1/runs' / name / 'featuretools'
    return {
        'base_config': manual / 'run_config.json',
        'base_metrics': manual / 'validation_metrics.csv',
        'dfs_config': dfs / 'run_config.json',
        'dfs_metrics': dfs / 'validation_metrics.csv',
        'features': project / 'data/features.parquet',
        'metadata': data / 'full_val_cumulative/val_all_cumulative_metadata.parquet',
        'targets': data / 'catboost_pools/val_targets.npz',
        'pool': data / 'catboost_pools/val.qbin',
        'ranker_scores': dfs / 'ranker_val_score.npy',
        'pointwise_scores': dfs / 'pointwise_val_score.npy',
    }


def model_information(base, dfs):
    """Привести имена моделей двух конфигураций к одному порядку."""
    return {
        'base_ranker': base['models']['YetiRank'],
        'base_pointwise': base['models']['MultiClass'],
        'dfs_ranker': dfs['models']['ranker'],
        'dfs_pointwise': dfs['models']['pointwise'],
    }


def input_fingerprint(paths, models, seed):
    """Связать кэш с содержимым данных, моделей, кода и правилом ties."""
    files = {name: {'path': str(path), 'sha256': file_sha256(path)}
             for name, path in paths.items()}
    for name, info in models.items():
        digest = file_sha256(info['path'])
        if digest != info['sha256']:
            raise ValueError(f'Hash модели не совпал: {name}')
        files[name] = {'path': info['path'], 'sha256': digest}
    for name in ('bootstrap_validation.py', 'block_bootstrap.py'):
        path = Path(__file__).with_name(name)
        files[name] = {'path': str(path), 'sha256': file_sha256(path)}
    return {'files': files, 'tie_seed': seed, 'ndcg': NDCG, 'versions': runtime_versions(),
            'tie_policy': 'descending score, then descending global seed permutation'}


def mixed_group_mask(labels, groups):
    """Оставить целиком группы, где есть хотя бы две различные метки."""
    if len(labels) == 0:
        raise ValueError('Пустая validation-выборка')
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    sizes = np.diff(np.r_[starts, len(groups)])
    mixed = np.minimum.reduceat(labels, starts) != np.maximum.reduceat(labels, starts)
    return np.repeat(mixed, sizes)


def read_validation_rows(path, columns, filters=None):
    """Отсортировать весь val по group_id и удалить постоянные группы."""
    table = pq.read_table(path, columns=columns, filters=filters).sort_by('group_id')
    labels = table['label'].to_numpy()
    groups = table['group_id'].to_numpy()
    table = table.filter(pa.array(mixed_group_mask(labels, groups)))
    return table.to_pandas(split_blocks=True, self_destruct=True)


def load_validation_inputs(paths, feature_names):
    """Прочитать manual-признаки и независимые metadata DFS только для val."""
    columns = list(dict.fromkeys(IDENTITY + feature_names))
    base = read_validation_rows(paths['features'], columns, [('split', '==', 'val')])
    columns = ['log_id'] + ['time' if c == 'start_ts' else c for c in IDENTITY]
    dfs = read_validation_rows(paths['metadata'], columns)
    dfs = dfs.rename(columns={'time': 'start_ts'})
    with np.load(paths['targets']) as saved:
        targets = {name: saved[name] for name in saved.files}
    return base, dfs, targets


def assert_paired_rows(base, dfs, targets):
    """Сверить каждую строку по идентичности события, метке и порядку."""
    for column in IDENTITY:
        np.testing.assert_array_equal(base[column], dfs[column], err_msg=column)
    for column in ('label', 'group_id', 'log_id'):
        np.testing.assert_array_equal(dfs[column], targets[column], err_msg=column)
    days = base['start_ts'].to_numpy() // DAY
    np.testing.assert_array_equal(days, base['group_id'].to_numpy() % 1000)
    np.testing.assert_array_equal(base['uid'], base['group_id'].to_numpy() // 1000)
    np.testing.assert_array_equal(np.unique(days), np.arange(270, 285))
    if len(base) != 2_882_983 or base['group_id'].nunique() != 56_946:
        raise ValueError('Состав оцениваемого val отличается от зафиксированного')


def load_models(information, prefix):
    """Загрузить два checkpoint; обучение в этом модуле не вызывается."""
    ranker = CatBoostRanker().load_model(information[f'{prefix}_ranker']['path'])
    pointwise = CatBoostClassifier().load_model(information[f'{prefix}_pointwise']['path'])
    np.testing.assert_array_equal(pointwise.classes_, [0, 1, 2])
    return {'ranker': ranker, 'pointwise': pointwise}


def predict_models(models, pool):
    """Получить ranking-score и ожидаемую метку MultiClass на одном Pool."""
    ranker = models['ranker'].predict(pool, thread_count=THREADS)
    probabilities = models['pointwise'].predict_proba(pool, thread_count=THREADS)
    pointwise = probabilities @ models['pointwise'].classes_.astype(float)
    return {'ranker': np.asarray(ranker), 'pointwise': pointwise}


def predict_baseline(base, config, information):
    """Посчитать прогнозы manual baseline по тем же исходным val-строкам."""
    models = load_models(information, 'base')
    names = config['features']
    for estimator in models.values():
        if estimator.feature_names_ != names:
            raise ValueError('Порядок manual-признаков не совпал с checkpoint')
    pool = Pool(base[names], label=base['label'], group_id=base['group_id'],
                thread_count=THREADS)
    return predict_models(models, pool)


def checked_dfs_predictions(paths, config, information, targets):
    """Сверить сохраненные DFS-score с прогнозами самих моделей по всему val."""
    pool = Pool('quantized://' + str(paths['pool']), thread_count=THREADS)
    if pool.get_feature_names() != config['features']:
        raise ValueError('Признаки DFS Pool не совпали с конфигурацией')
    np.testing.assert_array_equal(np.asarray(pool.get_label(), dtype='float64'),
                                  targets['label'])
    models = load_models(information, 'dfs')
    fresh = predict_models(models, pool)
    for kind, score in fresh.items():
        saved = np.load(paths[f'{kind}_scores'], mmap_mode='r')
        np.testing.assert_array_equal(score, saved, err_msg=f'DFS {kind}')
    return fresh


def external_ndcg(labels, groups, score, ties):
    """Воспроизвести внешний NDCG baseline через CatBoost для проверки формулы."""
    order = np.lexsort((ties, score))
    ranks = np.empty(len(score), dtype='float64')
    ranks[order] = np.arange(len(score))
    return eval_metric(labels, ranks, NDCG, group_id=groups, thread_count=THREADS)[0]


def expected_ndcg(paths):
    """Прочитать ожидаемые NDCG именно выбранных checkpoint из их отчетов."""
    base = pd.read_csv(paths['base_metrics']).set_index('model')['NDCG@10']
    dfs = pd.read_csv(paths['dfs_metrics']).set_index('model')['NDCG@10']
    return {
        'base_ranker': float(base['catboost YetiRank']),
        'base_pointwise': float(base['catboost MultiClass, E[label]']),
        'dfs_ranker': float(dfs['DFS YetiRank']),
        'dfs_pointwise': float(dfs['DFS MultiClass E[label]']),
    }


def group_metadata(base, log_ids):
    """Сохранить одну строку на целую user–day группу в исходном порядке."""
    groups = base['group_id'].to_numpy()
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    sizes = np.diff(np.r_[starts, len(groups)])
    days = base['start_ts'].to_numpy() // DAY
    np.testing.assert_array_equal(days, np.repeat(days[starts], sizes))
    table = pd.DataFrame({'group_id': groups[starts], 'uid': base['uid'].to_numpy()[starts],
                          'day': days[starts], 'rows': sizes})
    identity = hashlib.sha256(np.asarray(log_ids, dtype='<i8').tobytes()).hexdigest()
    return table, identity


def add_group_scores(table, base, score_pairs, expected, seed):
    """Посчитать NDCG групп и сверить их среднее с CatBoost и сохраненным CSV."""
    labels, groups = base['label'].to_numpy(), base['group_id'].to_numpy()
    ties = np.random.default_rng(seed).permutation(len(base))
    checks = {}
    for name, score in score_pairs.items():
        values = group_ndcg(labels, groups, score, ties)
        native = external_ndcg(labels, groups, score, ties)
        np.testing.assert_allclose(values.mean(), native, rtol=0, atol=1e-12)
        np.testing.assert_allclose(native, expected[name], rtol=0, atol=1e-12)
        table[name] = values
        checks[name] = {'group_mean': float(values.mean()), 'catboost': native,
                        'saved_metric': expected[name], 'match': True}
    return table, checks


def compute_group_table(paths, base_config, dfs_config, information, seed):
    """Подготовить парные данные и вычислить достаточную для bootstrap таблицу."""
    base, dfs, targets = load_validation_inputs(paths, base_config['features'])
    assert_paired_rows(base, dfs, targets)
    print('PAIRED_ROWS', len(base), 'groups', base['group_id'].nunique(), flush=True)
    base_scores = predict_baseline(base, base_config, information)
    dfs_scores = checked_dfs_predictions(paths, dfs_config, information, targets)
    scores = {f'base_{key}': value for key, value in base_scores.items()}
    scores.update({f'dfs_{key}': value for key, value in dfs_scores.items()})
    table, identity = group_metadata(base, targets['log_id'])
    table, checks = add_group_scores(table, base, scores, expected_ndcg(paths), seed)
    return table, {'paired_rows': len(base), 'groups': len(table), 'days': 15,
                   'target_log_id_order_sha256': identity, 'ndcg_checks': checks,
                   'dfs_saved_predictions_match_models': True}


def verify_source_configs(base, dfs, fingerprint):
    """Проверить, что используемые файлы относятся к сохраненным моделям и Pool."""
    if fingerprint['files']['features']['sha256'] != base['evaluated_dataset']['sha256']:
        raise ValueError('Manual features.parquet изменился после сохраненного запуска')
    saved = dfs['pool_configs']['val']
    for name, key in [('pool', 'pool_sha256'), ('metadata', 'metadata_sha256')]:
        if fingerprint['files'][name]['sha256'] != saved[key]:
            raise ValueError(f'DFS источник изменился: {name}')
    for info in base['models'].values():
        if info['mode'] != 'trained_with_validation':
            raise ValueError('Выбранный baseline должен быть обучен с validation')


def verify_comparable_models(information):
    """Сверить objective и основные параметры, а не выбирать модели по score."""
    keys = ['loss_function', 'depth', 'iterations', 'learning_rate', 'random_seed',
            'border_count', 'od_wait', 'use_best_model', 'eval_metric']
    for kind in ('ranker', 'pointwise'):
        base, dfs = information[f'base_{kind}'], information[f'dfs_{kind}']
        for key in keys:
            if base['params'][key] != dfs['actual_params'][key]:
                raise ValueError(f'Различается параметр {kind}: {key}')
        if base['training_threads'] != dfs['training_threads']:
            raise ValueError(f'Различается число потоков {kind}')


def cached_group_table(output, fingerprint):
    """Повторно использовать кэш только при совпадении всех источников и hash."""
    report_path, table_path = output / 'preparation_report.json', output / 'group_ndcg.parquet'
    if not report_path.exists() or not table_path.exists():
        return None
    report = read_json(report_path)
    if report['inputs'] != fingerprint:
        return None
    if file_sha256(table_path) != report['group_table_sha256']:
        raise ValueError('Поврежден кэш group_ndcg.parquet')
    return pd.read_parquet(table_path)


def save_prepared_groups(table, output, checks, fingerprint, started):
    """Сохранить групповые NDCG и отчет проверок с идентичностью источников."""
    table.to_parquet(output / 'group_ndcg.parquet', index=False)
    report = {'completed': True, 'inputs': fingerprint, 'checks': checks,
              'group_table_sha256': file_sha256(output / 'group_ndcg.parquet'),
              'elapsed_seconds': time.perf_counter() - started,
              'versions': runtime_versions()}
    write_json(output / 'preparation_report.json', report)
    print('PREPARED', checks, 'seconds', report['elapsed_seconds'], flush=True)


def prepare_validation_groups(project, output, seed=42):
    """Создать либо проверить кэш групп; не читать train/test и не учить модели."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    paths = analysis_paths(project)
    base, dfs = read_json(paths['base_config']), read_json(paths['dfs_config'])
    information = model_information(base, dfs)
    verify_comparable_models(information)
    fingerprint = input_fingerprint(paths, information, seed)
    verify_source_configs(base, dfs, fingerprint)
    cached = cached_group_table(output, fingerprint)
    if cached is not None:
        print('REUSE_GROUP_METRICS', len(cached), flush=True)
        return cached
    table, checks = compute_group_table(paths, base, dfs, information, seed)
    save_prepared_groups(table, output, checks, fingerprint, started)
    return table


def main():
    """Запустить подготовку отдельным процессом и освободить его память на выходе."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if args.run_name:
        os.environ['TLAB_EXPERIMENT_RUN'] = args.run_name
    project = Path(__file__).resolve().parents[1]
    prepare_validation_groups(project, args.output, args.seed)
    gc.collect()


if __name__ == '__main__':
    main()
