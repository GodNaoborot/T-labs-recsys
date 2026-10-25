"""CatBoost на DFS-признаках: тот же split, группы и параметры, что в baseline.

Подготовка каждого Pool идет отдельным процессом. Временная float32-матрица
хранится на диске; пороги квантования учатся на train и применяются к val.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import re
import resource
import threading
import time

import catboost
from catboost import CatBoostRanker, CatBoostClassifier, Pool
from catboost.utils import eval_metric
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import log_loss

def experiment_run_name():
    """Проверяет имя отдельного запуска, исключая выход за его каталоги."""
    name = os.environ.get('TLAB_EXPERIMENT_RUN')
    if name is not None and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) is None:
        raise ValueError('TLAB_EXPERIMENT_RUN: нужны 1–64 буквы/цифры/_/-, начиная с буквы или цифры')
    return name


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = experiment_run_name()
RUN_FOLDER = Path('runs') / RUN_NAME if RUN_NAME is not None else None
DATA = (ROOT / 'data/experiment-1/featuretools' if RUN_FOLDER is None
        else ROOT / 'data/experiment-1' / RUN_FOLDER / 'featuretools')
CACHE = DATA / 'catboost_pools'
MODELS = (ROOT / 'models/experiment-1/dfs' if RUN_FOLDER is None
          else ROOT / 'models/experiment-1' / RUN_FOLDER / 'dfs')
RESULTS = (ROOT / 'results/experiment-1/dfs' if RUN_FOLDER is None
           else ROOT / 'results/experiment-1' / RUN_FOLDER / 'dfs')
SEED, THREADS, CHUNK = 42, 4, 100_000
NDCG = 'NDCG:top=10;type=Base;denominator=LogPosition'


def sha256(path):
    """Считает SHA256 файла небольшими порциями."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Атомарно заменяет JSON только после полной записи временного файла."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def release():
    """Освобождает память Python и Arrow после большого этапа."""
    gc.collect()
    pa.default_memory_pool().release_unused()
    import ctypes
    ctypes.CDLL(None).malloc_trim(0)


def pool_paths(split):
    """Проверяет входные файлы и запрет перезаписи готового Pool."""
    folder = DATA / f'full_{split}_cumulative'
    assert (folder / 'SUCCESS.json').exists(), f'Нет готовых признаков {split}'
    paths = {'features': folder / f'{split}_all_cumulative_X.parquet',
             'metadata': folder / f'{split}_all_cumulative_metadata.parquet',
             'pool': CACHE / f'{split}.qbin', 'matrix': CACHE / f'{split}_float32.npy',
             'borders': CACHE / 'train_borders.tsv'}
    assert not paths['pool'].exists(), f'Готовый Pool уже существует: {paths["pool"]}'
    if split == 'val':
        assert paths['borders'].exists(), 'Сначала подготовьте train'
    return paths


def read_pool_metadata(path, split):
    """Загружает метаданные и сверяет исходный порядок строк с baseline."""
    columns = ['log_id', 'label', 'group_id', 'uid', 'item_id', 'time',
               'is_organic', 'track_length_seconds']
    meta = pq.read_table(path, columns=columns).to_pandas(split_blocks=True, self_destruct=True)
    baseline_columns = ['uid', 'item_id', 'start_ts', 'label', 'group_id',
                        'is_organic', 'track_length_seconds']
    expected = pq.read_table(ROOT / 'data/features.parquet', columns=baseline_columns,
                             filters=[('split', '==', split)]).to_pandas()
    for column in baseline_columns:
        np.testing.assert_array_equal(meta['time' if column == 'start_ts' else column], expected[column])
    return meta


def select_mixed_groups(meta):
    """Сортирует строки по группе и оставляет группы с разными метками."""
    order = pc.sort_indices(pa.table({'group_id': meta['group_id']}),
                            sort_keys=[('group_id', 'ascending')]).to_numpy()
    groups = meta['group_id'].to_numpy()[order]
    labels = meta['label'].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    sizes = np.diff(np.r_[starts, len(groups)])
    mixed = np.minimum.reduceat(labels, starts) != np.maximum.reduceat(labels, starts)
    keep = np.repeat(mixed, sizes)
    positions = order[keep]
    stats = {'rows_total': len(meta), 'groups_total': len(starts),
             'rows_used': len(positions), 'groups_used': int(mixed.sum())}
    return positions, labels[keep], groups[keep], stats


def prepare_row_mapping(meta, split):
    """Готовит метки и отображение исходных строк в отсортированный Pool."""
    positions, labels, groups, stats = select_mixed_groups(meta)
    print('SPLIT', split, stats, flush=True)
    expected_counts = {'train': (7_931_435, 155_600), 'val': (2_882_983, 56_946)}
    assert (len(labels), stats['groups_used']) == expected_counts[split]
    log_ids = meta['log_id'].to_numpy()[positions]
    destination = np.full(len(meta), -1, dtype='int64')
    destination[positions] = np.arange(len(positions))
    source_ids = meta['log_id'].to_numpy(copy=True)
    return labels, groups, log_ids, destination, source_ids, stats


def copy_feature_batch(matrix, table, destination, names):
    """Переносит одну порцию признаков в нужные строки float32-матрицы."""
    use = destination >= 0
    for index, name in enumerate(names):
        values = table[name].to_numpy().astype('float32')
        assert not np.isinf(values).any(), name
        matrix[destination[use], index] = values[use]


def fill_feature_matrix(matrix, feature_path, names, destination, source_ids, split):
    """Потоково читает Parquet, проверяя log_id перед записью каждой порции."""
    offset, last_update = 0, time.monotonic()
    batches = pq.ParquetFile(feature_path).iter_batches(batch_size=CHUNK, columns=['log_id'] + names)
    for batch in batches:
        table = pa.Table.from_batches([batch])
        right = offset + len(batch)
        np.testing.assert_array_equal(table['log_id'].to_numpy(), source_ids[offset:right])
        copy_feature_batch(matrix, table, destination[offset:right], names)
        offset = right
        if time.monotonic() - last_update > 20:
            print('READ', split, offset, '/', len(source_ids), flush=True)
            last_update = time.monotonic()
    assert offset == len(source_ids)
    matrix.flush()


def create_feature_matrix(paths, names, labels, destination, source_ids, split):
    """Создает дисковую матрицу в порядке столбцов для чтения CatBoost."""
    assert not paths['matrix'].exists(), paths['matrix']
    matrix = np.lib.format.open_memmap(paths['matrix'], mode='w+', dtype='float32',
                                      shape=(len(labels), len(names)), fortran_order=True)
    fill_feature_matrix(matrix, paths['features'], names, destination, source_ids, split)
    return matrix


def quantize_and_save(matrix, labels, groups, names, paths, split):
    """Квантует Pool по train-порогам и сохраняет его без обучения модели."""
    matrix.flags.writeable = False
    pool = Pool(matrix, label=labels, group_id=groups, feature_names=names, thread_count=THREADS)
    print('QUANTIZE', split, 'RSS GiB', round(psutil.Process().memory_info().rss / 2**30, 2), flush=True)
    pool.quantize(border_count=254, random_seed=SEED, nan_mode='Min', used_ram_limit='6GB',
                  input_borders=str(paths['borders']) if split == 'val' else None)
    if split == 'train':
        pool.save_quantization_borders(str(paths['borders']))
    pool.save(str(paths['pool']))


def record_pool(split, paths, names, stats):
    """Фиксирует порядок признаков и хеши входов и сохраненного Pool."""
    stats.update(features=names, pool_bytes=paths['pool'].stat().st_size,
                 pool_sha256=sha256(paths['pool']), feature_file_sha256=sha256(paths['features']),
                 metadata_sha256=sha256(paths['metadata']), borders_sha256=sha256(paths['borders']),
                 targets_match_baseline=True)
    write_json(CACHE / f'{split}_config.json', stats)
    return stats


def prepare_pool(split):
    """Проверяет выборку и последовательно готовит дисковую матрицу и Pool."""
    paths = pool_paths(split)
    names = [name for name in pq.read_schema(paths['features']).names if name not in {'log_id', 'artist_id'}]
    assert len(names) == 130
    meta = read_pool_metadata(paths['metadata'], split)
    labels, groups, log_ids, destination, source_ids, stats = prepare_row_mapping(meta, split)
    del meta
    release()
    matrix = create_feature_matrix(paths, names, labels, destination, source_ids, split)
    del destination, source_ids
    release()
    quantize_and_save(matrix, labels, groups, names, paths, split)
    np.savez_compressed(CACHE / f'{split}_targets.npz', label=labels, group_id=groups, log_id=log_ids)
    del matrix
    release()
    paths['matrix'].unlink()
    return record_pool(split, paths, names, stats)


def ranking_metrics(labels, groups, score):
    """Считает NDCG с фиксированным разрешением ties и QueryAUC по raw score."""
    score = np.asarray(score, dtype='float64')
    assert np.isfinite(score).all() and score.shape == labels.shape
    tie_breaker = np.random.default_rng(SEED).permutation(len(labels))
    order = np.lexsort((tie_breaker, score))
    ranks = np.empty(len(score), dtype='float64')
    ranks[order] = np.arange(len(score))
    return {'NDCG@10': eval_metric(labels, ranks, NDCG, group_id=groups, thread_count=THREADS)[0],
            'QueryAUC': eval_metric(labels, score, 'QueryAUC:type=Ranking',
                                    group_id=groups, thread_count=THREADS)[0]}


def load_training_data():
    """Загружает готовые Pool и проверяет метки val и порядок признаков."""
    train = Pool('quantized://' + str(CACHE / 'train.qbin'), thread_count=THREADS)
    val = Pool('quantized://' + str(CACHE / 'val.qbin'), thread_count=THREADS)
    with np.load(CACHE / 'val_targets.npz') as targets:
        labels, groups = targets['label'], targets['group_id']
    np.testing.assert_array_equal(np.asarray(val.get_label(), dtype=float), labels)
    assert train.get_feature_names() == val.get_feature_names()
    return train, val, labels, groups


def common_parameters():
    """Возвращает неизмененные параметры обучения обеих моделей."""
    return dict(random_seed=SEED, learning_rate=0.1, iterations=600,
                early_stopping_rounds=50, use_best_model=True, verbose=25,
                thread_count=THREADS, allow_writing_files=False)


def build_models(common):
    """Создает ranker и classifier с прежними лоссами и числом потоков."""
    return [('ranker', CatBoostRanker(loss_function='YetiRank', eval_metric=NDCG, **common)),
            ('pointwise', CatBoostClassifier(loss_function='MultiClass', eval_metric='MultiClass',
                                             **{**common, 'thread_count': 8}))]


def training_config(names, common):
    """Описывает признаки, параметры и ограничения текущего обучения."""
    return {'features': names, 'categorical_features': [], 'excluded_raw_feature': 'artist_id',
            'random_seed': SEED, 'drop_constant_groups': True, 'evaluation_split': 'val',
            'test_used': False, 'time_rule': 'history.end_ts < prediction.start_ts',
            'history_protocol': 'rolling, preceding completed outcomes within val are available',
            'ndcg': NDCG, 'ndcg_ties': 'fixed-seed random permutation, exact score ties only',
            'query_auc': 'QueryAUC:type=Ranking; tied scores count as 0.5',
            'common_params': common, 'models': {},
            'versions': {'catboost': catboost.__version__, 'numpy': np.__version__,
                         'pandas': pd.__version__, 'pyarrow': pa.__version__, 'sklearn': sklearn.__version__},
            'pool_configs': {split: json.loads((CACHE / f'{split}_config.json').read_text())
                             for split in ('train', 'val')},
            'limitations': ['User-day observed-event ranking; per-event cutoffs, not whole catalogue.',
                            'Fixed configurations, no hyperparameter or feature search.',
                            'Comparison uses the manual baseline of the same replay run.'],
            'source_sha256': sha256(Path(__file__))}


def restore_previous_results(config):
    """Разрешает продолжение только при совпадении признаков и хешей Pool."""
    path = RESULTS / 'run_config.json'
    if not path.exists():
        return []
    previous = json.loads(path.read_text())
    assert previous['features'] == config['features']
    for split in ('train', 'val'):
        assert previous['pool_configs'][split]['pool_sha256'] == config['pool_configs'][split]['pool_sha256']
    config['models'] = previous['models']
    config['native_ranker_ndcg'] = previous.get('native_ranker_ndcg')
    config['previous_source_sha256'] = previous['source_sha256']
    return pd.read_csv(RESULTS / 'validation_metrics.csv').to_dict('records')


def completed_model(kind, path, config):
    """Проверяет, что существующий файл модели имеет завершенный отчет и хеш."""
    if not path.exists():
        return False
    assert kind in config['models'], f'Есть файл без завершенных результатов: {path}'
    assert sha256(path) == config['models'][kind]['sha256']
    print('REUSE_COMPLETED', kind, flush=True)
    return True


def fit_and_save(kind, model, train, val, path):
    """Измеряет только model.fit и сохраняет модель и кривую обучения."""
    print('FIT', kind, flush=True)
    started = time.perf_counter()
    model.fit(train, eval_set=val)
    elapsed = time.perf_counter() - started
    model.save_model(str(path))
    write_json(RESULTS / f'{kind}_learning_curve.json', model.get_evals_result())
    return elapsed


def evaluate_model(kind, model, val, labels, groups, config):
    """Получает score тем же способом и сохраняет прогнозы для сравнения."""
    if kind == 'ranker':
        score = model.predict(val, thread_count=THREADS)
        metrics = ranking_metrics(labels, groups, score)
        metrics['logloss'] = float('nan')
        config['native_ranker_ndcg'] = eval_metric(labels, score, NDCG, group_id=groups, thread_count=THREADS)[0]
    else:
        probabilities = model.predict_proba(val, thread_count=THREADS)
        score = probabilities @ model.classes_.astype(float)
        metrics = ranking_metrics(labels, groups, score)
        metrics['logloss'] = log_loss(labels, probabilities, labels=model.classes_)
        np.save(RESULTS / 'pointwise_val_proba.npy', probabilities)
    np.save(RESULTS / f'{kind}_val_score.npy', score)
    return metrics


def verify_reload(model, path, val, groups):
    """Проверяет точное совпадение прогнозов после загрузки на целых группах."""
    end = int(np.searchsorted(groups, groups[min(2047, len(groups) - 1)], side='right'))
    probe = val.slice(np.arange(end))
    loaded = type(model)().load_model(str(path))
    np.testing.assert_allclose(model.predict(probe), loaded.predict(probe), rtol=0, atol=0)


def model_record(model, path, fit_seconds, metrics):
    """Фиксирует параметры модели, время fit и проверенные значения метрик."""
    return {'path': str(path), 'sha256': sha256(path), 'fit_seconds': fit_seconds,
            'training_threads': model.get_params()['thread_count'],
            'training_source_sha256': sha256(Path(__file__)), 'best_iteration': model.get_best_iteration(),
            'tree_count': model.tree_count_, 'actual_params': model.get_all_params(),
            'reload_predictions_exact': True,
            'validation_metrics': {name: value for name, value in metrics.items() if np.isfinite(value)}}


def save_results(kind, model, path, fit_seconds, metrics, config, rows, baseline):
    """Сохраняет важности, отчет модели и прежние сравнительные CSV."""
    model.get_feature_importance(type='PredictionValuesChange', prettified=True).to_csv(
        RESULTS / f'{kind}_feature_importance.csv', index=False)
    config['models'][kind] = model_record(model, path, fit_seconds, metrics)
    label = 'DFS YetiRank' if kind == 'ranker' else 'DFS MultiClass E[label]'
    rows.append({'model': label, **metrics})
    pd.concat([baseline, pd.DataFrame(rows)], ignore_index=True).to_csv(RESULTS / 'validation_comparison.csv', index=False)
    pd.DataFrame(rows).to_csv(RESULTS / 'validation_metrics.csv', index=False)
    write_json(RESULTS / 'run_config.json', config)
    print('MODEL_DONE', kind, metrics, 'seconds', fit_seconds, flush=True)


def train_one_model(kind, model, train, val, labels, groups, config, rows, baseline):
    """Обучает отсутствующую модель, проверяет загрузку и сохраняет результаты."""
    path = MODELS / f'{kind}.cbm'
    if completed_model(kind, path, config):
        return
    fit_seconds = fit_and_save(kind, model, train, val, path)
    metrics = evaluate_model(kind, model, val, labels, groups, config)
    verify_reload(model, path, val, groups)
    save_results(kind, model, path, fit_seconds, metrics, config, rows, baseline)
    release()


def train_models():
    """Загружает Pool и выполняет прежние два обучения с проверкой результатов."""
    train, val, labels, groups = load_training_data()
    common = common_parameters()
    models = build_models(common)
    config = training_config(train.get_feature_names(), common)
    rows = restore_previous_results(config)
    config['training_threads_by_model'] = {'ranker': 4, 'pointwise': 8}
    baseline_dir = (ROOT / 'results/baselines/timing-20261020' if RUN_NAME is None
                    else ROOT / 'results/baselines/runs' / RUN_NAME)
    baseline = pd.read_csv(baseline_dir / 'validation_metrics.csv')
    for kind, model in models:
        train_one_model(kind, model, train, val, labels, groups, config, rows, baseline)
    return config


def build_parser():
    """Описывает прежний CLI подготовки Pool и обучения моделей."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'train'], required=True)
    parser.add_argument('--split', choices=['train', 'val'])
    return parser


def monitor_resources(stop, resources):
    """Отслеживает RSS процесса, доступную RAM и системный swap."""
    process = psutil.Process()
    while not stop.is_set():
        resources['peak_rss_bytes'] = max(resources['peak_rss_bytes'], process.memory_info().rss)
        resources['min_system_available_bytes'] = min(resources['min_system_available_bytes'], psutil.virtual_memory().available)
        resources['peak_system_swap_bytes'] = max(resources['peak_system_swap_bytes'], psutil.swap_memory().used)
        stop.wait(0.5)


def run_stage(args):
    """Выполняет этап и сохраняет отчет о времени, ресурсах и ошибках."""
    started, stop = time.perf_counter(), threading.Event()
    resources = {'peak_rss_bytes': 0, 'min_system_available_bytes': psutil.virtual_memory().available,
                 'initial_swap_bytes': psutil.swap_memory().used, 'peak_system_swap_bytes': 0}
    threading.Thread(target=monitor_resources, args=(stop, resources), daemon=True).start()
    report = {'stage': args.stage, 'split': args.split, 'completed': False}
    try:
        prepare_pool(args.split) if args.stage == 'prepare' else train_models()
        report['completed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        stop.set()
        report.update(elapsed_seconds=time.perf_counter() - started, resources=resources,
                      max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        write_json(RESULTS / f'{args.stage}_{args.split or "models"}_report.json', report)
        print('STAGE_END', report, flush=True)


def protect_recorded_run(args):
    """Запрещает CLI изменять сохраненные канонические модели и Pool."""
    if RUN_NAME is not None:
        return
    paths = ([RESULTS / 'run_config.json', MODELS / 'ranker.cbm', MODELS / 'pointwise.cbm']
             if args.stage == 'train' else [CACHE / f'{args.split}.qbin', CACHE / f'{args.split}_config.json'])
    if any(path.exists() for path in paths):
        raise FileExistsError('Сохраненный канонический расчет защищен. '
                              'Для нового расчета задайте TLAB_EXPERIMENT_RUN=my_run '
                              'при генерации признаков, подготовке Pool и обучении.')


def main(argv=None):
    """Проверяет CLI и запускает этап только при явном вызове скрипта."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage == 'prepare' and args.split is None:
        parser.error('Для prepare нужен --split')
    protect_recorded_run(args)
    for directory in (CACHE, MODELS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    run_stage(args)


if __name__ == '__main__':
    main()
