"""Полные train/val без хранения всей широкой матрицы в RAM.

История каждой сущности остается в одном блоке. Ветви сначала записываются
в дисковые матрицы, затем итоговый Parquet собирается порциями строк.
"""
from pathlib import Path
from dataclasses import dataclass
import argparse
import gc
import hashlib
import json
import os
import re
import resource
import shutil
import sys
import threading
import time

import featuretools as ft
import numba
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cumulative_features import calculate_cumulative, accumulate_history

DAY = 86400
TRAIN_START, VAL_START, TEST_START, GAP = 225 * DAY, 270 * DAY, 285 * DAY, 1800
HISTORY_CHUNK = 500_000
WRITE_CHUNK = 100_000
BRANCHES = [('Users', 'uid'), ('Items', 'item_id'),
            ('UserItem', 'user_item_id'), ('UserArtist', 'user_artist_id')]
SOURCE_COLUMNS = ['uid', 'item_id', 'artist_id', 'start_ts', 'timestamp',
                  'label', 'track_length_seconds', 'is_organic']
BASE_COLUMNS = ['log_id', 'artist_id', 'is_organic', 'track_length_seconds']


@dataclass
class GenerationRun:
    """Пути, границы времени и отчет одного запуска генерации."""
    args: argparse.Namespace
    run_dir: Path
    output: Path
    scratch: Path
    report_path: Path
    query_start: int
    query_end: int
    started: float
    stop_monitor: threading.Event
    resources: dict
    report: dict


def build_parser():
    """Описывает прежние параметры командной строки без запуска расчета."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--assemble-only', action='store_true',
                        help='Собрать готовые ветви после прерванной сборки')
    return parser


def experiment_run_name():
    """Проверяет имя отдельного запуска, исключая выход за его каталоги."""
    name = os.environ.get('TLAB_EXPERIMENT_RUN')
    if name is not None and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) is None:
        raise ValueError('TLAB_EXPERIMENT_RUN: нужны 1–64 буквы/цифры/_/-, начиная с буквы или цифры')
    return name


def initial_report(args):
    """Создает описание протокола и незавершенного запуска."""
    return {'split': args.split, 'completed': False, 'query_limit': args.limit,
            'history_chunk_rows': HISTORY_CHUNK, 'write_chunk_rows': WRITE_CHUNK,
            'branches': {}, 'timings': {},
            'time_rule': 'history.end_ts < prediction.start_ts',
            'constant_groups_filtered': False,
            'versions': {'featuretools': ft.__version__, 'pandas': pd.__version__,
                         'numpy': np.__version__, 'numba': numba.__version__},
            'engine_sha256': hashlib.sha256((ROOT / 'src/cumulative_features.py').read_bytes()).hexdigest()}


def run_directories(args):
    """Выбирает отдельные каталоги данных и служебных отчетов запуска."""
    name = experiment_run_name()
    if name is None:
        return (ROOT / 'results/local/feature_generation' / args.split,
                ROOT / 'data/experiment-1/featuretools')
    return (ROOT / f'results/experiment-1/runs/{name}/feature_generation/{args.split}',
            ROOT / f'data/experiment-1/runs/{name}/featuretools')


def create_run(args):
    """Готовит каталоги запуска, не разрешая перезапись готового результата."""
    run_dir, data = run_directories(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / 'smoke' if args.limit is not None else data / f'full_{args.split}_cumulative'
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'SUCCESS.json').exists(), 'Расчет уже готов; для нового задайте TLAB_EXPERIMENT_RUN=my_run'
    scratch = output / 'scratch'
    scratch.mkdir(exist_ok=True)
    bounds = (TRAIN_START, VAL_START) if args.split == 'train' else (VAL_START, TEST_START)
    resources = {'sampled_peak_rss_bytes': 0,
                 'min_available_bytes': psutil.virtual_memory().available,
                 'peak_system_swap_bytes': psutil.swap_memory().used}
    return GenerationRun(args, run_dir, output, scratch, output / 'run_report.json',
                         *bounds, time.perf_counter(), threading.Event(), resources,
                         initial_report(args))


def monitor(run):
    """Периодически записывает память процесса и доступную память системы."""
    process = psutil.Process()
    while not run.stop_monitor.is_set():
        stats = run.resources
        stats['sampled_peak_rss_bytes'] = max(stats['sampled_peak_rss_bytes'], process.memory_info().rss)
        stats['min_available_bytes'] = min(stats['min_available_bytes'], psutil.virtual_memory().available)
        stats['peak_system_swap_bytes'] = max(stats['peak_system_swap_bytes'], psutil.swap_memory().used)
        run.stop_monitor.wait(0.2)


def release():
    """Возвращает освобожденную память Python и Arrow операционной системе."""
    gc.collect()
    pa.default_memory_pool().release_unused()
    import ctypes
    ctypes.CDLL(None).malloc_trim(0)


def save_report(run):
    """Атомарно сохраняет текущий прогресс и измерения ресурсов."""
    run.report['elapsed_seconds'] = time.perf_counter() - run.started
    run.report['peak_rss_gib'] = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                                     run.report.get('peak_rss_gib', 0),
                                     run.resources['sampled_peak_rss_bytes'] / 1024**3)
    run.report['resources'] = dict(run.resources)
    temporary = run.report_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(run.report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(run.report_path)


def target_mask(table, query_start, query_end):
    """Оставляет события окна с тем же временным зазором до его конца."""
    starts_in_window = pc.and_(pc.greater_equal(table['start_ts'], query_start),
                               pc.less(table['start_ts'], query_end))
    return pc.and_(starts_in_window, pc.less(table['timestamp'], query_end - GAP))


def read_filtered_tables(run):
    """Читает узкие столбцы истории и целевые строки с исходными log_id."""
    history_parts, target_parts, offset = [], [], 0
    batches = pq.ParquetFile(ROOT / 'data/listens.parquet').iter_batches(
        batch_size=1_000_000, columns=SOURCE_COLUMNS)
    for batch in batches:
        table = pa.Table.from_batches([batch])
        table = table.append_column('log_id', pa.array(np.arange(offset, offset + len(batch), dtype='int64')))
        offset += len(batch)
        targets = table.filter(target_mask(table, run.query_start, run.query_end))
        if len(targets):
            target_parts.append(targets)
        past = table.filter(pc.less(table['timestamp'], run.query_end))
        if len(past):
            history_parts.append(past)
    return pa.concat_tables(history_parts), pa.concat_tables(target_parts), offset


def read_history_and_targets(run):
    """Переводит узкие таблицы в pandas после освобождения промежуточных блоков."""
    history_table, target_table, offset = read_filtered_tables(run)
    history = history_table.to_pandas(split_blocks=True, self_destruct=True)
    targets = target_table.to_pandas(split_blocks=True, self_destruct=True)
    return history, targets, offset


def validate_targets(targets, split):
    """Проверяет совпадение целевых строк с сохраненным разбиением samples."""
    expected = pq.read_table(ROOT / 'data/samples.parquet', columns=SOURCE_COLUMNS + ['group_id'],
                             filters=[('split', '==', split)]).to_pandas(split_blocks=True, self_destruct=True)
    assert len(targets) == len(expected) == {'train': 8_523_909, 'val': 3_091_912}[split]
    for column in SOURCE_COLUMNS:
        assert np.array_equal(targets[column].to_numpy(), expected[column].to_numpy()), column
    targets['group_id'] = expected['group_id'].to_numpy(copy=True)


def add_pair_keys(frame):
    """Кодирует пары user–item и user–artist; неизвестный артист остается NA."""
    uid = frame['uid'].to_numpy(dtype='int64')
    frame['user_item_id'] = (uid << 32) | frame['item_id'].to_numpy(dtype='int64')
    artist = frame['artist_id'].to_numpy(dtype='int64')
    keys = pd.array((uid << 32) | np.maximum(artist, 0), dtype='Int64')
    keys[artist < 0] = pd.NA
    frame['user_artist_id'] = keys


def prepare_data(run):
    """Проверяет выборку, выбирает smoke-поднабор и готовит ключи сущностей."""
    history, targets, source_rows = read_history_and_targets(run)
    validate_targets(targets, run.args.split)
    if run.args.limit is not None:
        targets = targets.sample(n=run.args.limit, random_state=42).sort_values('log_id').reset_index(drop=True)
    history.rename(columns={'timestamp': 'end_ts'}, inplace=True)
    targets.rename(columns={'timestamp': 'end_ts', 'start_ts': 'time'}, inplace=True)
    for frame in (history, targets):
        add_pair_keys(frame)
    history.drop(columns=['start_ts', 'artist_id'], inplace=True)
    release()
    return history, targets, source_rows


def sorted_queries(queries, key, matrix):
    """Возвращает позиции запросов в устойчивом порядке ключа сущности."""
    missing = queries[key].isna().to_numpy()
    matrix[missing, :] = np.nan
    positions = np.flatnonzero(~missing)
    keys = queries.iloc[positions][key].to_numpy(dtype='int64')
    order = np.argsort(keys, kind='stable')
    return positions[order], keys[order]


def sorted_history(history, key):
    """Сортирует индексы истории только тогда, когда ключи еще не упорядочены."""
    keys = history[key].to_numpy(dtype='int64', na_value=-1)
    if np.all(keys[1:] >= keys[:-1]):
        return None, keys
    order = np.argsort(keys, kind='stable')
    return order, keys[order]


def entity_blocks(sorted_keys):
    """Делит историю на блоки, сохраняя все строки одного ключа вместе."""
    left = 0
    while left < len(sorted_keys):
        proposed = min(left + HISTORY_CHUNK, len(sorted_keys))
        right = int(np.searchsorted(sorted_keys, sorted_keys[proposed - 1], side='right'))
        yield left, right
        left = right


def calculate_block(history, queries, parent, key, definitions, history_order, left, right):
    """Считает один блок и проверяет порядок его строк по log_id."""
    history_block = (history.iloc[left:right] if history_order is None
                     else history.iloc[history_order[left:right]])
    block = calculate_cumulative(history_block, queries, parent, key, definitions)
    assert block.index.equals(pd.Index(queries['log_id'], name='log_id'))
    return block.to_numpy(dtype='float64')


def branch_progress(run, parent, left, history_rows, processed, query_rows):
    """Печатает промежуточный размер расчета и сохраняет отчет."""
    print('PROGRESS', parent, 'history', left, '/', history_rows, 'queries', processed, '/', query_rows,
          'RSS_GiB', round(psutil.Process().memory_info().rss / 1024**3, 2), flush=True)
    save_report(run)


def fill_branch(run, history, queries, parent, key, definitions, matrix):
    """Заполняет дисковую матрицу ветви, проходя историю блоками сущностей."""
    positions, query_keys = sorted_queries(queries, key, matrix)
    history_order, history_keys = sorted_history(history, key)
    processed, last_log = 0, time.monotonic()
    for left, right in entity_blocks(history_keys):
        qlo = int(np.searchsorted(query_keys, history_keys[left], side='left'))
        qhi = int(np.searchsorted(query_keys, history_keys[right - 1], side='right'))
        if qhi > qlo:
            selected = positions[qlo:qhi]
            matrix[selected, :] = calculate_block(history, queries.iloc[selected], parent, key,
                                                   definitions, history_order, left, right)
            processed += len(selected)
        if time.monotonic() - last_log >= 15:
            branch_progress(run, parent, right, len(history), processed, len(positions))
            last_log = time.monotonic()
    assert processed == len(positions), (parent, processed, len(positions))
    matrix.flush()


def branch_tables(queries, parent, names, matrix):
    """Выдает порции признаков в исходном порядке целевых log_id."""
    for left in range(0, len(queries), WRITE_CHUNK):
        right = min(left + WRITE_CHUNK, len(queries))
        arrays = {'log_id': pa.array(queries['log_id'].to_numpy()[left:right])}
        arrays.update({parent + '__' + name: pa.array(matrix[left:right, j])
                       for j, name in enumerate(names)})
        yield pa.table(arrays)


def write_tables(path, tables):
    """Сохраняет поток таблиц в Parquet через временный файл."""
    temporary = path.with_suffix('.parquet.incomplete')
    writer = None
    try:
        for table in tables:
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='snappy', use_dictionary=False)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)


def write_branch(run, history, queries, parent, key, definitions):
    """Считает ветвь на диске и сохраняет ее признаки, формулы и время."""
    started = time.perf_counter()
    names = [feature.get_name() for feature in definitions]
    mapped_path = run.scratch / f'{parent}.npy'
    assert not mapped_path.exists(), mapped_path
    matrix = np.lib.format.open_memmap(mapped_path, mode='w+', dtype='float64',
                                      shape=(len(queries), len(names)))
    fill_branch(run, history, queries, parent, key, definitions, matrix)
    compute_seconds = time.perf_counter() - started
    release()
    write_started = time.perf_counter()
    path = run.output / f'{parent}.parquet'
    write_tables(path, branch_tables(queries, parent, names, matrix))
    del matrix
    mapped_path.unlink()
    release()
    record_branch(run, parent, definitions, path, len(queries), started, compute_seconds, write_started)
    return path


def record_branch(run, parent, definitions, path, rows, started, compute_seconds, write_started):
    """Проверяет сохраненную ветвь и записывает ее показатели в отчет."""
    assert pq.ParquetFile(path).metadata.num_rows == rows
    ft.save_features(definitions, str(run.output / f'{parent}_definitions.json'))
    run.report['branches'][parent] = {'rows': rows, 'features': len(definitions),
                                      'compute_seconds': compute_seconds,
                                      'write_seconds': time.perf_counter() - write_started,
                                      'total_seconds': time.perf_counter() - started,
                                      'parquet_bytes': path.stat().st_size}
    save_report(run)
    print('BRANCH_DONE', parent, run.report['branches'][parent], flush=True)


def read_golden(run):
    """Загружает независимый train-эталон или проверку val на том же движке."""
    path = (ROOT / 'tests/fixtures/dfs_reference_5000.parquet' if run.args.split == 'train'
            else run.output / 'validation_probe.parquet')
    return pq.read_table(path).to_pandas().set_index('log_id').sort_index()


def join_branch_batches(base_batch, iterators):
    """Соединяет очередные порции ветвей, проверяя совпадение log_id."""
    table = pa.Table.from_batches([base_batch])
    ids = table['log_id'].to_numpy()
    for iterator in iterators:
        branch = pa.Table.from_batches([next(iterator)])
        assert np.array_equal(ids, branch['log_id'].to_numpy())
        for name in branch.column_names[1:]:
            table = table.append_column(name, branch[name])
    return table


def check_golden_rows(table, golden):
    """Сверяет встретившиеся в порции эталонные строки с допуском 1e-9."""
    ids = table['log_id'].to_numpy()
    golden_ids = golden.index.to_numpy()
    positions = np.searchsorted(ids, golden_ids)
    valid = positions < len(ids)
    safe = np.minimum(positions, len(ids) - 1)
    valid &= ids[safe] == golden_ids
    if not valid.any():
        return 0
    actual = table.take(pa.array(positions[valid])).to_pandas().set_index('log_id')
    expected = golden.loc[golden_ids[valid]]
    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_allclose(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float, na_value=np.nan),
                               rtol=1e-9, atol=1e-9, equal_nan=True)
    return len(actual)


def assembled_tables(run, metadata_path, iterators, golden, counters):
    """Собирает широкие порции и считает число проверенных эталонных строк."""
    batches = pq.ParquetFile(metadata_path).iter_batches(batch_size=WRITE_CHUNK, columns=BASE_COLUMNS)
    last_log = time.monotonic()
    for batch in batches:
        table = join_branch_batches(batch, iterators)
        counters['checked'] += check_golden_rows(table, golden)
        counters['rows'] += len(table)
        yield table
        if time.monotonic() - last_log >= 15:
            print('ASSEMBLE', counters['rows'], 'rows', flush=True)
            last_log = time.monotonic()
            save_report(run)
    for iterator in iterators:
        assert next(iterator, None) is None


def verify_assembled(run, final, golden, counters):
    """Проверяет покрытие эталона, размер и чтение сохраненных групп строк."""
    expected = len(golden) if run.args.limit is None or run.args.split == 'val' else run.args.limit
    assert counters['checked'] == expected, (counters['checked'], expected)
    parquet = pq.ParquetFile(final)
    assert parquet.metadata.num_rows == counters['rows'] == run.report['query_rows']
    assert len(parquet.schema_arrow.names) == 132
    for index in sorted({0, parquet.metadata.num_row_groups // 2, parquet.metadata.num_row_groups - 1}):
        group = parquet.read_row_group(index)
        assert group.num_rows > 0 and group.num_columns == 132
    run.report['golden_comparison_rows'] = counters['checked']
    run.report['saved_row_groups_read_checked'] = 3
    run.report['output_bytes'] = final.stat().st_size


def assemble_output(run, metadata_path, branch_paths):
    """Собирает ветви с диска и сверяет эталонные или проверочные строки."""
    final = run.output / f'{run.args.split}_all_cumulative_X.parquet'
    assert not final.exists(), final
    iterators = [pq.ParquetFile(path).iter_batches(batch_size=WRITE_CHUNK) for path in branch_paths]
    golden = read_golden(run)
    counters = {'rows': 0, 'checked': 0}
    write_tables(final, assembled_tables(run, metadata_path, iterators, golden, counters))
    verify_assembled(run, final, golden, counters)
    return final


def prepare_queries(run):
    """Сохраняет узкие метаданные целевых строк и время их подготовки."""
    started = time.perf_counter()
    history, queries, source_rows = prepare_data(run)
    run.report.update(query_rows=len(queries), history_rows=len(history), source_rows=source_rows,
                      targets_match_samples=True)
    run.report['timings']['prepare_seconds'] = time.perf_counter() - started
    print('DATA_READY', 'history', len(history), 'train', len(queries), flush=True)
    metadata_path = run.output / f'{run.args.split}_all_cumulative_metadata.parquet'
    queries.to_parquet(metadata_path, index=False)
    return history, queries, metadata_path


def warm_up_engine(run):
    """Компилирует накопительный расчет на крошечном входе и учитывает время."""
    started = time.perf_counter()
    accumulate_history(np.array([1], dtype='int64'), np.array([0.]), np.array([[1.]]),
                       np.array([1], dtype='int64'), np.array([1.]))
    run.report['timings']['jit_seconds'] = time.perf_counter() - started
    save_report(run)


def validation_probe(run, queries):
    """Выбирает запросы для сверки блочного расчета val тем же движком."""
    if run.args.split != 'val':
        return None, None
    probe = queries.sample(n=min(100, len(queries)), random_state=42).sort_values('log_id')
    return probe, probe[BASE_COLUMNS].set_index('log_id')


def generate_branches(run, history, queries):
    """Считает ветви и контрольный val-probe тем же кумулятивным движком."""
    paths = []
    probe, probe_x = validation_probe(run, queries)
    for parent, key in BRANCHES:
        print('BRANCH_START', parent, flush=True)
        definitions = ft.load_features(str(ROOT / f'data/experiment-1/featuretools/train_sample_200_cumulative_{parent}_definitions.json'))
        if probe_x is not None:
            features = calculate_cumulative(history, probe, parent, key, definitions).add_prefix(parent + '__')
            probe_x = probe_x.join(features, validate='one_to_one')
        paths.append(write_branch(run, history, queries, parent, key, definitions))
    if probe_x is not None:
        probe_x.rename_axis('log_id').reset_index().to_parquet(run.output / 'validation_probe.parquet', index=False)
    return paths


def complete_run(run, metadata_path, paths):
    """Собирает итоговый файл и отмечает завершенный расчет маркером SUCCESS."""
    started = time.perf_counter()
    final = assemble_output(run, metadata_path, paths)
    run.report['timings']['assemble_and_verify_seconds'] = time.perf_counter() - started
    run.report.update(completed=True, output=str(final))
    save_report(run)
    success = {'rows': run.report['query_rows'], 'features': 131, 'output': str(final)}
    (run.output / 'SUCCESS.json').write_text(json.dumps(success, indent=2) + '\n')


def resume_assembly(run):
    """Продолжает только сборку уже сохраненных ветвей с прежним отчетом."""
    previous = json.loads(run.report_path.read_text())
    run.report.update(previous)
    run.report.pop('error', None)
    run.started -= previous['elapsed_seconds']
    run.resources.update(previous['resources'])
    metadata = run.output / f'{run.args.split}_all_cumulative_metadata.parquet'
    paths = [run.output / f'{parent}.parquet' for parent, key in BRANCHES]
    complete_run(run, metadata, paths)
    print('ASSEMBLY_FINISHED', json.dumps(run.report, ensure_ascii=False), flush=True)


def generate_all(run):
    """Выполняет подготовку, расчет ветвей и сборку без полной матрицы в RAM."""
    history, queries, metadata = prepare_queries(run)
    warm_up_engine(run)
    paths = generate_branches(run, history, queries)
    del history, queries
    release()
    complete_run(run, metadata, paths)
    shutil.copyfile(run.report_path, run.run_dir / ('smoke_report.json' if run.args.limit is not None else 'report.json'))
    print('FINISHED', json.dumps(run.report, ensure_ascii=False), flush=True)


def main(argv=None):
    """Запускает расчет только при явном вызове скрипта."""
    run = create_run(build_parser().parse_args(argv))
    threading.Thread(target=monitor, args=(run,), daemon=True).start()
    try:
        if run.args.assemble_only:
            resume_assembly(run)
        else:
            try:
                generate_all(run)
            except Exception as error:
                run.report['error'] = repr(error)
                save_report(run)
                raise
    finally:
        run.stop_monitor.set()


if __name__ == '__main__':
    main()
