"""Фиксированные DFS-признаки для произвольного временного окна.

История остается компактной. Каждая ветвь считается блоками целых сущностей,
пишется через memmap, а итоговый Parquet собирается порциями по строкам.
"""
from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
import json
import resource
import time
import uuid

import featuretools as ft
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cumulative_features import calculate_cumulative

ROOT = Path(__file__).resolve().parents[1]
HISTORY_CHUNK = 500_000
WRITE_CHUNK = 100_000
BRANCHES = [('Users', 'uid'), ('Items', 'item_id'),
            ('UserItem', 'user_item_id'), ('UserArtist', 'user_artist_id')]
HISTORY_COLUMNS = ['uid', 'item_id', 'artist_id', 'timestamp', 'label',
                   'track_length_seconds', 'is_organic']
QUERY_COLUMNS = ['log_id', 'uid', 'item_id', 'artist_id', 'start_ts',
                 'is_organic', 'track_length_seconds']
REQUIRED_COLUMNS = QUERY_COLUMNS + ['timestamp', 'label', 'group_id', 'split']


@dataclass
class _Run:
    """Общие настройки, маленький контрольный набор и журнал одного расчета."""
    queries: pd.DataFrame
    output: Path
    prefix: str
    features: list
    definitions: dict
    probe_positions: np.ndarray
    probe: pd.DataFrame
    report: dict
    started: float


def _sha256(path):
    """Получить контрольную сумму небольшого файла с кодом или формулами."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_memory():
    """Освободить уже ненужные Python- и Arrow-буферы между ветвями."""
    gc.collect()
    pa.default_memory_pool().release_unused()


def _check_integer_column(frame, name, lower=None, upper=None):
    """Проверить целочисленное поле без пропусков и допустимые границы."""
    values = frame[name]
    if values.isna().any() or values.dtype.kind not in 'iu':
        raise ValueError(f'{name}: нужны целые значения без пропусков')
    if lower is not None and (values < lower).any():
        raise ValueError(f'{name}: значения должны быть >= {lower}')
    if upper is not None and (values > upper).any():
        raise ValueError(f'{name}: значения должны быть <= {upper}')


def _validate_targets(targets):
    """Проверить схему запросов, идентификаторы и конечное время предсказания."""
    if not isinstance(targets, pd.DataFrame) or targets.empty:
        raise ValueError('targets должен быть непустой pandas.DataFrame')
    missing = set(REQUIRED_COLUMNS).difference(targets.columns)
    if missing or not targets.columns.is_unique:
        raise ValueError(f'Некорректные столбцы targets; отсутствуют: {sorted(missing)}')
    for name in ['log_id', 'start_ts', 'timestamp', 'label', 'group_id',
                 'is_organic', 'track_length_seconds', 'artist_id']:
        _check_integer_column(targets, name)
    _check_integer_column(targets, 'uid', 0, 2 ** 31 - 1)
    _check_integer_column(targets, 'item_id', 0, 2 ** 32 - 1)
    _check_integer_column(targets, 'artist_id', upper=2 ** 32 - 1)
    if not targets['log_id'].is_unique:
        raise ValueError('log_id запросов должен быть уникальным')


def _load_specification():
    """Загрузить список модели и зафиксированные на исходном train формулы."""
    config = ROOT / 'results/experiment-1/dfs/run_config.json'
    features = json.loads(config.read_text())['features']
    definitions, hashes = {}, {'feature_config': _sha256(config)}
    for parent, key in BRANCHES:
        path = ROOT / f'data/experiment-1/featuretools/train_sample_200_cumulative_{parent}_definitions.json'
        definitions[parent] = ft.load_features(str(path))
        hashes[parent] = _sha256(path)
        if len(definitions[parent]) != 32:
            raise ValueError(f'{parent}: ожидались 32 фиксированные формулы')
    expected = ['is_organic', 'track_length_seconds'] + [
        parent + '__' + f.get_name() for parent, key in BRANCHES for f in definitions[parent]]
    if features != expected or len(features) != 130 or len(set(features)) != 130:
        raise ValueError('Фиксированные формулы не совпадают с 130 признаками модели')
    return features, definitions, hashes


def _output_paths(output):
    """Подготовить новые конечные пути, не перезаписывая результаты запуска."""
    output = Path(output).resolve()
    report = output.with_suffix('.report.json')
    probe = output.with_suffix('.probe.parquet')
    for path in [output, report, probe]:
        if path.exists():
            raise FileExistsError(f'Результат уже существует: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _control_sample(queries):
    """Выбрать до 100 контрольных запросов с прежним seed и порядком строк."""
    generator = np.random.default_rng(42)
    positions = np.sort(generator.choice(len(queries), min(100, len(queries)), replace=False))
    columns = ['log_id', 'is_organic', 'track_length_seconds']
    probe = queries.iloc[positions][columns].set_index('log_id')
    return positions, probe


def _prepare_run(targets, output_path):
    """Проверить вход, загрузить фиксированные формулы и подготовить контрольные строки."""
    started = time.perf_counter()
    _validate_targets(targets)
    features, definitions, hashes = _load_specification()
    output = _output_paths(output_path)
    queries = targets[QUERY_COLUMNS].rename(columns={'start_ts': 'time'}, copy=False)
    queries.index = pd.RangeIndex(len(queries))
    positions, probe = _control_sample(queries)
    report = {'completed': False, 'query_rows': len(queries), 'features': features,
              'feature_count': len(features), 'definitions_sha256': hashes,
              'time_rule': 'history.end_ts < prediction.start_ts', 'branches': {}}
    return _Run(queries, output, f'.{output.name}.dfs-{uuid.uuid4().hex}',
                features, definitions, positions, probe, report, started)


def _read_history_table(parquet, cutoff):
    """Собрать историю до cutoff порциями; для пустой истории сохранить исходную схему."""
    parts = []
    for batch in parquet.iter_batches(batch_size=1_000_000, columns=HISTORY_COLUMNS):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.less(table['timestamp'], cutoff))
        if len(table):
            parts.append(table)
    if parts:
        return pa.concat_tables(parts)
    return pa.table({
        name: pa.array([], type=parquet.schema_arrow.field(name).type)
        for name in HISTORY_COLUMNS
    })


def _load_history(cutoff):
    """Загрузить компактную историю и записать границу ее предварительного отбора."""
    source = ROOT / 'data/listens.parquet'
    parquet = pq.ParquetFile(source)
    table = _read_history_table(parquet, cutoff)
    history = table.to_pandas(split_blocks=True, self_destruct=True)
    history.rename(columns={'timestamp': 'end_ts'}, inplace=True)
    return history, {'path': str(source), 'source_rows': parquet.metadata.num_rows,
                     'history_rows': len(history), 'history_cutoff_exclusive': int(cutoff)}


def _packed_key(frame, attribute, nullable=False):
    """Создать точный int64-ключ пары; неизвестного артиста оставить без ключа."""
    uid = frame['uid'].to_numpy(dtype='int64')
    value = frame[attribute].to_numpy(dtype='int64')
    keys = (uid << 32) | np.maximum(value, 0)
    if attribute != 'artist_id':
        return keys
    keys[value < 0] = -1
    if not nullable:
        return keys
    result = pd.array(keys, dtype='Int64', copy=False)
    result[value < 0] = pd.NA
    return result


def _attach_key(history, queries, parent, key):
    """Добавить составной ключ только на время соответствующей ветви."""
    if parent not in {'UserItem', 'UserArtist'}:
        return False
    attribute = 'item_id' if parent == 'UserItem' else 'artist_id'
    history[key] = _packed_key(history, attribute)
    queries[key] = _packed_key(queries, attribute, nullable=True)
    return True


def _ordered_queries(queries, key):
    """Упорядочить позиции запросов по ключу, не меняя исходные строки."""
    positions = np.flatnonzero(queries[key].notna().to_numpy())
    keys = queries.iloc[positions][key].to_numpy(dtype='int64')
    order = np.argsort(keys, kind='stable')
    return positions[order], keys[order]


def _query_layout(queries, key):
    """Разместить valid-запросы по ключу, пропуски в конце; вернуть обратный порядок."""
    positions, keys = _ordered_queries(queries, key)
    missing = np.flatnonzero(queries[key].isna().to_numpy())
    inverse = np.empty(len(queries), dtype='int64')
    inverse[positions] = np.arange(len(positions), dtype='int64')
    inverse[missing] = np.arange(len(positions), len(queries), dtype='int64')
    return positions, keys, inverse


def _ordered_history(history, key):
    """Отсортировать только индексы истории, сохранив целиком компактную таблицу."""
    keys = history[key].to_numpy(dtype='int64', na_value=-1)
    if np.all(keys[1:] >= keys[:-1]):
        return None, keys
    order = np.argsort(keys, kind='stable')
    return order, keys[order]


def _entity_blocks(keys):
    """Сформировать блоки около 500 тысяч строк без разрыва одной сущности."""
    left = 0
    while left < len(keys):
        proposed = min(left + HISTORY_CHUNK, len(keys))
        right = int(np.searchsorted(keys, keys[proposed - 1], side='right'))
        yield left, right
        left = right


def _work_path(run, name):
    """Получить уникальный путь временного артефакта только текущего запуска."""
    return run.output.parent / f'{run.prefix}.{name}'


def _initialize_matrix(run, parent, names, n_valid):
    """Последовательно подготовить дисковую ветвь, оставив пропуски ключа в конце."""
    path = _work_path(run, f'{parent}.npy')
    matrix = np.lib.format.open_memmap(path, mode='w+', dtype='float64',
                                     shape=(len(run.queries), len(names)))
    count_column = names.index('COUNT(Interactions)')
    for left in range(0, len(matrix), WRITE_CHUNK):
        right = min(left + WRITE_CHUNK, len(matrix))
        matrix[left:right] = np.nan
        valid_right = max(left, min(right, n_valid))
        matrix[left:valid_right, count_column] = 0.0
    return path, matrix


def _calculate_block(history, queries, parent, key, definitions):
    """Посчитать блок и убедиться, что результат соответствует порядку его запросов."""
    block = calculate_cumulative(history, queries, parent, key, definitions)
    if not block.index.equals(pd.Index(queries['log_id'], name='log_id')):
        raise AssertionError(f'{parent}: нарушен порядок log_id')
    return block


def _fill_matrix(history, run, parent, key, definitions, matrix, layout):
    """Заполнять матрицу блоками целых сущностей, сохраняя порядок запросов по ключу."""
    positions, query_keys, inverse = layout
    history_order, history_keys = _ordered_history(history, key)
    processed = 0
    for left, right in _entity_blocks(history_keys):
        query_left = np.searchsorted(query_keys, history_keys[left], side='left')
        query_right = np.searchsorted(query_keys, history_keys[right - 1], side='right')
        if query_right <= query_left:
            continue
        selected = positions[query_left:query_right]
        queries = run.queries.iloc[selected]
        block_history = (history.iloc[left:right] if history_order is None
                         else history.iloc[history_order[left:right]])
        block = _calculate_block(block_history, queries, parent, key, definitions)
        matrix[query_left:query_right] = block.to_numpy(dtype='float64')
        processed += len(block)
        del block, queries, block_history, selected
    return processed


def _write_tables(path, tables):
    """Записать последовательность таблиц с одинаковой схемой без большой копии."""
    writer = None
    try:
        for table in tables:
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression='snappy', use_dictionary=False)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _branch_tables(run, matrix, parent, names, inverse):
    """Вернуть исходный порядок строк случайным чтением, не меняя дисковую матрицу."""
    ids = run.queries['log_id'].to_numpy()
    for left in range(0, len(ids), WRITE_CHUNK):
        right = min(left + WRITE_CHUNK, len(ids))
        values = matrix[inverse[left:right]]
        arrays = {'log_id': pa.array(ids[left:right])}
        arrays.update({parent + '__' + name: pa.array(values[:, j])
                       for j, name in enumerate(names)})
        yield pa.table(arrays)


def _add_branch_probe(history, run, parent, key, definitions):
    """Дополнить контрольный набор прямым расчетом текущей ветви."""
    queries = run.queries.iloc[run.probe_positions]
    expected = calculate_cumulative(history, queries, parent, key, definitions)
    run.probe = run.probe.join(expected.add_prefix(parent + '__'), validate='one_to_one')


def _export_branch_matrix(run, parent, names, matrix, inverse):
    """Сохранить заполненную ветвь в Parquet, вернув исходный порядок запросов."""
    matrix.flush()
    path = _work_path(run, f'{parent}.parquet')
    _write_tables(path, _branch_tables(run, matrix, parent, names, inverse))
    return path


def _calculate_branch(history, run, parent, key):
    """Проверить контрольные строки, заполнить дисковую матрицу и сохранить ветвь."""
    started = time.perf_counter()
    definitions = run.definitions[parent]
    names = [feature.get_name() for feature in definitions]
    layout = _query_layout(run.queries, key)
    mapped_path, matrix = _initialize_matrix(run, parent, names, len(layout[0]))
    _add_branch_probe(history, run, parent, key, definitions)
    processed = _fill_matrix(history, run, parent, key, definitions, matrix, layout)
    path = _export_branch_matrix(run, parent, names, matrix, layout[2])
    del matrix
    mapped_path.unlink()
    run.report['branches'][parent] = {'features': len(names), 'rows': len(run.queries),
        'processed_queries': processed, 'unassigned_cold_queries': len(layout[0]) - processed,
        'seconds': time.perf_counter() - started, 'parquet_bytes': path.stat().st_size}
    return path


def _generate_branches(history, run):
    """Обработать четыре ветви с одними и теми же зафиксированными формулами."""
    paths = []
    for parent, key in BRANCHES:
        print(f'DFS: {parent}, запросов {len(run.queries):,}', flush=True)
        added = _attach_key(history, run.queries, parent, key)
        paths.append(_calculate_branch(history, run, parent, key))
        if added:
            history.drop(columns=key, inplace=True)
            run.queries.drop(columns=key, inplace=True)
        _release_memory()
    return paths


def _append_branch_batch(table, batch):
    """Проверить log_id очередной порции и добавить ее столбцы к общей таблице."""
    if batch is None or not np.array_equal(table['log_id'].to_numpy(), batch.column(0).to_numpy()):
        raise AssertionError('Нарушены длина или порядок строк ветви')
    branch = pa.Table.from_batches([batch])
    for name in branch.column_names[1:]:
        table = table.append_column(name, branch[name])
    return table


def _joined_batch(run, left, iterators):
    """Соединить четыре порции по проверенному порядку log_id."""
    columns = ['log_id', 'is_organic', 'track_length_seconds']
    base = run.queries.iloc[left:left + WRITE_CHUNK][columns]
    table = pa.Table.from_pandas(base, preserve_index=False)
    for iterator in iterators:
        table = _append_branch_batch(table, next(iterator, None))
    return table.select(['log_id'] + run.features)


def _check_probe_batch(run, table, left):
    """Сверить случайные контрольные строки с прямым calculate_cumulative."""
    mask = (run.probe_positions >= left) & (run.probe_positions < left + len(table))
    if not mask.any():
        return 0
    actual = table.take(pa.array(run.probe_positions[mask] - left)).to_pandas().set_index('log_id')
    expected = run.probe.iloc[np.flatnonzero(mask)][run.features]
    if not actual.index.equals(expected.index):
        raise AssertionError('Нарушен порядок log_id контрольных строк')
    np.testing.assert_allclose(actual.to_numpy(dtype='float64'), expected.to_numpy(dtype='float64'),
                               rtol=1e-9, atol=1e-9, equal_nan=True)
    return len(actual)


def _assembled_tables(run, branch_paths):
    """Выдавать итоговые порции и проверять их схему, порядок и probe."""
    iterators = [pq.ParquetFile(path).iter_batches(batch_size=WRITE_CHUNK) for path in branch_paths]
    checked = 0
    for left in range(0, len(run.queries), WRITE_CHUNK):
        table = _joined_batch(run, left, iterators)
        checked += _check_probe_batch(run, table, left)
        yield table
    if any(next(iterator, None) is not None for iterator in iterators):
        raise AssertionError('Ветви содержат лишние строки')
    if checked != len(run.probe_positions):
        raise AssertionError('Не все контрольные строки проверены')
    run.report.update(probe_comparison_rows=checked, probe_equal=True, row_order_checked=True)


def _assemble_output(run, branch_paths):
    """Собрать окончательный Parquet порциями и проверить сохраненную схему."""
    temporary = _work_path(run, 'output.incomplete')
    _write_tables(temporary, _assembled_tables(run, branch_paths))
    parquet = pq.ParquetFile(temporary)
    if parquet.metadata.num_rows != len(run.queries) or parquet.schema_arrow.names != ['log_id'] + run.features:
        raise AssertionError('Сохраненный файл имеет неверные число строк или схему')
    temporary.replace(run.output)


def _finish_run(run, branch_paths):
    """Сохранить контрольный набор и отчет, затем удалить только свои ветви."""
    probe_path = run.output.with_suffix('.probe.parquet')
    run.probe[run.features].reset_index().to_parquet(probe_path, index=False)
    run.report.update(completed=True, output=str(run.output), probe_path=str(probe_path),
        output_bytes=run.output.stat().st_size, elapsed_seconds=time.perf_counter() - run.started,
        process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
    report_path = run.output.with_suffix('.report.json')
    report_path.write_text(json.dumps(run.report, ensure_ascii=False, indent=2) + '\n')
    for path in branch_paths:
        path.unlink()
    return run.report


def generate_dfs_features(targets: pd.DataFrame, output_path: Path) -> dict:
    """Сохранить log_id и прежние 130 DFS-признаков для заданных target-строк.

    Формулы не выбираются заново. Для каждой строки доступна вся история
    завершенных событий со строгим end_ts < start_ts. Порядок targets сохранен.
    Рядом сохраняются .probe.parquet на 100 строках и .report.json.
    """
    run = _prepare_run(targets, output_path)
    prepare_started = time.perf_counter()
    history, source = _load_history(int(run.queries['time'].max()) + 1)
    run.report['history'] = source
    run.report['prepare_seconds'] = time.perf_counter() - prepare_started
    run.report['matrix_layout'] = 'query_entity_sorted; missing_last; contiguous_block_writes; inverse_export'
    run.report['engine_sha256'] = _sha256(ROOT / 'src/cumulative_features.py')
    run.report['generator_sha256'] = _sha256(Path(__file__))
    branch_paths = _generate_branches(history, run)
    del history
    _release_memory()
    _assemble_output(run, branch_paths)
    return _finish_run(run, branch_paths)
