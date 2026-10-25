"""Общий порядок test-событий и проверка замороженного плана."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from generate_cumulative import SOURCE_COLUMNS, add_pair_keys
from train_dfs_catboost import select_mixed_groups
from temporal_sample_data import _validate_queries
from replay_paths import saved_path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / 'results/test_plans/final_test_20261023_v1/plan.json'


def sha256(path):
    """Хешировать файл порциями без загрузки в память."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Сохранить JSON атомарно в папку новой оценки."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def verify_plan(path=None):
    """Проверить прежние модели, пороги и исходники до доступа к test."""
    path = PLAN_PATH if path is None else Path(path)
    expected = path.with_name('plan.sha256').read_text().split()[0]
    if sha256(path) != expected:
        raise ValueError('Изменён зафиксированный план')
    plan = json.loads(path.read_text())
    if len(plan['models']) != 7 or len(plan['comparisons']) != 7:
        raise ValueError('Нужны семь зафиксированных моделей и сравнений')
    if plan['new_training_allowed'] or plan['full_C2b_included']:
        raise ValueError('План не допускает нового обучения')
    records = [model['model'] for model in plan['models']]
    records += list(plan['train_borders'].values()) + list(plan['source_files'].values())
    records += list(plan['provenance'].values())
    for record in records:
        if sha256(saved_path(record['path'])) != record['sha256']:
            raise ValueError('Изменён артефакт: ' + record['path'])
    def relocate(value):
        if isinstance(value, dict):
            return {key: str(saved_path(item)) if key == 'path' else relocate(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value
    return relocate(plan)


def canonical_queries(raw):
    """Сохранить физические log_id и прежнюю Arrow-сортировку целых групп."""
    if not raw['log_id'].is_unique or not raw['log_id'].is_monotonic_increasing:
        raise ValueError('Исходные log_id должны строго возрастать')
    raw = raw.copy()
    raw['source_position'] = np.arange(len(raw), dtype='int64')
    positions, labels, groups, stats = select_mixed_groups(raw)
    selected = raw.iloc[positions].reset_index(drop=True)
    selected['time'] = selected['start_ts']
    selected['end_ts'] = selected['timestamp']
    selected['day'] = np.maximum(selected['time'].to_numpy(dtype='int64') // 86400, 0)
    add_pair_keys(selected)
    targets = {'label': labels.astype('int8'), 'group_id': groups.astype('int64'),
               'log_id': selected['log_id'].to_numpy(dtype='int64'),
               'day': selected['day'].to_numpy(dtype='int64'),
               'source_position': positions.astype('int64')}
    _validate_queries(selected, targets)
    return selected, targets, stats


def test_mask(table):
    """Исходный test до конца данных с прежней формальной конечной границей."""
    upper = np.iinfo(np.int64).max
    return pc.and_(pc.greater_equal(table['start_ts'], 285 * 86400),
                   pc.less(table['timestamp'], int(upper) - 1800))


def prepare_targets(data_dir, result_dir, plan):
    """Независимо сверить listens и samples до формирования общего test-порядка."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    if (result_dir / 'targets.json').exists():
        raise FileExistsError('Targets уже подготовлены; повтор требует регистрации причины')
    started = time.perf_counter()
    source = ROOT / 'data/listens.parquet'
    expected = plan['upstream_data_provenance_from_full_run']['listens']['sha256']
    if sha256(source) != expected:
        raise ValueError('Изменён listens.parquet относительно полного обучения')
    parts, offset = [], 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=500_000, columns=SOURCE_COLUMNS):
        table = pa.Table.from_batches([batch])
        table = table.append_column('log_id', pa.array(
            np.arange(offset, offset + len(batch), dtype='int64')))
        offset += len(batch)
        selected = table.filter(test_mask(table))
        if len(selected):
            parts.append(selected)
    if not parts:
        raise ValueError('Test пуст')
    raw = pa.concat_tables(parts).to_pandas(split_blocks=True, self_destruct=True)
    del parts
    expected = pq.read_table(ROOT / 'data/samples.parquet',
                             columns=SOURCE_COLUMNS + ['group_id'],
                             filters=[('split', '==', 'test')]).to_pandas()
    if len(raw) != len(expected):
        raise ValueError('Samples и исходный журнал дали разный размер test')
    for column in SOURCE_COLUMNS:
        np.testing.assert_array_equal(raw[column], expected[column], err_msg=column)
    raw['group_id'] = expected['group_id'].to_numpy(copy=True)
    del expected
    day = np.maximum(raw['start_ts'].to_numpy(dtype='int64') // 86400, 0)
    np.testing.assert_array_equal(raw['group_id'], raw['uid'].to_numpy(dtype='int64') * 1000 + day)
    raw['source_position'] = np.arange(len(raw), dtype='int64')
    raw['time'], raw['end_ts'] = raw['start_ts'], raw['timestamp']
    queries, targets, stats = canonical_queries(raw)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(data_dir / 'raw_queries.parquet', index=False)
    queries.to_parquet(data_dir / 'queries.parquet', index=False)
    np.savez_compressed(data_dir / 'targets.npz', **targets)
    records = {name: sha256(data_dir / name)
               for name in ['raw_queries.parquet', 'queries.parquet', 'targets.npz']}
    report = {'completed': True, 'split': 'test', **stats,
              'source_rows': offset, 'day_min': int(targets['day'].min()),
              'day_max': int(targets['day'].max()), 'hashes': records,
              'samples_match_listens': True, 'all_model_rows_shared': True,
              'elapsed_seconds': time.perf_counter() - started}
    write_json(result_dir / 'targets.json', report)
    return report
