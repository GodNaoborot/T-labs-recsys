"""Повторная test-оценка C4: те же формулы и строгая история timestamp<t.

Здесь нет обучения, выбора признаков или старых test-матриц. Расчётные ядра
C4 переиспользуются; расширяется только допустимый диапазон запросов/истории.
"""
from pathlib import Path
from replay_paths import saved_path
import gc
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import c4_features as c4
from project_artifacts import TASK_ROOT, file_hash
from temporal_sample_data import _array_hash, _validate_queries
from temporal_full_data import BRANCHES, load_selected_definitions, _release_heap
from user_session_features import SESSION_FEATURES


def _read_test_history(path, columns, limit, batch_size=500_000):
    """Читать доступную историю, назначая physical ID перед фильтрацией."""
    if not isinstance(limit, (int, np.integer)) or limit < c4.TEST_START:
        raise ValueError('Для test нужен целочисленный предел не раньше 285-го дня')
    if batch_size <= 0:
        raise ValueError('batch_size должен быть положительным')
    parts, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        table = pa.Table.from_batches([batch])
        mask = table['timestamp'].to_numpy() < limit
        if mask.any():
            frame = table.filter(pa.array(mask)).to_pandas(split_blocks=True, self_destruct=True)
            frame['source_event_id'] = offset + np.flatnonzero(mask)
            parts.append(frame)
        offset += len(batch)
    if not parts:
        return pd.DataFrame({name: pd.Series(dtype='int64')
                             for name in columns + ['source_event_id']})
    return pd.concat(parts, ignore_index=True)


class C4TestManualHistory(c4.C4ManualHistory):
    """Те же ручные/сессионные формулы; история продолжается на test без fit."""
    def __init__(self, limit, listens_path=None, events_path=None):
        listens_path = listens_path or TASK_ROOT / 'data/listens.parquet'
        events_path = events_path or TASK_ROOT / 'data/events.parquet'
        listens = c4._history_order(_read_test_history(listens_path,
            ['uid', 'item_id', 'artist_id', 'timestamp', 'played_ratio_pct'], limit))
        self.listens = {name: listens[name].to_numpy() for name in
                       ['uid', 'item_id', 'artist_id', 'timestamp', 'source_event_id']}
        self.listens['skip'] = (listens['played_ratio_pct'] < 35).to_numpy('int8')
        self.listens['pr'] = listens['played_ratio_pct'].clip(upper=100).to_numpy('float32')
        self.session_history = listens[['uid', 'timestamp', 'played_ratio_pct', 'source_event_id']]
        del listens
        reactions = _read_test_history(events_path, ['uid', 'item_id', 'timestamp', 'event_type'], limit)
        reactions = c4._history_order(reactions.loc[reactions.event_type.isin(['like', 'unlike'])])
        self.likes = {name: reactions[name].to_numpy() for name in
                      ['uid', 'item_id', 'timestamp', 'source_event_id']}
        self.likes['is_like'] = (reactions.event_type == 'like').to_numpy('int8')
        self.reference_time = int(np.max(self.listens['timestamp'], initial=0))
        self._build_item_indices()


def _load_test_queries(data_dir):
    """Проверить общий test-порядок, alias cutoff и новые user-day группы."""
    queries = pd.read_parquet(data_dir / 'test_queries.parquet')
    with np.load(data_dir / 'test_targets.npz') as source:
        targets = {name: source[name] for name in ['label', 'group_id', 'log_id', 'day']}
    if queries.empty or not queries.log_id.is_unique:
        raise ValueError('Нужны непустые уникальные test-запросы')
    times = queries.time.to_numpy()
    if times.dtype.kind not in 'iu' or np.any(times < c4.TEST_START) or np.any(times % 5):
        raise ValueError('Cutoff test должен быть целочисленным на исходной 5s сетке')
    for alias in ['t', 'start_ts']:
        if alias in queries:
            np.testing.assert_array_equal(times, queries[alias].to_numpy(), err_msg=alias)
    if 'end_ts' not in queries:
        queries['end_ts'] = queries['timestamp']
    if not (queries.end_ts.to_numpy() > times).all():
        raise ValueError('Target-пакет должен быть строго позже cutoff')
    if np.any(queries.end_ts.to_numpy() % 5):
        raise ValueError('Исходный timestamp target должен лежать на 5s сетке')
    groups = queries.group_id.to_numpy()
    ids = queries.log_id.to_numpy()
    if np.any(groups[1:] < groups[:-1]) or np.any((groups[1:] == groups[:-1]) & (ids[1:] <= ids[:-1])):
        raise ValueError('Порядок test должен быть (group_id, log_id)')
    _validate_queries(queries, targets)
    return queries, targets


def _feature_spec(plan):
    """Проверить frozen имена/порядок и hashes файлов формул до их загрузки."""
    schemas = plan.get('schemas', {})
    manual_names = schemas.get('manual49', plan.get('manual_features'))
    dfs_names = schemas.get('dfs462', plan.get('dfs_features'))
    if not isinstance(manual_names, list) or not isinstance(dfs_names, list):
        raise ValueError('План должен содержать schemas.manual49/dfs462')
    if (len(manual_names) != 49 or len(dfs_names) != 462 or
            len(set(manual_names)) != 49 or len(set(dfs_names)) != 462):
        raise ValueError('Нарушена frozen размерность признаков 49/462')
    expected_manual = json.loads(c4.OLD_CONTROLS.read_text())['manual_features'] + list(SESSION_FEATURES)
    if manual_names != expected_manual:
        raise ValueError('Изменён порядок frozen 49 ручных признаков')
    selection = plan['selection']
    paths = {parent: saved_path(plan['definitions'][parent]['path']) for parent, _ in BRANCHES}
    records = {'selection': selection, **{parent: plan['definitions'][parent] for parent, _ in BRANCHES}}
    for name, record in records.items():
        if file_hash(saved_path(record['path'])) != record['sha256']:
            raise ValueError(f'Изменён frozen источник формул: {name}')
    parents = {path.parent.resolve() for path in paths.values()}
    if len(parents) != 1 or any(path.name != parent + '.json' for parent, path in paths.items()):
        raise ValueError('Definitions должны быть отдельными parent.json в общем каталоге')
    loaded, definitions = load_selected_definitions(saved_path(selection['path']), next(iter(parents)))
    if loaded != dfs_names:
        raise ValueError('Порядок 462 признаков отличается от frozen плана')
    return manual_names, dfs_names, definitions, {name: record['sha256'] for name, record in records.items()}


def _check_availability(result_dir, limit):
    availability = json.loads((result_dir / 'history_label_availability.json').read_text())
    flags = ['completed', 'frozen_labels_match', 'all_attached_reactions_at_or_before_listen_end', 'test_used']
    if not all(availability.get(name) is True for name in flags):
        raise ValueError('Не подтверждена доступность frozen исторических меток на test')
    upper = availability.get('history_upper_exclusive')
    if upper is not None and upper < limit:
        raise ValueError('Проверка доступности истории не покрывает test cutoff')
    return availability


def _mark_test_report(result_dir, kind, report, targets):
    """Обозначить test в отчёте общего C4 writer; формулы writer не меняются."""
    report.update(test_used=True, recomputed_for_c4_test=True, fitted=False,
                  target_arrays={name: _array_hash(values) for name, values in targets.items()})
    c4._write_json(result_dir / f'generation_{kind}_test.json', report)
    return report


def prepare_test_features(data_dir, result_dir, plan, progress=None):
    """Сохранить общие F-order float32 test-матрицы 49/462 из сырой истории."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    manual_names, dfs_names, definitions, formula_hashes = _feature_spec(plan)
    queries, targets = _load_test_queries(data_dir)
    limit = int(queries.time.max())
    availability = _check_availability(result_dir, limit)
    # DFS reader использует эти же постоянные пути. Иной источник не подменяем.
    for name in ['listens', 'events']:
        if saved_path(plan['sources'][name]['path']).resolve() != (TASK_ROOT / 'data' / f'{name}.parquet').resolve():
            raise ValueError('Источник истории отличается от frozen рабочего файла')
    contract = {'query_arrays': c4._query_record(queries),
                'target_arrays': {name: _array_hash(value) for name, value in targets.items()},
                'schemas': {'manual49': manual_names, 'dfs462': dfs_names},
                'formula_sha256': formula_hashes, 'sources': plan['sources'],
                'history_label_availability_sha256': file_hash(result_dir / 'history_label_availability.json')}
    contract_path = result_dir / 'test_features_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Изменён frozen контракт признаков при возобновлении')
    if not contract_path.exists():
        c4._write_json(contract_path, contract)
    history = C4TestManualHistory(limit)
    manual_report = c4._manual_split(queries, history, manual_names,
                                   data_dir / 'manual49', result_dir, 'test', progress)
    manual_report = _mark_test_report(result_dir, 'manual', manual_report, targets)
    del history
    gc.collect()
    _release_heap()
    dfs_report = c4._dfs_split(queries, targets, dfs_names, definitions,
                              data_dir / 'dfs462', result_dir, 'test', progress)
    dfs_report = _mark_test_report(result_dir, 'dfs', dfs_report, targets)
    for folder, features in [('manual49', manual_names), ('dfs462', dfs_names)]:
        c4._write_json(data_dir / folder / 'features.json', {'features': features,
                       'feature_count': len(features), 'dtype': 'float32', 'layout': 'F-order'})
    report = {'completed': True, 'test_used': True, 'fitted': False, 'new_feature_selection': False,
              'history_rule': 'original_timestamp < t; [t-window,t)',
              'history_upper_exclusive': limit, 'history_label_availability': availability,
              'manual': manual_report, 'dfs': dfs_report, **contract}
    c4._write_json(result_dir / 'features_preparation.json', report)
    return report
