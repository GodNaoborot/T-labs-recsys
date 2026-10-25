"""Признаки C4: прежние формулы, новая граница t и ограниченная память.

Старые матрицы и start_ts не читаются. Все ветви используют исходный timestamp<t.
Набор 462 формул фиксирован; пороги CatBoost рассчитываются следующим этапом.
"""
from pathlib import Path
import gc
import json
import time

import featuretools as ft
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import rolling_manual_features as manual
from project_artifacts import TASK_ROOT, file_hash
from temporal_sample_data import _array_hash, _validate_queries, load_branch_history
from temporal_full_data import (BRANCHES, DEFAULT_SELECTION, DEFAULT_DEFINITIONS,
    load_selected_definitions, _branch_matrix, _export_branch, release_mapping, _release_heap)
from user_session_features import SESSION_FEATURES, calculate_session_features

DAY = 86400
TEST_START = 285 * DAY
OLD_CONTROLS = TASK_ROOT / 'results/experiment-1/temporal_controls/dfs462_controls_20261023_v1/protocol.json'


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.incomplete')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _read_before(path, columns, limit, batch_size=500_000):
    """Оставить только доступную историю, сохраняя физический ID до фильтрации."""
    if not 0 < limit < TEST_START:
        raise ValueError('История C4 должна заканчиваться до TEST_START')
    parts, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        table = pa.Table.from_batches([batch])
        selected = table['timestamp'].to_numpy() < limit
        if selected.any():
            frame = table.filter(pa.array(selected)).to_pandas(split_blocks=True, self_destruct=True)
            frame['source_event_id'] = offset + np.flatnonzero(selected)
            parts.append(frame)
        offset += len(batch)
    if not parts:
        return pd.DataFrame({name: pd.Series(dtype='int64') for name in columns + ['source_event_id']})
    return pd.concat(parts, ignore_index=True)


def _history_order(frame):
    """Технический порядок не зависит от исходной перестановки и значений label/R/L."""
    users, identifiers = frame.uid.to_numpy(), frame.source_event_id.to_numpy()
    physical_order = len(frame) < 2 or np.all(identifiers[1:] > identifiers[:-1])
    if not physical_order and not frame['source_event_id'].is_unique:
        raise ValueError('source_event_id истории должен быть уникальным')
    # Индексы движка сами стабильно сортируют время; физический порядок ties
    # задаётся ID. Уже упорядоченный исходный файл не копируем целиком.
    if len(frame) < 2 or (np.all(users[1:] >= users[:-1]) and
            np.all(identifiers[1:][users[1:] == users[:-1]] > identifiers[:-1][users[1:] == users[:-1]])):
        return frame
    return frame.sort_values(['uid', 'source_event_id'], kind='stable').reset_index(drop=True)


class C4ManualHistory(manual._ManualHistory):
    """Переиспользовать формулы baseline, загрузив только историю раньше C4 cutoff."""
    def __init__(self, limit, listens_path=None, events_path=None):
        listens_path = listens_path or TASK_ROOT / 'data/listens.parquet'
        events_path = events_path or TASK_ROOT / 'data/events.parquet'
        listens = _history_order(_read_before(listens_path,
            ['uid', 'item_id', 'artist_id', 'timestamp', 'played_ratio_pct'], limit))
        self.listens = {name: listens[name].to_numpy() for name in
                        ['uid', 'item_id', 'artist_id', 'timestamp', 'source_event_id']}
        self.listens['skip'] = (listens['played_ratio_pct'] < 35).to_numpy('int8')
        self.listens['pr'] = listens['played_ratio_pct'].clip(upper=100).to_numpy('float32')
        self.session_history = listens[['uid', 'timestamp', 'played_ratio_pct', 'source_event_id']]
        del listens
        likes = _read_before(events_path, ['uid', 'item_id', 'timestamp', 'event_type'], limit)
        likes = _history_order(likes.loc[likes['event_type'].isin(['like', 'unlike'])])
        self.likes = {name: likes[name].to_numpy() for name in
                      ['uid', 'item_id', 'timestamp', 'source_event_id']}
        self.likes['is_like'] = (likes['event_type'] == 'like').to_numpy('int8')
        self.reference_time = int(np.max(self.listens['timestamp'], initial=0))
        self._build_item_indices()

    def features(self, queries, names):
        """Вернуть 42+7 входов; alias start_ts здесь означает только разрешённый t."""
        adapted = queries.copy()
        adapted['start_ts'] = adapted['time'].to_numpy('int64')
        if 'timestamp' not in adapted:
            adapted['timestamp'] = adapted['end_ts']
        if 'split' not in adapted:
            adapted['split'] = 'C4'
        block = self.chunk_features(adapted)
        session = calculate_session_features(adapted[['uid', 'start_ts']], self.session_history)
        for name in SESSION_FEATURES:
            block[name] = session[name].to_numpy()
        return block[names].to_numpy(dtype='float32')


def _probe_positions(rows, count=40):
    return np.unique(np.linspace(0, rows - 1, min(count, rows), dtype='int64'))


def _oracle_manual(history, queries, names):
    """Независимые pandas-фильтры без prefix-сумм и поисковых индексов движка."""
    h = pd.DataFrame(history.listens, copy=False)
    selected = h['uid'].isin(queries.uid) | h['item_id'].isin(queries.item_id)
    h = h.loc[selected].copy()
    likes = pd.DataFrame(history.likes, copy=False)
    likes = likes.loc[likes.uid.isin(queries.uid) | likes.item_id.isin(queries.item_id)]
    session = history.session_history
    session = session.loc[session.uid.isin(queries.uid)]
    results, raw_recencies = [], []
    for q in queries.itertuples(index=False):
        t, values = int(q.time), {'is_organic': q.is_organic, 'track_length_seconds': q.track_length_seconds}
        past = h.loc[h.timestamp < t]
        ui = past.loc[(past.uid == q.uid) & (past.item_id == q.item_id)]
        ua = past.loc[(past.uid == q.uid) & (past.artist_id == q.artist_id)]
        item = past.loc[past.item_id == q.item_id]
        for prefix, frame, windows in [('ui', ui, manual.UI_WINDOWS), ('ua', ua, manual.UA_WINDOWS)]:
            for days in windows:
                window = frame.loc[frame.timestamp >= t - days * DAY]
                values[f'{prefix}_lst_amt_{days}'] = len(window)
                skip = window['skip'].sum()
                suffix = 'cnt' if prefix == 'ui' else 'rate'
                values[f'{prefix}_skip_{suffix}_{days}'] = (skip if prefix == 'ui' else skip / len(window) if len(window) else np.nan)
            for half in manual.HALF_LIVES:
                values[f'{prefix}_lst_decay_{half}'] = np.exp2((frame.timestamp.to_numpy('float64') - t) / (half * DAY)).sum()
        values['ui_time_since_last_lst'] = t - int(ui.timestamp.max()) if len(ui) else np.nan
        reactions = likes.loc[(likes.uid == q.uid) & (likes.item_id == q.item_id) & (likes.timestamp < t)]
        values['ui_liked_now'] = int(reactions.sort_values(['timestamp', 'source_event_id']).iloc[-1].is_like) if len(reactions) else 0
        for days in manual.ITEM_WINDOWS:
            window = item.loc[item.timestamp >= t - days * DAY]
            liked = likes.loc[(likes.item_id == q.item_id) & (likes.timestamp < t) &
                             (likes.timestamp >= t - days * DAY) & (likes.is_like == 1)]
            values[f'item_lst_amt_{days}'] = len(window)
            values[f'item_skip_rate_{days}'] = window['skip'].mean()
            values[f'item_like_rate_{days}'] = len(liked) / len(window) if len(window) else np.nan
            values[f'item_avg_played_ratio_{days}'] = window['pr'].astype('float64').mean()
        if q.artist_id < 0:
            for name in list(values):
                if name.startswith('ua_'):
                    values[name] = np.nan
        user = session.loc[(session.uid == q.uid) & (session.timestamp < t)].sort_values(['timestamp', 'source_event_id'])
        recent = user.loc[user.timestamp >= t - DAY]
        previous = user.iloc[-1] if len(user) else None
        session_values = [min(float(previous.played_ratio_pct), 100) if previous is not None else np.nan,
            float(previous.played_ratio_pct < 35) if previous is not None else np.nan,
            (user.tail(5).played_ratio_pct < 35).mean() if len(user) else np.nan,
            (user.tail(20).played_ratio_pct < 35).mean() if len(user) else np.nan,
            (recent.played_ratio_pct < 35).mean() if len(recent) else np.nan, len(recent),
            t - int(previous.timestamp) if previous is not None else np.nan]
        values.update(zip(SESSION_FEATURES, session_values))
        for value in (values['ui_time_since_last_lst'], values['user_time_since_last']):
            if np.isfinite(value):
                if int(value) % 5:
                    raise AssertionError('Исходная целочисленная давность не кратна пяти')
                raw_recencies.append(int(value))
        results.append([values[name] for name in names])
    return np.asarray(results, dtype='float32'), len(raw_recencies)


def _event_expression(feature, frame):
    if isinstance(feature, ft.IdentityFeature):
        return frame[feature.column_name].astype('float64')
    if isinstance(feature, ft.TransformFeature) and feature.primitive.name == 'multiply_numeric':
        left, right = feature.base_features
        return _event_expression(left, frame) * _event_expression(right, frame)
    raise NotImplementedError(feature.get_name())


def _oracle_formula(feature, past, t):
    """Прямой pandas oracle формулы, включая окна и преобразование свежести в дни."""
    if isinstance(feature, ft.TransformFeature) and feature.primitive.name == 'multiply_numeric':
        left, right = feature.base_features
        return _oracle_formula(left, past, t) * _oracle_formula(right, past, t)
    if not isinstance(feature, ft.AggregationFeature):
        raise NotImplementedError(feature.get_name())
    window = feature.use_previous
    if window is not None:
        seconds = window.delta_obj.total_seconds()
        past = past.loc[past.end_ts >= t - seconds]
    primitive = feature.primitive.name
    if primitive == 'count':
        return len(past)
    if primitive in ('time_since_first', 'time_since_last'):
        if not len(past):
            return np.nan
        event = past.end_ts.min() if primitive == 'time_since_first' else past.end_ts.max()
        seconds = int(t) - int(event)
        if seconds % 5:
            raise AssertionError('DFS: исходная давность не кратна пяти')
        return seconds / DAY
    values = _event_expression(feature.base_features[0], past)
    return {'mean': values.mean, 'std': values.std, 'min': values.min, 'max': values.max}[primitive]()


def verify_dfs_probe(history, queries, key, chosen, stored):
    """Сверить все выбранные формулы на 40 строках, а не только свежесть."""
    positions = _probe_positions(len(queries))
    probe = queries.iloc[positions]
    old = history.loc[history[key].isin(probe[key].dropna())]
    comparisons = 0
    for position, q in zip(positions, probe.itertuples(index=False)):
        value = getattr(q, key)
        if pd.isna(value):
            expected = np.full(len(chosen), np.nan, dtype='float32')
        else:
            past = old.loc[(old[key] == value) & (old.end_ts < q.time)]
            if len(past) and int(past.end_ts.max()) >= int(q.time):
                raise AssertionError('История содержит недоступное событие')
            expected = np.asarray([_oracle_formula(feature, past, int(q.time))
                                   for _, feature in chosen], dtype='float32')
        actual = np.asarray(stored[position, [column for column, _ in chosen]])
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5, equal_nan=True,
                                   err_msg=f'DFS oracle {key}, log_id={q.log_id}')
        for j, (_, feature) in enumerate(chosen):
            if feature.primitive.name in ('time_since_first', 'time_since_last'):
                np.testing.assert_array_equal(actual[j], expected[j])
        comparisons += len(chosen)
    return {'rows': len(probe), 'values': comparisons, 'all_selected_formulas_checked': True,
            'pandas_oracle': True, 'freshness_exact_float32_days': True}


def _load_queries(data_dir, split):
    if split not in ('train', 'val'):
        raise ValueError('C4 не подготавливает test')
    queries = pd.read_parquet(data_dir / f'{split}_queries.parquet')
    with np.load(data_dir / f'{split}_targets.npz') as source:
        targets = {name: source[name] for name in ('label', 'group_id', 'log_id', 'day')}
    if len(queries) == 0 or np.any(queries.time.to_numpy() % 5) or not (queries.time < TEST_START).all():
        raise ValueError('Нужны непустые C4 train/val запросы на исходной 5s сетке')
    if 'end_ts' not in queries:
        queries['end_ts'] = queries['timestamp']
    if not (queries.end_ts > queries.time).all():
        raise ValueError('Target-пакет должен быть строго позже cutoff')
    _validate_queries(queries, targets)
    return queries, targets


def _query_record(queries):
    return {name: _array_hash(queries[name].to_numpy(dtype='int64'))
            for name in ('log_id', 'uid', 'item_id', 'artist_id', 'time', 'end_ts',
                         'label', 'group_id', 'track_length_seconds', 'is_organic')}


def _manual_split(queries, history, names, folder, result_dir, split, progress):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{split}_X.npy'
    report_path = result_dir / f'generation_manual_{split}.json'
    expected_queries = _query_record(queries)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get('completed') and report.get('query_arrays') == expected_queries
                and report.get('features') == names and file_hash(path) == report['matrix_sha256']):
            return report
        raise ValueError('Неполный либо изменённый C4 manual checkpoint')
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    matrix = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(queries), len(names)), fortran_order=True)
    probe = _probe_positions(len(queries))
    for left in range(0, len(queries), 30_000):
        right = min(left + 30_000, len(queries))
        matrix[left:right] = history.features(queries.iloc[left:right], names)
        release_mapping(matrix)
        if progress:
            progress({'stage': 'manual_features', 'split': split, 'rows': right, 'total_rows': len(queries)})
    expected, recencies = _oracle_manual(history, queries.iloc[probe], names)
    np.testing.assert_allclose(matrix[probe], expected, rtol=3e-5, atol=1e-5, equal_nan=True)
    report = {'completed': True, 'matrix_path': str(path), 'features': names, 'rows': len(queries),
        'feature_count': len(names), 'matrix_sha256': file_hash(path), 'query_arrays': expected_queries,
        'oracle': {'rows': len(probe), 'values': len(probe) * len(names), 'pandas_oracle': True,
                   'raw_integer_recencies_checked': recencies},
        'elapsed_seconds': time.perf_counter() - started, 'test_used': False}
    del matrix
    _write_json(report_path, report)
    return report


def _dfs_split(queries, targets, names, definitions, folder, result_dir, split, progress):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{split}_X.npy'
    report_path = result_dir / f'generation_dfs_{split}.json'
    expected_queries = _query_record(queries)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get('completed') and report.get('query_arrays') == expected_queries
                and report.get('features') == names and file_hash(path) == report['matrix_sha256']):
            return report
        raise ValueError('Неполный либо изменённый C4 DFS checkpoint')
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    final = np.lib.format.open_memmap(path, mode='w+', dtype='float32',
        shape=(len(queries), len(names)), fortran_order=True)
    branches = {}
    for column, name in enumerate(names):
        if '__' not in name:
            final[:, column] = queries[name].to_numpy(dtype='float32')
            release_mapping(final)
    for parent, key in BRANCHES:
        chosen = definitions[parent]
        if not chosen:
            continue
        branch_started = time.perf_counter()
        if progress:
            progress({'stage': 'dfs_history', 'split': split, 'branch': parent})
        requests = queries[['log_id', 'time', key]].copy()
        history = load_branch_history(requests, key)
        if len(history) and not (history.end_ts < int(queries.time.max())).all():
            raise AssertionError('В ветвь попала недоступная история')
        scratch = folder / f'.{split}_{parent}_scratch.npy'
        matrix, inverse = _branch_matrix(requests, history, parent, key, chosen, scratch,
            30_000, 500_000, progress)
        _export_branch(matrix, inverse, final, [column for column, _ in chosen], 30_000)
        oracle = verify_dfs_probe(history, requests, key, chosen, final)
        branches[parent] = {'seconds': time.perf_counter() - branch_started,
                            'history_rows': len(history), 'features': len(chosen), 'oracle': oracle}
        del matrix, inverse, history, requests
        scratch.unlink()
        _release_heap()
    release_mapping(final)
    del final
    report = {'completed': True, 'matrix_path': str(path), 'features': names, 'rows': len(queries),
        'feature_count': len(names), 'matrix_sha256': file_hash(path), 'query_arrays': expected_queries, 'branches': branches,
        'target_arrays': {name: _array_hash(value) for name, value in targets.items()},
        'elapsed_seconds': time.perf_counter() - started, 'test_used': False}
    _write_json(report_path, report)
    return report


def prepare_features(data_dir, result_dir, progress=None):
    """Подготовить общие матрицы 49 и 462 C4-входов, train/val без test."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    availability_path = result_dir / 'history_label_availability.json'
    availability = json.loads(availability_path.read_text())
    if not all(availability.get(name) is True for name in ('completed',
            'frozen_labels_match', 'all_attached_reactions_at_or_before_listen_end')):
        raise ValueError('Не подтверждена доступность исторических меток')
    if availability.get('test_used') is not False:
        raise ValueError('Историческая проверка не должна использовать test')
    old_protocol = json.loads(OLD_CONTROLS.read_text())
    manual_names = list(old_protocol['manual_features']) + list(SESSION_FEATURES)
    names, definitions = load_selected_definitions(DEFAULT_SELECTION, DEFAULT_DEFINITIONS)
    if len(manual_names) != 49 or len(names) != 462 or names != old_protocol['dfs_features']:
        raise AssertionError('Изменён frozen список признаков')
    split_inputs = {split: _load_queries(data_dir, split) for split in ('train', 'val')}
    history = C4ManualHistory(max(int(queries.time.max()) for queries, _ in split_inputs.values()))
    report = {'completed': False, 'history_rule': 'original_timestamp < t; [t-window,t)',
              'test_used': False, 'new_feature_selection': False,
              'history_label_availability': availability, 'manual': {}, 'dfs': {}}
    for split, (queries, _) in split_inputs.items():
        report['manual'][split] = _manual_split(queries, history, manual_names,
            data_dir / 'manual49', result_dir, split, progress)
    del history
    gc.collect()
    _release_heap()
    for split, (queries, targets) in split_inputs.items():
        report['dfs'][split] = _dfs_split(queries, targets, names, definitions,
            data_dir / 'dfs462', result_dir, split, progress)
    for directory, features in [('manual49', manual_names), ('dfs462', names)]:
        _write_json(data_dir / directory / 'features.json', {'features': features,
                    'feature_count': len(features), 'dtype': 'float32', 'layout': 'F-order'})
    report['completed'] = True
    report['sources_sha256'] = {'selection': file_hash(DEFAULT_SELECTION),
        'old_controls_protocol': file_hash(OLD_CONTROLS),
        **{parent: file_hash(DEFAULT_DEFINITIONS / f'{parent}.json') for parent, _ in BRANCHES}}
    _write_json(result_dir / 'features_preparation.json', report)
    return report
