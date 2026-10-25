"""Общий протокол пяти временных окон; исходные артефакты не изменяются."""
from pathlib import Path
import gc
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bootstrap_validation import file_sha256, read_json, write_json, runtime_versions

ROOT = Path(__file__).resolve().parents[1]
DAY, GAP, SEED = 86_400, 1800, 42
WINDOWS = [(121, 165), (136, 180), (151, 195), (206, 250), (221, 265)]
RUN_NAME = os.environ.get('TLAB_ROLLING_RUN')
RUN_HELP = (
    'Без TLAB_ROLLING_RUN используются прежние папки rolling_windows.\n'
    'Завершенный прогон только показывает статус и наличие отчетов.\n'
    'Для нового прогона задайте TLAB_ROLLING_RUN=my_run: данные, модели и результаты\n'
    'попадут в отдельные rolling_windows_runs/my_run. Допустимы 1–64 символа:\n'
    'латинские буквы, цифры, _ и -, первый символ — буква или цифра.\n'
    'Частичный прогон продолжается только при точном совпадении protocol.json.\n'
    'Публичный Markdown-отчет: python src/rolling_report.py с тем же окружением.'
)


def _run_folder(name):
    """Выбрать прежнюю папку либо безопасное отдельное пространство нового запуска."""
    if name is None:
        return Path('rolling_windows')
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) is None:
        raise ValueError('TLAB_ROLLING_RUN: нужно имя из 1–64 букв/цифр/_/-, начиная с буквы или цифры')
    return Path('rolling_windows_runs') / name


RUN_FOLDER = _run_folder(RUN_NAME)
DATA_ROOT = ROOT / 'data/experiment-1' / RUN_FOLDER
MODELS_ROOT = ROOT / 'models/experiment-1' / RUN_FOLDER
RESULTS = ROOT / 'results/experiment-1' / RUN_FOLDER
META = ['log_id', 'uid', 'item_id', 'artist_id', 'start_ts', 'timestamp',
        'label', 'group_id', 'is_organic', 'track_length_seconds', 'split']


def window_spec(start, end):
    """Перевести включительные дни с единицы в полуинтервалы секунд с нуля."""
    if end - start + 1 != 45:
        raise ValueError('Train должен содержать 45 дней')
    return {'id': f'train_{start}_{end}', 'train_days': [start, end],
            'val_days': [end + 1, end + 15],
            'train_start': (start - 1) * DAY, 'train_end': end * DAY,
            'val_start': end * DAY, 'val_end': (end + 15) * DAY}


def get_spec(window_id):
    """Найти ровно одно из заранее зафиксированных окон."""
    values = [window_spec(*bounds) for bounds in WINDOWS]
    return next(spec for spec in values if spec['id'] == window_id)


def window_paths(window_id):
    """Разнести новые данные, модели и результаты по отдельным папкам окна."""
    allowed = {window_spec(*bounds)['id'] for bounds in WINDOWS} | {'integration_smoke'}
    if window_id not in allowed:
        raise ValueError(f'Неизвестный идентификатор окна: {window_id}')
    return {'data': DATA_ROOT / window_id,
            'models': MODELS_ROOT / window_id,
            'results': RESULTS / window_id}


def make_directories(paths):
    """Создать только папки нового эксперимента."""
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def release_memory():
    """Вернуть временные Arrow/Python буферы ОС между тяжелыми этапами."""
    gc.collect()
    pa.default_memory_pool().release_unused()
    import ctypes
    ctypes.CDLL(None).malloc_trim(0)


def filter_target_batch(table, spec):
    """Оставить train/val запросы, завершившиеся до границы своего split минус GAP."""
    start, end = table['start_ts'], table['timestamp']
    interval = pc.and_(pc.greater_equal(start, spec['train_start']),
                       pc.less(start, spec['val_end']))
    is_train = pc.less(start, spec['train_end'])
    boundary = pc.if_else(is_train, spec['train_end'] - GAP, spec['val_end'] - GAP)
    table = table.filter(pc.and_(interval, pc.less(end, boundary)))
    split = pc.if_else(pc.less(table['start_ts'], spec['train_end']), 'train', 'val')
    return table.append_column('split', split)


def prepare_targets(spec):
    """Выбрать целевые события из полного журнала, не обрезая будущую историю признаков."""
    paths = window_paths(spec['id'])
    make_directories(paths)
    columns = [name for name in META if name not in {'log_id', 'split'}]
    pieces, offset = [], 0
    source = pq.ParquetFile(ROOT / 'data/listens.parquet')
    for batch in source.iter_batches(batch_size=1_000_000, columns=columns):
        table = pa.Table.from_batches([batch])
        ids = pa.array(np.arange(offset, offset + len(table), dtype='int64'))
        offset += len(table)
        selected = filter_target_batch(table.append_column('log_id', ids), spec)
        if len(selected):
            pieces.append(selected)
    table = pa.concat_tables(pieces).select(META)
    pq.write_table(table, paths['data'] / 'targets.parquet', row_group_size=100_000)
    return target_report(table, spec, paths)


def target_report(table, spec, paths):
    """Проверить идентификаторы/времена и сохранить границы и размеры окна."""
    targets = table.to_pandas()
    assert targets['log_id'].is_unique
    assert np.all(np.diff(targets['log_id']) > 0)
    np.testing.assert_array_equal(targets['group_id'],
                                  targets['uid'].astype('int64') * 1000 + targets['start_ts'] // DAY)
    counts = targets.groupby('split', observed=True).size().to_dict()
    report = {'completed': True, 'window': spec, 'rows': counts,
              'targets_sha256': file_sha256(paths['data'] / 'targets.parquet'),
              'gap_seconds': GAP, 'day_numbering': 'one-based inclusive display',
              'time_rule': 'history.timestamp < query.start_ts', 'test_used': False}
    write_json(paths['results'] / 'targets_report.json', report)
    return report


def fixed_model_params(objective):
    """Повторить параметры исходного сравнения без тюнинга по новым окнам."""
    params = dict(random_seed=SEED, learning_rate=0.1, iterations=600, depth=6,
                  early_stopping_rounds=50, use_best_model=True, verbose=25,
                  allow_writing_files=False, border_count=254)
    if objective == 'ranker':
        params.update(loss_function='YetiRank', thread_count=4,
                      eval_metric='NDCG:top=10;type=Base;denominator=LogPosition')
    elif objective == 'pointwise':
        params.update(loss_function='MultiClass', eval_metric='MultiClass', thread_count=8)
    else:
        raise ValueError(objective)
    return params


def protocol_sources():
    """Перечислить данные, определения признаков и код, от которых зависит запуск."""
    files = [ROOT / f'data/{name}.parquet' for name in ('listens', 'events')]
    files += sorted((ROOT / 'src').glob('rolling_*.py'))
    files += [ROOT / 'src/block_bootstrap.py', ROOT / 'src/cumulative_features.py']
    files += [ROOT / 'src/run_rolling_windows.py']
    files += [ROOT / f'results/{suffix}/run_config.json'
              for suffix in ('baselines/timing-20261020', 'experiment-1/dfs')]
    files += sorted((ROOT / 'data/experiment-1/featuretools').glob(
        'train_sample_200_cumulative_*_definitions.json'))
    return files


def experiment_protocol():
    """Зафиксировать окна, параметры и правило общего успеха до первого обучения."""
    return {'windows': [window_spec(*b) for b in WINDOWS], 'seed': SEED, 'run_name': RUN_NAME,
            'params': {name: fixed_model_params(name) for name in ('ranker', 'pointwise')},
            'bootstrap': {'block_days': [3, 4], 'n_resamples': 20_000, 'margin': .03,
                          'alpha_sig': .05, 'seed': SEED, 'upper_bound': 'percentile'},
            'success_rule': 'for each objective: U95 < 3% in all windows and both block lengths',
            'test_used': False, 'versions': runtime_versions(),
            'source_files': {str(p): file_sha256(p) for p in protocol_sources()},
            'limitations': ['Overlapping train; dependent windows.',
                            'Per-window bootstrap conditional on fixed trained models.',
                            'One seed; margin chosen after first experiment.',
                            'Last val overlaps original val; exploratory robustness analysis.']}


def freeze_protocol():
    """Не подменять протокол или код при продолжении прерванного запуска."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / 'protocol.json'
    current = experiment_protocol()
    if path.exists():
        saved = read_json(path)
        saved.pop('created_unix')
        assert saved == current, 'Протокол изменился: нужен отдельный эксперимент'
    else:
        current['created_unix'] = time.time()
        write_json(path, current)
    return path
