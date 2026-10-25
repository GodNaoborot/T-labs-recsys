"""Подготовить samples и 42 ручных признака исходного временного протокола."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import rolling_manual_features as manual

ROOT = Path(__file__).resolve().parents[1]
DAY, GAP = 86400, 1800
SAMPLE_COLUMNS = ['uid', 'item_id', 'artist_id', 'timestamp', 'start_ts',
                  'group_id', 'label', 'is_organic', 'track_length_seconds']
BASE_METADATA = ['uid', 'item_id', 'start_ts', 'group_id', 'split', 'label',
                 'is_organic', 'track_length_seconds']


def select_samples(listens):
    """Повторить границы train/val/test и GAP исходного dataset-analysis."""
    rows = listens.loc[listens['start_ts'] >= 225 * DAY, SAMPLE_COLUMNS].copy()
    is_train = rows['start_ts'].to_numpy() < 270 * DAY
    is_val = rows['start_ts'].to_numpy() < 285 * DAY
    split_end = np.select([is_train, is_val], [270 * DAY, 285 * DAY],
                          np.iinfo('int64').max)
    rows['split'] = pd.Categorical(
        np.select([is_train, is_val], ['train', 'val'], 'test'),
        categories=['train', 'val', 'test'])
    return rows.loc[rows['timestamp'].to_numpy('int64') < split_end - GAP].reset_index(drop=True)


def prepare_samples(data_dir):
    """Сохранить узкие целевые строки без исторических признаков."""
    output = data_dir / 'samples.parquet'
    if output.exists():
        raise FileExistsError(f'Файл уже существует: {output}')
    listens = pq.read_table(data_dir / 'listens.parquet', columns=SAMPLE_COLUMNS,
                            filters=[('start_ts', '>=', 225 * DAY)]).to_pandas()
    rows = select_samples(listens)
    if rows.empty or not rows['uid'].is_monotonic_increasing:
        raise ValueError('Нужны непустые целевые строки, упорядоченные по uid')
    temporary = output.with_suffix('.parquet.tmp')
    try:
        rows.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'rows': len(rows), 'split_rows': rows.groupby('split', observed=True).size().to_dict()}


def prepare_features(data_dir):
    """Посчитать прежние 42 входа по полной истории и записать порциями."""
    output = data_dir / 'features.parquet'
    if output.exists():
        raise FileExistsError(f'Файл уже существует: {output}')
    targets = pd.read_parquet(data_dir / 'samples.parquet')
    targets.insert(0, 'log_id', np.arange(len(targets), dtype='int64'))
    names = json.loads((ROOT / 'results/baselines/timing-20261020/run_config.json').read_text())['features']
    columns = list(dict.fromkeys(BASE_METADATA + names))
    temporary = output.with_suffix('.parquet.tmp')
    original_project = manual.PROJECT
    manual.PROJECT = data_dir.parent
    try:
        manual._validate_targets(targets)
        history = manual._ManualHistory()
        writer = None
        try:
            for chunk in manual._target_chunks(targets):
                values = history.chunk_features(chunk)[columns]
                table = pa.Table.from_pandas(values, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError('Пустая выборка')
        temporary.replace(output)
    finally:
        manual.PROJECT = original_project
        temporary.unlink(missing_ok=True)
    return {'rows': len(targets), 'feature_count': len(names), 'features': names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['samples', 'features', 'all'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args(argv)
    data = args.data_dir.resolve()
    # Расчёт истории ожидает стандартный каталог project/data.
    if data.name != 'data':
        parser.error('--data-dir должен оканчиваться на data')
    report = {'completed': False, 'stage': args.stage, 'test_evaluated': False}
    started = time.perf_counter()
    if args.stage in ('samples', 'all'):
        report['samples'] = prepare_samples(data)
    if args.stage in ('features', 'all'):
        report['manual'] = prepare_features(data)
    report.update(completed=True, elapsed_seconds=time.perf_counter() - started)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
