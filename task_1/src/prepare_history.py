"""Подготовить полную историю Yambda без EDA, признаков и обучения.

Начало прослушивания сохраняется только для прежней разметки. Основной
временной протокол берёт cutoff из исходных timestamp и не использует его в X.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

SOURCE_CONFIG = Path(__file__).resolve().parents[1] / 'dataset_source.json'
SOURCE = json.loads(SOURCE_CONFIG.read_text(encoding='utf-8'))
DATASET = SOURCE['dataset']
DEFAULT_REVISION = SOURCE['revision']
RAW_FILE = 'flat/50m/multi_event.parquet'
ARTIST_FILE = 'artist_item_mapping.parquet'
RAW_COLUMNS = ['uid', 'item_id', 'timestamp', 'event_type', 'played_ratio_pct',
               'track_length_seconds', 'is_organic']
REACTIONS = ['like', 'unlike', 'dislike', 'undislike']
OUTPUT_FILES = ['listens.parquet', 'events.parquet', 'history_preparation.json']
DAY = 86400
SLACK = 5
SHIFT = 2 ** 26
PAIR_BASE = 10 ** 7


def file_hash(path):
    """Посчитать SHA256 порциями, не загружая весь parquet в память."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def check_output(output_dir):
    output_dir = Path(output_dir)
    occupied = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if occupied:
        raise FileExistsError('История уже существует: ' + ', '.join(occupied)
                              + '. Используйте новую папку.')
    if (output_dir / '.history-preparation.lock').exists():
        raise FileExistsError('В этой папке уже идёт подготовка истории')


def check_columns(path, required):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    schema = pq.read_schema(path)
    missing = set(required) - set(schema.names)
    if missing:
        raise ValueError(f'{path.name}: нет столбцов {sorted(missing)}')
    return path


def input_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_hash(path)}


def check_numeric(series, lower, upper):
    """Не допускать переполнения компактных типов и индекса пары."""
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f'{series.name}: нужны конечные значения без пропусков')
    if len(series) and (series.min() < lower or series.max() > upper):
        raise ValueError(f'{series.name}: допустим диапазон [{lower}, {upper}]')


def validate_raw(frame):
    if frame.empty:
        raise ValueError('Исходный журнал пуст')
    check_numeric(frame.uid, 0, np.iinfo('uint32').max)
    check_numeric(frame.item_id, 0, PAIR_BASE - 1)
    check_numeric(frame.timestamp, 0, SHIFT - 1)
    if not frame.uid.is_monotonic_increasing:
        raise ValueError('Нужен исходный порядок Yambda по uid; строки не пересортировываются')
    mask = frame.event_type.eq('listen').to_numpy()
    if not mask.any():
        raise ValueError('В журнале нет listen')
    for column, lower, upper in [('played_ratio_pct', 0, 255),
                                 ('track_length_seconds', 0, 65535), ('is_organic', 0, 1)]:
        check_numeric(frame.loc[mask, column], lower, upper)
    return mask


def listen_values(frame, mask, column, dtype=None):
    """Копировать один столбец listen, не всю большую таблицу."""
    values = frame[column].to_numpy()[mask]
    return values if dtype is None else values.astype(dtype, copy=False)


def attach_reaction(kind, frame, pair, code, ordered_positions, starts):
    """Привязать реакцию к первому следующему listen той же пары.

    Совпадение с концом listen разрешено. При равном времени выбирается первая
    исходная listen-строка; реакция раньше восстановленного начала не подходит.
    """
    rows = np.flatnonzero(frame.event_type.eq(kind).to_numpy())
    timestamps = frame.timestamp.to_numpy()
    positions = np.searchsorted(code, pair[rows] * SHIFT + timestamps[rows], side='left')
    safe = np.minimum(positions, len(code) - 1)
    same_pair = (positions < len(code)) & (code[safe] // SHIFT == pair[rows])
    listen_positions = ordered_positions[safe[same_pair]]
    event_times = timestamps[rows[same_pair]]
    accepted = listen_positions[event_times >= starts[listen_positions]]
    return np.unique(accepted), {'events': len(rows), 'attached_events': len(accepted),
                                 'listens': len(np.unique(accepted))}


def listen_labels(frame, mask):
    """Повторить сохранённую разметку: like, затем dislike/порог 35%."""
    starts_float = listen_values(frame, mask, 'timestamp', 'float64')
    played_seconds = listen_values(frame, mask, 'track_length_seconds', 'float64')
    played_seconds *= listen_values(frame, mask, 'played_ratio_pct', 'float64')
    played_seconds /= 100
    starts_float -= played_seconds
    starts_float -= SLACK
    np.floor(starts_float, out=starts_float)
    del played_seconds
    if starts_float.min() < np.iinfo('int32').min:
        raise ValueError('Восстановленное начало не помещается в int32')
    starts = starts_float.astype('int32')
    del starts_float
    pair = pd.factorize(frame.uid.to_numpy('int64') * PAIR_BASE
                        + frame.item_id.to_numpy('int64'))[0]
    code = pair[mask] * SHIFT + frame.loc[mask, 'timestamp'].to_numpy()
    order = np.argsort(code, kind='stable')
    code = code[order]
    labels = (frame.loc[mask, 'played_ratio_pct'].to_numpy() >= 35).astype('int8')
    liked, like_report = attach_reaction('like', frame, pair, code, order, starts)
    disliked, dislike_report = attach_reaction('dislike', frame, pair, code, order, starts)
    labels[disliked] = 0
    labels[liked] = 2
    report = {'like': like_report, 'dislike': dislike_report,
              'listens_with_both': len(np.intersect1d(liked, disliked))}
    return starts, labels, report


def build_history(frame, artists):
    """Сохранить физический порядок исходных listen и реакций."""
    mask = validate_raw(frame)
    starts, labels, attachment = listen_labels(frame, mask)
    mapping = artists.drop_duplicates('item_id', keep='first').set_index('item_id')['artist_id']
    group_ids = listen_values(frame, mask, 'uid', 'int64')
    group_ids *= 1000
    group_ids += np.maximum(starts // DAY, 0)
    listens = pd.DataFrame({
        'uid': listen_values(frame, mask, 'uid', 'uint32'),
        'item_id': listen_values(frame, mask, 'item_id', 'uint32'),
        'artist_id': frame.loc[mask, 'item_id'].map(mapping).fillna(-1).to_numpy('int32'),
        'timestamp': listen_values(frame, mask, 'timestamp', 'int32'), 'start_ts': starts,
        'played_ratio_pct': listen_values(frame, mask, 'played_ratio_pct', 'uint8'),
        'track_length_seconds': listen_values(frame, mask, 'track_length_seconds', 'uint16'),
        'is_organic': listen_values(frame, mask, 'is_organic', 'uint8'), 'label': labels,
        'group_id': group_ids,
    })
    events = frame.loc[frame.event_type.isin(REACTIONS),
                       ['uid', 'item_id', 'timestamp', 'event_type']].copy()
    events = events.astype({'timestamp': 'int32', 'event_type': 'category'})
    return listens, events, attachment


def download_inputs(revision):
    """Определить ревизию и скачать оба файла из одного snapshot."""
    from huggingface_hub import HfApi, hf_hub_download
    resolved = HfApi().dataset_info(DATASET, revision=revision).sha
    if not resolved:
        raise ValueError('Не удалось определить commit датасета')
    paths = [Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=resolved))
             for name in [RAW_FILE, ARTIST_FILE]]
    return *paths, {'dataset': DATASET, 'requested_revision': revision,
                    'resolved_revision': resolved}


def publish_staged(staging, output_dir):
    """Перенести готовые файлы, не заменяя существующую историю."""
    created = []
    try:
        for name in OUTPUT_FILES:
            target = output_dir / name
            os.link(staging / name, target)
            created.append(target)
    except Exception:
        for path in created:
            if path.stat().st_ino == (staging / path.name).stat().st_ino:
                path.unlink()
        raise


def prepare_history(raw_events, artist_mapping, output_dir, download=None):
    output_dir = Path(output_dir).resolve()
    check_output(output_dir)
    raw_events = check_columns(raw_events, RAW_COLUMNS)
    artist_mapping = check_columns(artist_mapping, ['item_id', 'artist_id'])
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / '.history-preparation.lock'
    started = time.monotonic()
    with lock.open('x'):
        try:
            check_output_files = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
            if check_output_files:
                raise FileExistsError('Выходы уже появились: ' + ', '.join(check_output_files))
            sources = {'raw_events': input_record(raw_events),
                       'artist_mapping': input_record(artist_mapping)}
            print('Чтение полного журнала и подготовка меток', flush=True)
            frame = pq.read_table(raw_events, columns=RAW_COLUMNS).to_pandas(
                split_blocks=True, self_destruct=True, categories=['event_type'])
            frame['timestamp'] = frame['timestamp'].astype('int32')
            artists = pq.read_table(artist_mapping).to_pandas(split_blocks=True, self_destruct=True)
            raw_rows = len(frame)
            listens, events, attachment = build_history(frame, artists)
            del frame, artists
            gc.collect()
            with tempfile.TemporaryDirectory(prefix='.history-', dir=output_dir) as temporary:
                staging = Path(temporary)
                listens.to_parquet(staging / 'listens.parquet', index=False)
                events.to_parquet(staging / 'events.parquet', index=False)
                report = {
                    'completed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'raw_rows': raw_rows, 'listen_rows': len(listens), 'event_rows': len(events),
                    'sources': sources, 'download': download,
                    'label_rule': 'like -> 2; else dislike or played_ratio_pct < 35 -> 0; else 1',
                    'attachment': attachment,
                    'saved_start_rule': 'floor(timestamp - track_length_seconds * played_ratio_pct / 100 - 5)',
                    'start_ts_role': 'legacy label attachment only; not C4 feature cutoff',
                    'physical_order_preserved': True,
                    'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                                 'pyarrow': pyarrow.__version__},
                    'code_sha256': file_hash(Path(__file__)),
                    'elapsed_seconds': time.monotonic() - started,
                    'outputs': {name: input_record(staging / name)
                                for name in OUTPUT_FILES[:2]},
                }
                for name, record in report['outputs'].items():
                    record['path'] = str(output_dir / name)
                (staging / 'history_preparation.json').write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                publish_staged(staging, output_dir)
            print(json.dumps({'completed': True, 'listen_rows': len(listens),
                              'event_rows': len(events), 'output_dir': str(output_dir)},
                             ensure_ascii=False), flush=True)
            return report
        finally:
            lock.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--raw-events', type=Path, help='Локальный multi_event.parquet')
    source.add_argument('--download', action='store_true', help='Явно разрешить скачивание Yambda')
    parser.add_argument('--artist-mapping', type=Path)
    parser.add_argument('--revision', default=DEFAULT_REVISION,
                        help='Commit/ref Hugging Face; по умолчанию зафиксирован dataset_source.json')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--check-only', action='store_true', help='Проверить входы/план без чтения строк и записи')
    args = parser.parse_args()
    if args.raw_events and not args.artist_mapping:
        parser.error('С --raw-events требуется --artist-mapping')
    if args.download and args.artist_mapping:
        parser.error('С --download оба файла берутся из одного snapshot')
    return args


def main():
    args = parse_args()
    check_output(args.output_dir)
    if args.check_only:
        if args.raw_events:
            check_columns(args.raw_events, RAW_COLUMNS)
            check_columns(args.artist_mapping, ['item_id', 'artist_id'])
        print(json.dumps({'checked': True, 'download_executed': False,
                          'history_generated': False, 'output_dir': str(args.output_dir)},
                         ensure_ascii=False))
        return
    download = None
    if args.download:
        args.raw_events, args.artist_mapping, download = download_inputs(args.revision)
    prepare_history(args.raw_events, args.artist_mapping, args.output_dir, download)


if __name__ == '__main__':
    main()
