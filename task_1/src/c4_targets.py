"""Общие C4-запросы: следующий timestamp-пакет без восстановления cutoff из R."""
from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from generate_cumulative import add_pair_keys
from temporal_sample_data import _array_hash, _validate_queries

ROOT = Path(__file__).resolve().parents[1]
DAY, GAP = 86400, 1800
TRAIN_START, VAL_START, TEST_START = 225 * DAY, 270 * DAY, 285 * DAY
LISTEN_COLUMNS = ['uid', 'item_id', 'artist_id', 'timestamp', 'start_ts',
                  'played_ratio_pct', 'track_length_seconds', 'is_organic', 'label']
QUERY_COLUMNS = ['log_id', 'uid', 'item_id', 'artist_id', 'is_organic',
                 'track_length_seconds', 'time', 't', 'start_ts', 'timestamp',
                 'end_ts', 'label', 'group_id', 'day', 'user_item_id', 'user_artist_id']
NO_REACTION = np.iinfo(np.int64).max


def _write_json(path, value):
    """Записать небольшой отчёт атомарно."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.incomplete')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


def packet_cutoffs(frame):
    """Вернуть предыдущий строго меньший timestamp пользователя; первый пакет = -1."""
    if not frame['log_id'].is_unique:
        raise ValueError('log_id должен быть уникальным')
    if frame.empty:
        return np.empty(0, dtype='int64')
    uid = frame.uid.to_numpy(dtype='int64')
    end = frame.timestamp.to_numpy(dtype='int64')
    ids = frame.log_id.to_numpy(dtype='int64')
    order = np.lexsort((ids, end, uid))
    users, times = uid[order], end[order]
    starts = np.flatnonzero(np.r_[True, (users[1:] != users[:-1]) | (times[1:] != times[:-1])])
    first_user_packet = np.r_[True, users[starts[1:]] != users[starts[:-1]]]
    previous = np.r_[-1, times[starts[:-1]]]
    previous[first_user_packet] = -1
    sorted_cutoffs = np.repeat(previous, np.diff(np.r_[starts, len(frame)]))
    cutoffs = np.empty(len(frame), dtype='int64')
    cutoffs[order] = sorted_cutoffs
    return cutoffs


def _pair_codes(frame):
    return ((frame.uid.to_numpy(dtype='uint64') << np.uint64(32)) |
            frame.item_id.to_numpy(dtype='uint64'))


def attach_reactions(listens, reactions):
    """Повторить старое next-listen правило; сохранить каждую привязанную реакцию.

    При одинаковом E выбирается первый физический log_id той же пары.
    Нижняя граница — прежний сохранённый start_ts, только для разметки.
    """
    empty = pd.DataFrame({'reaction_source_id': pd.Series(dtype='int64'),
                          'log_id': pd.Series(dtype='int64'),
                          'timestamp': pd.Series(dtype='int64'),
                          'event_type': pd.Series(dtype='object')})
    if listens.empty or reactions.empty:
        return empty
    required = {'reaction_source_id', 'uid', 'item_id', 'timestamp', 'event_type'}
    if not required.issubset(reactions.columns):
        raise ValueError('Реакциям нужны исходные physical source IDs')
    chosen = reactions[reactions.event_type.isin(['like', 'dislike'])].reset_index(drop=True)
    if chosen.empty:
        return empty
    pairs = _pair_codes(listens)
    end = listens.timestamp.to_numpy(dtype='int64')
    ids = listens.log_id.to_numpy(dtype='int64')
    order = np.lexsort((ids, end, pairs))
    key_type = np.dtype([('pair', '<u8'), ('time', '<i8')])
    keys = np.empty(len(listens), dtype=key_type)
    keys['pair'], keys['time'] = pairs[order], end[order]
    lookup = np.empty(len(chosen), dtype=key_type)
    lookup['pair'], lookup['time'] = _pair_codes(chosen), chosen.timestamp.to_numpy(dtype='int64')
    positions = np.searchsorted(keys, lookup, side='left')
    safe = np.minimum(positions, len(keys) - 1)
    valid = ((positions < len(keys)) & (keys['pair'][safe] == lookup['pair']))
    destination = order[safe]
    start = listens.start_ts.to_numpy(dtype='int64')
    valid &= lookup['time'] >= start[destination]
    if np.any(lookup['time'][valid] > end[destination[valid]]):
        raise AssertionError('Реакция стала доступна после конца своего listen')
    return pd.DataFrame({
        'reaction_source_id': chosen.reaction_source_id.to_numpy(dtype='int64')[valid],
        'log_id': ids[destination[valid]],
        'timestamp': lookup['time'][valid],
        'event_type': chosen.event_type.astype(str).to_numpy()[valid]})


def reaction_state(listens, attachments):
    """Проверить frozen labels и найти самую раннюю like/dislike каждого listen."""
    if not listens['log_id'].is_unique:
        raise ValueError('log_id должен быть уникальным')
    earliest = np.full(len(listens), NO_REACTION, dtype='int64')
    liked, disliked = np.zeros(len(listens), dtype='bool'), np.zeros(len(listens), dtype='bool')
    if len(attachments):
        positions = pd.Index(listens.log_id).get_indexer(attachments.log_id)
        if (positions < 0).any():
            raise ValueError('Привязанная реакция не найдена в listen-блоке')
        times = attachments.timestamp.to_numpy(dtype='int64')
        if np.any(times > listens.timestamp.to_numpy(dtype='int64')[positions]):
            raise AssertionError('Исторический label ещё не доступен в E')
        np.minimum.at(earliest, positions, times)
        liked[positions[attachments.event_type.to_numpy() == 'like']] = True
        disliked[positions[attachments.event_type.to_numpy() == 'dislike']] = True
    labels = (listens.played_ratio_pct.to_numpy() >= 35).astype('int8')
    labels[disliked], labels[liked] = 0, 2
    np.testing.assert_array_equal(labels, listens.label.to_numpy(dtype='int8'),
                                  err_msg='Изменена исходная разметка listen')
    return earliest, liked, disliked


def target_masks(frame, cutoff, earliest_reaction):
    """Явные причины исключения; guard учитывает любую привязанную реакцию <t."""
    cutoff = np.asarray(cutoff, dtype='int64')
    end = frame.timestamp.to_numpy(dtype='int64')
    train = (cutoff >= TRAIN_START) & (cutoff < VAL_START)
    val = (cutoff >= VAL_START) & (cutoff < TEST_START)
    boundary = np.where(train, VAL_START - GAP, TEST_START - GAP)
    first = cutoff < 0
    outside = ~(train | val)
    gap = (train | val) & (end >= boundary)
    reaction = np.asarray(earliest_reaction, dtype='int64') < cutoff
    return {'first_packet': first, 'outside_train_val': outside,
            'target_end_gap': gap, 'attached_reaction_before_cutoff': reaction,
            'train': train & ~gap & ~reaction,
            'val': val & ~gap & ~reaction}


def _counts_by_label(labels, mask):
    values = np.bincount(np.asarray(labels, dtype='int8')[mask], minlength=3)
    return {str(i): int(values[i]) for i in range(3)}


def _add_counts(target, labels, masks, cutoff):
    """Сохранить пересечения причин и отдельный подсчёт по новой принадлежности split."""
    reasons = ['first_packet', 'outside_train_val', 'target_end_gap',
               'attached_reaction_before_cutoff']
    bits = np.zeros(len(labels), dtype='uint8')
    remaining = np.ones(len(labels), dtype='bool')
    for i, name in enumerate(reasons):
        mask = masks[name]
        bits |= mask.astype('uint8') << i
        for kind, selected in [('nonexclusive', mask), ('priority', mask & remaining)]:
            record = target.setdefault(kind, {}).setdefault(name, {'rows': 0, 'labels': {str(j): 0 for j in range(3)}})
            record['rows'] += int(selected.sum())
            for label, count in _counts_by_label(labels, selected).items():
                record['labels'][label] += count
        remaining &= ~mask
    for bitset, count in zip(*np.unique(bits, return_counts=True)):
        key = str(int(bitset))
        target.setdefault('intersection_bitsets', {})[key] = target.get('intersection_bitsets', {}).get(key, 0) + int(count)
    for split, lo, hi in [('train', TRAIN_START, VAL_START), ('val', VAL_START, TEST_START)]:
        assigned = (cutoff >= lo) & (cutoff < hi)
        record = target.setdefault('by_split', {}).setdefault(split, {})
        for reason in ['assigned', *reasons, 'eligible']:
            selected = assigned if reason == 'assigned' else assigned & (masks[split] if reason == 'eligible' else masks[reason])
            item = record.setdefault(reason, {'rows': 0, 'labels': {str(j): 0 for j in range(3)}})
            item['rows'] += int(selected.sum())
            for label, count in _counts_by_label(labels, selected).items():
                item['labels'][label] += count


def _read_reactions(path):
    """Идентификатор назначается до фильтрации времени и типа реакции."""
    frame = pq.read_table(path).to_pandas()
    frame['reaction_source_id'] = np.arange(len(frame), dtype='int64')
    if not frame.uid.is_monotonic_increasing:
        raise ValueError('История реакций должна идти по uid')
    return frame[frame.timestamp < TEST_START].reset_index(drop=True)


def _listen_blocks(path, counters, batch_size=500_000):
    """Читать до-285d историю порциями, не разрывая пользователя между блоками."""
    carry, offset, last_uid = None, 0, -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=LISTEN_COLUMNS):
        table = pa.Table.from_batches([batch])
        ids = np.arange(offset, offset + len(table), dtype='int64')
        offset += len(table)
        users = table['uid'].to_numpy()
        if len(users) and (np.any(users[1:] < users[:-1]) or users[0] < last_uid):
            raise ValueError('История listen должна идти по uid')
        if len(users):
            last_uid = int(users[-1])
        # Будущие строки не участвуют ни в пакетах, ни в проверке меток.
        before_test = table['timestamp'].to_numpy() < TEST_START
        table = table.filter(pa.array(before_test))
        frame = table.to_pandas()
        frame['log_id'] = ids[before_test]
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            carry = None
            continue
        end_user = int(frame.uid.iloc[-1])
        last_start = int(np.searchsorted(frame.uid.to_numpy(), end_user, side='left'))
        carry = frame.iloc[last_start:].copy()
        if last_start:
            yield frame.iloc[:last_start].copy()
        counters['source_rows_scanned'] = offset
    if carry is not None and len(carry):
        yield carry
    counters['source_rows_scanned'] = offset


def _queries(frame, cutoff, mask):
    """В запросе start_ts — только совместимый alias нового t, прежний S не сохраняем."""
    query = frame.loc[mask, ['log_id', 'uid', 'item_id', 'artist_id', 'is_organic',
                              'track_length_seconds', 'timestamp', 'label']].copy()
    query['time'] = cutoff[mask]
    query['t'], query['start_ts'] = query['time'], query['time']
    query['end_ts'] = query['timestamp']
    query['day'] = query['time'].to_numpy(dtype='int64') // DAY
    query['group_id'] = query.uid.to_numpy(dtype='int64') * 1000 + query.day.to_numpy(dtype='int64')
    add_pair_keys(query)
    return query[QUERY_COLUMNS]


def select_queries(frame, train_groups=None, seed=42):
    """Пересчитать смешанные группы и выбрать заданное число целых train-групп."""
    if frame.empty or not frame.log_id.is_unique:
        raise ValueError('Нужны непустые уникальные C4 запросы')
    grouped = frame.groupby('group_id', sort=True, observed=True).label.agg(['min', 'max'])
    mixed = grouped.index[grouped['min'] != grouped['max']].to_numpy(dtype='int64')
    constant_rows = int((~frame.group_id.isin(mixed)).sum())
    if train_groups is not None:
        if int(train_groups) != train_groups or train_groups <= 0:
            raise ValueError('Число train-групп должно быть положительным целым')
        if len(mixed) < train_groups:
            raise ValueError(f'Для C4 нужно {train_groups} mixed train-групп, доступно {len(mixed)}')
        selected = np.sort(np.random.default_rng(seed).choice(mixed, size=int(train_groups), replace=False))
    else:
        selected = mixed
    query = frame[frame.group_id.isin(selected)].sort_values(['group_id', 'log_id'], kind='stable').reset_index(drop=True)
    arrays = {name: query[name].to_numpy(dtype='int8' if name == 'label' else 'int64')
              for name in ['label', 'group_id', 'log_id', 'day']}
    _validate_queries(query, arrays)
    if np.any(query.end_ts.to_numpy() <= query.time.to_numpy()):
        raise AssertionError('Target-пакет должен находиться строго после cutoff')
    hashes = {name: _array_hash(values) for name, values in arrays.items()}
    hashes['time'] = _array_hash(query.time.to_numpy(dtype='int64'))
    digest = hashlib.sha256()
    for name in ['label', 'group_id', 'log_id', 'day', 'time']:
        digest.update(name.encode())
        digest.update(hashes[name].encode())
    stats = {'rows_before_mixed': len(frame), 'constant_group_rows_removed': constant_rows,
             'mixed_groups_available': len(mixed), 'rows': len(query), 'groups': len(selected),
             'labels': _counts_by_label(query.label, np.ones(len(query), dtype='bool')),
             'day_min': int(query.day.min()), 'day_max': int(query.day.max()),
             'array_sha256': hashes, 'shared_order_sha256': digest.hexdigest()}
    return query, arrays, stats


def _write_table(writer, table, path):
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression='zstd')
    writer.write_table(table, row_group_size=100_000)
    return writer


def prepare_targets(data_dir, result_dir, progress=None, train_groups=40000, seed=42):
    """Сохранить общие train/val запросы и отчёт; progress получает один dict.

    Источники читаются порциями целых uid. Test-запросы и test-метрики не готовятся.
    Реакции и labels сохраняют прежнее правило разметки, зависящее от старого S.
    """
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if (result_dir / 'targets.json').exists():
        raise FileExistsError('C4 targets уже подготовлены; нужен отдельный запуск')
    destinations = [data_dir / f'{split}_{kind}.{suffix}' for split in ['train', 'val']
                    for kind, suffix in [('queries', 'parquet'), ('targets', 'npz')]]
    destinations += [data_dir / 'reaction_attachments.parquet']
    if any(path.exists() for path in destinations):
        raise FileExistsError('Обнаружены неполные C4 targets; нужно отдельное имя')
    started = time.perf_counter()
    paths = {split: data_dir / f'{split}_eligible.incomplete.parquet' for split in ['train', 'val']}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Остался неполный C4 этап; нужно отдельное имя')
    writers, reaction_writer = {}, None
    counters = {'source_rows_scanned': 0, 'historical_listen_rows': 0, 'verified_labels': 0,
                'attached_reaction_rows': 0, 'attached_like_rows': 0, 'attached_dislike_rows': 0,
                'listens_with_both_reaction_types': 0}
    exclusions = {'priority_order': ['first_packet', 'outside_train_val', 'target_end_gap',
                                      'attached_reaction_before_cutoff'],
                  'intersection_bits': {'first_packet': 1, 'outside_train_val': 2,
                                        'target_end_gap': 4, 'attached_reaction_before_cutoff': 8}}
    reactions = _read_reactions(ROOT / 'data/events.parquet')
    reaction_users = reactions.uid.to_numpy()
    try:
        for block in _listen_blocks(ROOT / 'data/listens.parquet', counters):
            lo, hi = np.searchsorted(reaction_users, [block.uid.iloc[0], block.uid.iloc[-1]], side='left')
            hi = int(np.searchsorted(reaction_users, block.uid.iloc[-1], side='right'))
            attachments = attach_reactions(block, reactions.iloc[lo:hi])
            earliest, liked, disliked = reaction_state(block, attachments)
            cutoff = packet_cutoffs(block)
            masks = target_masks(block, cutoff, earliest)
            _add_counts(exclusions, block.label.to_numpy(dtype='int8'), masks, cutoff)
            counters['historical_listen_rows'] += len(block)
            counters['verified_labels'] += len(block)
            counters['attached_reaction_rows'] += len(attachments)
            counters['attached_like_rows'] += int((attachments.event_type == 'like').sum())
            counters['attached_dislike_rows'] += int((attachments.event_type == 'dislike').sum())
            counters['listens_with_both_reaction_types'] += int((liked & disliked).sum())
            if len(attachments):
                reaction_writer = _write_table(reaction_writer, pa.Table.from_pandas(attachments, preserve_index=False),
                                               data_dir / 'reaction_attachments.parquet')
            for split in ['train', 'val']:
                query = _queries(block, cutoff, masks[split])
                if len(query):
                    writers[split] = _write_table(writers.get(split), pa.Table.from_pandas(query, preserve_index=False), paths[split])
            if progress is not None:
                progress({'stage': 'targets', **counters})
            del block, attachments, cutoff, masks, earliest, liked, disliked
            gc.collect()
    finally:
        for writer in writers.values():
            writer.close()
        if reaction_writer is not None:
            reaction_writer.close()
    del reactions
    gc.collect()
    reports = {}
    for split in ['train', 'val']:
        if split not in writers:
            raise ValueError(f'C4 {split} пуст')
        eligible = pq.read_table(paths[split]).to_pandas()
        query, arrays, reports[split] = select_queries(eligible, train_groups if split == 'train' else None, seed)
        query_path, targets_path = data_dir / f'{split}_queries.parquet', data_dir / f'{split}_targets.npz'
        query.to_parquet(query_path, index=False, row_group_size=100_000)
        np.savez_compressed(targets_path, **arrays)
        reports[split].update({'queries_path': str(query_path), 'targets_path': str(targets_path),
                              'queries_sha256': _sha256(query_path), 'targets_sha256': _sha256(targets_path)})
        paths[split].unlink()
        del eligible, query, arrays
        gc.collect()
    availability = {'completed': True, 'test_used': False, 'history_upper_exclusive': TEST_START,
                    'frozen_labels_match': True, 'all_attached_reactions_at_or_before_listen_end': True,
                    'rule': 'next_same_pair_end>=reaction_ts; reaction_ts>=saved start_ts; like overrides dislike',
                    **counters}
    _write_json(result_dir / 'history_label_availability.json', availability)
    _write_json(result_dir / 'exclusions.json', exclusions)
    report = {'completed': True, 'test_used': False, 'train_groups_requested': train_groups,
              'seed': seed, 'history_rule': 'original_timestamp < t (previous distinct user timestamp)',
              'group_rule': 'uid*1000+floor(t/86400); mixed groups recomputed; order group_id,log_id',
              'label_rule_preserved': True, 'label_mapping_still_depends_on_reconstructed_start': True,
              'guard': 'exclude any attached like/dislike with timestamp<t; equality permitted',
              'historical_ties': '(timestamp, immutable physical source_event_id)',
              'split': {'train_start': TRAIN_START, 'val_start': VAL_START,
                        'validation_end': TEST_START, 'gap_seconds': GAP},
              'train': reports['train'], 'val': reports['val'], 'exclusions': exclusions,
              'history_label_availability': availability,
              'sources': {name: {'path': str(ROOT / 'data' / f'{name}.parquet'),
                                  'sha256': _sha256(ROOT / 'data' / f'{name}.parquet')}
                          for name in ['listens', 'events']},
              'elapsed_seconds': time.perf_counter() - started}
    _write_json(result_dir / 'targets.json', report)
    return report
