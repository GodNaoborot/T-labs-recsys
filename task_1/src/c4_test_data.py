"""Test-запросы C4: те же пакеты и guard, без переобучения и новых правил метки."""
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from c4_targets import (DAY, TEST_START, LISTEN_COLUMNS, _sha256, _write_json,
    _queries, _write_table, _counts_by_label, packet_cutoffs, attach_reactions,
    reaction_state, select_queries)


def test_target_masks(frame, cutoff, earliest_reaction):
    """Период задаёт t, guard исключает любую attached like/dislike строго раньше t."""
    cutoff = np.asarray(cutoff, dtype='int64')
    end = frame.timestamp.to_numpy(dtype='int64')
    if len(frame) != len(cutoff) or len(frame) != len(earliest_reaction):
        raise ValueError('Различаются размеры строк, cutoff и реакций')
    if np.any((cutoff >= 0) & (end <= cutoff)):
        raise AssertionError('Target-пакет должен находиться строго после cutoff')
    first = cutoff < 0
    before = cutoff < TEST_START
    guard = np.asarray(earliest_reaction, dtype='int64') < cutoff
    return {'first_packet': first, 'cutoff_before_test': before,
            'attached_reaction_before_cutoff': guard,
            'test': ~before & ~guard}


def _add_counts(target, labels, masks):
    """Считать причины отдельно и с приоритетом; пересечения не теряются."""
    reasons = target['priority_order']
    bits = np.zeros(len(labels), dtype='uint8')
    remaining = np.ones(len(labels), dtype='bool')
    assigned = ~masks['cutoff_before_test']
    by_test = target.setdefault('by_split', {}).setdefault('test', {})
    for i, name in enumerate(reasons):
        mask = masks[name]
        bits |= mask.astype('uint8') << i
        for kind, selected in [('nonexclusive', mask), ('priority', mask & remaining)]:
            _accumulate(target.setdefault(kind, {}), name, labels, selected)
        remaining &= ~mask
        _accumulate(by_test, name, labels, assigned & mask)
    _accumulate(by_test, 'assigned', labels, assigned)
    _accumulate(by_test, 'eligible', labels, masks['test'])
    for value, count in zip(*np.unique(bits, return_counts=True)):
        key = str(int(value))
        counts = target.setdefault('intersection_bitsets', {})
        counts[key] = counts.get(key, 0) + int(count)


def _accumulate(records, name, labels, mask):
    record = records.setdefault(name, {'rows': 0, 'labels': {str(i): 0 for i in range(3)}})
    record['rows'] += int(mask.sum())
    for label, count in _counts_by_label(labels, mask).items():
        record['labels'][label] += count


def _listen_blocks(path, counters, batch_size=500_000):
    """Полный журнал порциями целых uid; log_id назначается до любых фильтров."""
    carry, offset, last_uid = None, 0, -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=LISTEN_COLUMNS):
        frame = pa.Table.from_batches([batch]).to_pandas()
        users = frame.uid.to_numpy()
        if len(users) and (np.any(users[1:] < users[:-1]) or users[0] < last_uid):
            raise ValueError('История listen должна идти по uid')
        if len(users):
            last_uid = int(users[-1])
        frame['log_id'] = np.arange(offset, offset + len(frame), dtype='int64')
        offset += len(frame)
        counters['source_rows_scanned'] = offset
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            carry = None
            continue
        last_start = int(np.searchsorted(frame.uid.to_numpy(), frame.uid.iloc[-1], side='left'))
        carry = frame.iloc[last_start:].copy()
        if last_start:
            yield frame.iloc[:last_start].copy()
    if carry is not None and len(carry):
        yield carry
    counters['source_rows_scanned'] = offset


def _read_reactions(path):
    """Сохранить полный events и физический ID также для like/unlike истории."""
    frame = pq.read_table(path).to_pandas()
    frame['reaction_source_id'] = np.arange(len(frame), dtype='int64')
    if not frame.uid.is_monotonic_increasing:
        raise ValueError('История реакций должна идти по uid')
    return frame


def _verify_plan_sources(plan):
    """Проверить разрешение predict и неизменность raw-источников до их чтения."""
    if not plan.get('test_evaluation_authorized') or plan.get('new_training_allowed') is not False:
        raise ValueError('Нужен разрешённый план оценки без обучения')
    paths = {}
    for name in ('listens', 'events'):
        source = plan['sources'][name]
        path = Path(source['path'])
        if _sha256(path) != source['sha256']:
            raise ValueError('Изменён исходный файл: ' + str(path))
        paths[name] = path
    return paths


def prepare_test_targets(data_dir, result_dir, plan, progress=None):
    """Сохранить общий test-порядок из raw хронологии после проверки frozen-плана.

    Новые cutoff, mixed user-day и guard повторяют C4. Старые split/S не выбирают
    строки. Верхнего GAP нет: оцениваются все наблюдавшиеся следующие пакеты.
    """
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    destinations = [data_dir / 'test_queries.parquet', data_dir / 'test_targets.npz', data_dir / 'targets.npz',
                    data_dir / 'reaction_attachments.parquet',
                    data_dir / 'test_eligible.incomplete.parquet', result_dir / 'targets.json']
    if any(path.exists() for path in destinations):
        raise FileExistsError('C4 test targets уже начаты; нужен отдельный запуск')
    sources = _verify_plan_sources(plan)
    started = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / 'test_eligible.incomplete.parquet'
    query_writer, reaction_writer = None, None
    reasons = ['first_packet', 'cutoff_before_test', 'attached_reaction_before_cutoff']
    exclusions = {'priority_order': reasons,
                  'intersection_bits': {name: 1 << i for i, name in enumerate(reasons)}}
    counters = {'source_rows_scanned': 0, 'historical_listen_rows': 0, 'verified_labels': 0,
                'attached_reaction_rows': 0, 'attached_like_rows': 0, 'attached_dislike_rows': 0,
                'listens_with_both_reaction_types': 0}
    original_end_max, cutoff_max = -1, -1
    reactions = _read_reactions(sources['events'])
    reaction_users = reactions.uid.to_numpy()
    try:
        for block in _listen_blocks(sources['listens'], counters):
            lo = int(np.searchsorted(reaction_users, block.uid.iloc[0], side='left'))
            hi = int(np.searchsorted(reaction_users, block.uid.iloc[-1], side='right'))
            attachments = attach_reactions(block, reactions.iloc[lo:hi])
            earliest, liked, disliked = reaction_state(block, attachments)
            cutoff = packet_cutoffs(block)
            masks = test_target_masks(block, cutoff, earliest)
            _add_counts(exclusions, block.label.to_numpy(dtype='int8'), masks)
            counters['historical_listen_rows'] += len(block)
            counters['verified_labels'] += len(block)
            counters['attached_reaction_rows'] += len(attachments)
            counters['attached_like_rows'] += int((attachments.event_type == 'like').sum())
            counters['attached_dislike_rows'] += int((attachments.event_type == 'dislike').sum())
            counters['listens_with_both_reaction_types'] += int((liked & disliked).sum())
            original_end_max = max(original_end_max, int(block.timestamp.max()))
            cutoff_max = max(cutoff_max, int(cutoff.max(initial=-1)))
            if len(attachments):
                reaction_writer = _write_table(reaction_writer,
                    pa.Table.from_pandas(attachments, preserve_index=False),
                    data_dir / 'reaction_attachments.parquet')
            query = _queries(block, cutoff, masks['test'])
            if len(query):
                query_writer = _write_table(query_writer,
                    pa.Table.from_pandas(query, preserve_index=False), raw_path)
            if progress is not None:
                progress({'stage': 'test_targets', **counters})
            del block, attachments, earliest, liked, disliked, cutoff, masks, query
            gc.collect()
    finally:
        if query_writer is not None:
            query_writer.close()
        if reaction_writer is not None:
            reaction_writer.close()
    del reactions
    gc.collect()
    if query_writer is None:
        raise ValueError('C4 test пуст')
    eligible = pq.read_table(raw_path).to_pandas()
    query, arrays, stats = select_queries(eligible, train_groups=None, seed=42)
    if np.any(query.time.to_numpy(dtype='int64') < TEST_START):
        raise AssertionError('В test попал cutoff раньше 285-го дня')
    query_path, target_path = data_dir / 'test_queries.parquet', data_dir / 'test_targets.npz'
    query.to_parquet(query_path, index=False, row_group_size=100_000)
    np.savez_compressed(target_path, **arrays)
    alias = data_dir / 'targets.npz'
    alias.hardlink_to(target_path)
    if _sha256(alias) != _sha256(target_path):
        raise AssertionError('Повреждён совместимый targets.npz')
    stats.update({'queries_path': str(query_path), 'targets_path': str(target_path),
                  'queries_sha256': _sha256(query_path), 'targets_sha256': _sha256(target_path),
                  'targets_alias_path': str(alias)})
    raw_path.unlink()
    availability = {'completed': True, 'test_used': True,
                    'history_scope': 'complete source journal; actual feature history remains original_timestamp<t',
                    'frozen_labels_match': True,
                    'all_attached_reactions_at_or_before_listen_end': True,
                    'rule': 'next_same_pair_end>=reaction_ts; reaction_ts>=saved start_ts; like overrides dislike',
                    'source_end_max': original_end_max, **counters}
    _write_json(result_dir / 'history_label_availability.json', availability)
    _write_json(result_dir / 'exclusions.json', exclusions)
    report = {'completed': True, 'test_used': True, 'new_training': False, 'test': stats,
              'history_rule': 'original_timestamp < t (previous distinct user timestamp)',
              'group_rule': 'uid*1000+floor(t/86400); all mixed test groups; order group_id,log_id',
              'label_rule_preserved': True,
              'label_mapping_still_depends_on_reconstructed_start': True,
              'guard': 'exclude any attached like/dislike with timestamp<t; equality permitted',
              'historical_ties': '(timestamp, immutable physical source_event_id)',
              'split': {'test_cutoff_start': TEST_START, 'cutoff_max_before_mixed': cutoff_max,
                        'upper_bound': 'last observed next packet', 'end_gap_seconds': 0},
              'exclusions': exclusions, 'history_label_availability': availability,
              'sources': plan['sources'], 'elapsed_seconds': time.perf_counter() - started}
    _write_json(result_dir / 'targets.json', report)
    return report
