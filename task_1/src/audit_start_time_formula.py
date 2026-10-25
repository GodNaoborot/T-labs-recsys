"""Диагностика формулы start_ts только на сохранённых train/validation строках."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from replay_paths import project_path, run_name

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'dfs462_controls_20261023_v1'


def exact_start(end, length, ratio):
    """Целочисленный эталон floor(E-LR/100-5), без float-округления."""
    end = np.asarray(end, dtype='int64')
    length = np.asarray(length, dtype='int64')
    ratio = np.asarray(ratio, dtype='int64')
    if np.any(length < 0) or np.any(ratio < 0):
        raise ValueError('Длительность и процент должны быть неотрицательными')
    return end - 5 - (length * ratio + 99) // 100


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests():
    """Разрешить ровно два старых split; test-пути не используются."""
    selections = ROOT / 'data/experiment-1/temporal_controls' / CONTROL / 'selection'
    requests = {}
    for split in ['train', 'val']:
        path = selections / (split + '.npz')
        with np.load(path, allow_pickle=False) as archive:
            ids, labels = archive['log_id'], archive['label']
        order = np.argsort(ids, kind='stable')
        size = len(ids)
        requests[split] = {'path': path, 'log_id': ids, 'label': labels,
            'order': order, 'sorted_ids': ids[order], 'found': np.zeros(size, dtype='bool'),
            **{name: np.empty(size, dtype='int64') for name in
               ['timestamp', 'start_ts', 'track_length_seconds', 'played_ratio_pct']}}
    return requests


def read_selected(requests):
    """Сохранить только ранее выбранные train/val log_id из физического журнала."""
    columns = ['timestamp', 'start_ts', 'track_length_seconds', 'played_ratio_pct', 'label']
    offset = 0
    maximum = max(int(row['sorted_ids'][-1]) for row in requests.values())
    for batch in pq.ParquetFile(ROOT / 'data/listens.parquet').iter_batches(
            batch_size=500_000, columns=columns):
        right = offset + len(batch)
        for record in requests.values():
            left_index, right_index = np.searchsorted(record['sorted_ids'], [offset, right])
            if left_index == right_index:
                continue
            destination = record['order'][left_index:right_index]
            local = record['sorted_ids'][left_index:right_index] - offset
            if record['found'][destination].any():
                raise ValueError('Повторное сопоставление log_id')
            for name in columns[:-1]:
                record[name][destination] = batch.column(name).to_numpy()[local]
            np.testing.assert_array_equal(record['label'][destination],
                                           batch.column('label').to_numpy()[local])
            if np.any(record['start_ts'][destination] >= 285 * 86400):
                raise ValueError('В диагностические строки попал test')
            record['found'][destination] = True
        offset = right
        if offset > maximum:
            break
    if not all(record['found'].all() for record in requests.values()):
        raise ValueError('Не все train/validation log_id найдены')


def summarize(requests):
    records, tables = {}, []
    for split, row in requests.items():
        end, length, ratio, start = [row[name] for name in
            ['timestamp', 'track_length_seconds', 'played_ratio_pct', 'start_ts']]
        reconstructed = exact_start(end, length, ratio)
        phase_formula = (-((length * ratio + 99) // 100)) % 5
        phase = start % 5
        records[split] = {
            'rows': len(start), 'targets_sha256': hash_file(row['path']),
            'timestamp_multiple5_fraction': float(np.mean(end % 5 == 0)),
            'length_multiple5_fraction': float(np.mean(length % 5 == 0)),
            'stored_start_matches_exact_fraction': float(np.mean(start == reconstructed)),
            'stored_start_exact_mismatches': int(np.sum(start != reconstructed)),
            'phase_matches_current_length_ratio_fraction': float(np.mean(phase == phase_formula)),
            'nonzero_start_phase_fraction': float(np.mean(phase != 0)),
            'ratio100_count': int(np.sum(ratio == 100)),
            'ratio100_phase0_fraction': float(np.mean(phase[ratio == 100] == 0)),
            'start_ts_min': int(start.min()), 'start_ts_max': int(start.max()),
        }
        for p in range(5):
            selected = phase == p
            tables.append({'split': split, 'phase': p, 'rows': int(selected.sum()),
                'ratio_lt35_fraction': float(np.mean(ratio[selected] < 35)),
                'ratio100_fraction': float(np.mean(ratio[selected] == 100)),
                'label0_fraction': float(np.mean(row['label'][selected] == 0)),
                'label1_fraction': float(np.mean(row['label'][selected] == 1)),
                'label2_fraction': float(np.mean(row['label'][selected] == 2))})
    return records, pd.DataFrame(tables)


def main(output=None, controls_run=CONTROL):
    global CONTROL
    CONTROL = run_name(controls_run)
    output = project_path(output) if output else ROOT / 'results/analysis/start_ts_formula_20261024_v1'
    if output.exists():
        raise FileExistsError(output)
    requests = load_requests()
    read_selected(requests)
    records, table = summarize(requests)
    output.mkdir(parents=True)
    table.to_csv(output / 'phase_ratio.csv', index=False)
    result = {'completed': True, 'test_used': False, 'training_performed': False,
              'splits': records,
              'identity': 'start_mod5 = (-ceil(current_length*current_ratio/100)) mod5 when end_mod5=0',
              'interpretation': 'a current-outcome-derived phase; true start is not independently observed',
              'source_sha256': hash_file(Path(__file__))}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'audit_start_time_formula.py').write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2))
    print(table.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-run', default=CONTROL)
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.output, args.controls_run)
