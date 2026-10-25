"""Diagnose a target-derived second phase on frozen train/validation rows only.

This audit reads no test data, fits no model, and changes no previous artifact.
Frequency tables are diagnostic probes, not new ranking model comparisons.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from project_artifacts import TASK_ROOT
from replay_paths import configure_legacy, project_path, saved_path, run_name
from temporal_sample_data import _source_paths

CONTROL_NAME = 'dfs462_controls_20261023_v1'
RESULT_NAME = 'start_ts_phase_20261024_v1'
META_COLUMNS = ['log_id', 'time', 'end_ts', 'track_length_seconds', 'label',
                'group_id', 'uid']
EXPECTED = {'train': (2039017, 40000), 'val': (2882983, 56946)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path, log_ids, batch_size=500000):
    """Read seven metadata columns and restore exact frozen selection order."""
    order = np.argsort(log_ids, kind='stable')
    ids = log_ids[order]
    if len(ids) > 1 and np.any(ids[1:] == ids[:-1]):
        raise ValueError('Repeated frozen log_id')
    found = np.zeros(len(ids), dtype=bool)
    output = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=META_COLUMNS):
        table = pa.Table.from_batches([batch])
        batch_ids = table['log_id'].to_numpy()
        positions = np.searchsorted(ids, batch_ids)
        matched = positions < len(ids)
        matched[matched] &= ids[positions[matched]] == batch_ids[matched]
        destination = order[positions[matched]]
        if np.any(found[destination]):
            raise ValueError('Repeated metadata log_id')
        found[destination] = True
        for name in META_COLUMNS:
            values = table[name].to_numpy(zero_copy_only=False)
            if name not in output:
                output[name] = np.empty(len(ids), dtype=values.dtype)
            output[name][destination] = values[matched]
    if not np.all(found):
        raise ValueError('Frozen rows missing in metadata')
    return output


def label_phase_table(labels, phases, split, source):
    """Count each target class within each integer second phase."""
    labels, phases = np.asarray(labels), np.asarray(phases)
    if len(labels) != len(phases):
        raise ValueError('Different row counts')
    counts = np.bincount(phases.astype(np.int64) * 3 + labels, minlength=15).reshape(5, 3)
    rows = []
    for phase, values in enumerate(counts):
        total = int(values.sum())
        row = {'split': split, 'source': source, 'phase': phase, 'rows': total}
        for label, value in enumerate(values):
            row[f'label_{label}_count'] = int(value)
            row[f'label_{label}_share'] = float(value / total) if total else None
        rows.append(row)
    return pd.DataFrame(rows)


def quantize(values, borders):
    """CatBoost finite bins: each border contributes when value > border."""
    return np.searchsorted(np.asarray(borders), np.asarray(values), side='left')


def singleton_integer_bins(borders):
    """Integer nonnegative seconds in each frozen bin, including edge bins."""
    borders = np.asarray(borders, dtype=float)
    if np.any(np.diff(borders) <= 0):
        raise ValueError('Borders must increase')
    rows = []
    for index in range(len(borders) + 1):
        lower = float(borders[index - 1]) if index else None
        upper = float(borders[index]) if index < len(borders) else None
        first = max(0, int(np.floor(lower)) + 1) if lower is not None else 0
        last = int(np.floor(upper)) if upper is not None else None
        count = max(0, last - first + 1) if last is not None else None
        rows.append({'bin': index, 'lower_exclusive': lower, 'upper_inclusive': upper,
                     'first_nonnegative_integer': first, 'last_integer': last,
                     'integer_count': count, 'singleton_integer_second': count == 1})
    return pd.DataFrame(rows)


def quantizer_phase_sensitivity(freshness, borders, unit="seconds"):
    """Whether phases 0 and 4 within one five-second cell reach different bins."""
    values = np.asarray(freshness, dtype=float)
    present = np.isfinite(values)
    result = np.zeros(len(values), dtype=bool)
    base = 5 * np.floor(values[present] / 5)
    if unit == 'days':
        left = np.asarray(base / 86400, dtype='float32')
        right = np.asarray((base + 4) / 86400, dtype='float32')
    elif unit == 'seconds':
        left, right = base, base + 4
    else:
        raise ValueError('Unknown freshness unit')
    result[present] = quantize(left, borders) != quantize(right, borders)
    return result


def frequency_probabilities(train_keys, train_labels, val_keys, alpha=1.0):
    """Train-only smoothed frequencies; unseen keys use the train global prior."""
    train_keys, val_keys = np.asarray(train_keys), np.asarray(val_keys)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    keys, inverse = np.unique(train_keys, return_inverse=True)
    counts = np.zeros((len(keys), 3), dtype=np.int64)
    np.add.at(counts, (inverse, train_labels), 1)
    prior = (np.bincount(train_labels, minlength=3) + alpha) / (len(train_labels) + 3 * alpha)
    probabilities = np.broadcast_to(prior, (len(val_keys), 3)).copy()
    indices = np.searchsorted(keys, val_keys)
    known = indices < len(keys)
    known[known] &= keys[indices[known]] == val_keys[known]
    selected = counts[indices[known]]
    probabilities[known] = (selected + alpha) / (selected.sum(axis=1, keepdims=True) + 3 * alpha)
    return probabilities, int((~known).sum()), int(len(keys))


def lookup_diagnostics(train, val):
    """Fixed diagnostic lookup tables; no fitting, tuning, or ranking metrics."""
    train_length = train['track_length_seconds'].astype(np.int64)
    val_length = val['track_length_seconds'].astype(np.int64)
    train_phase = train['time'].astype(np.int64) % 5
    val_phase = val['time'].astype(np.int64) % 5
    candidates = [('global', np.zeros(len(train_length), dtype=int), np.zeros(len(val_length), dtype=int)),
                  ('track_length', train_length, val_length),
                  ('phase', train_phase, val_phase),
                  ('track_length_and_phase', train_length * 5 + train_phase, val_length * 5 + val_phase)]
    rows = []
    for name, train_keys, val_keys in candidates:
        probabilities, unseen, unique_keys = frequency_probabilities(train_keys, train['label'], val_keys)
        observed = probabilities[np.arange(len(val_phase)), val['label'].astype(int)]
        rows.append({'diagnostic': name, 'alpha': 1.0, 'train_keys': unique_keys,
                     'val_unseen_rows': unseen, 'validation_rows': len(val_phase),
                     'multiclass_logloss': float(-np.log(observed).mean()),
                     'skip_vs_other_roc_auc': float(roc_auc_score(val['label'] == 0, probabilities[:, 0])),
                     'diagnostic_not_model_comparison': True})
        del probabilities
        gc.collect()
    return pd.DataFrame(rows)


def split_summary(metadata, session, borders, dfs_borders, split):
    """Check phase invariants and quantizer sensitivity without model scores."""
    start = metadata['time'].astype(np.int64)
    end = metadata['end_ts'].astype(np.int64)
    length = metadata['track_length_seconds'].astype(np.int64)
    labels = metadata['label']
    freshness = np.asarray(session[:, 6], dtype=float)
    present = np.isfinite(freshness)
    rounded = np.rint(freshness[present]).astype(np.int64)
    start_phase = start % 5
    fresh_phase = rounded % 5
    exactly_representable = freshness[present] <= 2 ** 24
    sensitive = quantizer_phase_sensitivity(freshness, borders)
    dfs_sensitive = quantizer_phase_sensitivity(freshness, dfs_borders, unit='days')
    bins = singleton_integer_bins(borders)
    singleton = bins['singleton_integer_second'].to_numpy(dtype=bool)
    bin_ids = quantize(freshness[present], borders)
    row = {'split': split, 'rows': len(start), 'groups': int(len(np.unique(metadata['group_id']))),
           'end_not_multiple_5': int(np.count_nonzero(end % 5)),
           'track_length_not_multiple_5': int(np.count_nonzero(length % 5)),
           'start_not_multiple_5': int(np.count_nonzero(start_phase)),
           'start_not_multiple_5_share': float(np.mean(start_phase != 0)),
           'freshness_nan': int(np.count_nonzero(~present)),
           'freshness_present': int(np.count_nonzero(present)),
           'freshness_non_integer_present': int(np.count_nonzero(freshness[present] != rounded)),
           'phase_equal_present': int(np.count_nonzero(start_phase[present] == fresh_phase)),
           'phase_equal_present_share': float(np.mean(start_phase[present] == fresh_phase)),
           'phase_equal_up_to_float32_exact_integer': int(np.count_nonzero((start_phase[present] == fresh_phase) & exactly_representable)),
           'present_up_to_float32_exact_integer': int(np.count_nonzero(exactly_representable)),
           'inferred_last_end_not_multiple_5': int(np.count_nonzero((start[present] - rounded) % 5)),
           'freshness_singleton_integer_bin': int(np.count_nonzero(singleton[bin_ids])),
           'freshness_singleton_integer_bin_present_share': float(np.mean(singleton[bin_ids])),
           'freshness_phase_sensitive_quantization': int(np.count_nonzero(sensitive)),
           'freshness_phase_sensitive_present_share': float(np.mean(sensitive[present])),
           'dfs_freshness_phase_sensitive_quantization': int(np.count_nonzero(dfs_sensitive)),
           'dfs_freshness_phase_sensitive_present_share': float(np.mean(dfs_sensitive[present]))}
    phase_tables = [label_phase_table(labels, start_phase, split, 'start_ts_mod5'),
                    label_phase_table(labels[present], fresh_phase, split, 'freshness_mod5_present')]
    sensitivity_rows = []
    for quantizer_name, mask_sensitive in [('manual49_seconds', sensitive),
                                           ('dfs462_float32_days', dfs_sensitive)]:
        for label in [None, 0, 1, 2]:
            for short in [False, True]:
                mask = present & ((labels == label) if label is not None else True)
                if short:
                    mask &= freshness <= 300
                sensitivity_rows.append({'split': split, 'quantizer': quantizer_name,
                                         'label': 'all' if label is None else label,
                                         'freshness_seconds_at_most_300': short,
                                         'present_rows': int(np.count_nonzero(mask)),
                                         'phase_sensitive_rows': int(np.count_nonzero(mask_sensitive & mask)),
                                         'phase_sensitive_share': float(np.mean(mask_sensitive[mask])) if np.any(mask) else None})
    return row, pd.concat(phase_tables, ignore_index=True), pd.DataFrame(sensitivity_rows)


def main(output=None, controls_run=CONTROL_NAME, plan_path=None):
    controls_run = run_name(controls_run)
    started = time.perf_counter()
    root = TASK_ROOT
    result = root / 'results/experiment-1/temporal_controls' / controls_run
    data = root / 'data/experiment-1/temporal_controls' / controls_run
    destination = Path(output) if output else root / 'results/analysis' / RESULT_NAME
    if destination.exists():
        raise FileExistsError('Audit results must use a new directory')
    selection_report = json.loads((result / 'selection.json').read_text())
    border_path = data / 'manual_quantized/train_borders.tsv'
    lines = [line.split('\t') for line in border_path.read_text().splitlines()]
    borders = np.array([float(fields[1]) for fields in lines if fields[0] == '48'])
    if len(borders) != 254:
        raise AssertionError('Unexpected frozen freshness border count')
    plan_path = project_path(plan_path) if plan_path else root / 'results/test_plans/final_test_20261023_v1/plan.json'
    plan = json.loads(plan_path.read_text())
    dfs_border_path = saved_path(plan['train_borders']['dfs462']['path'])
    if sha256(dfs_border_path) != plan['train_borders']['dfs462']['sha256']:
        raise AssertionError('Frozen DFS border hash mismatch')
    dfs_lines = [line.split('\t') for line in dfs_border_path.read_text().splitlines()]
    dfs_column = plan['schemas']['dfs462'].index(
        'Users__TIME_SINCE_LAST(Interactions.end_ts, unit=days)')
    dfs_borders = np.array([float(fields[1]) for fields in dfs_lines if fields[0] == str(dfs_column)])
    if len(dfs_borders) == 0:
        raise AssertionError('Missing DFS User freshness borders')
    metadata = {}
    summaries, phase_tables, sensitivity_tables, inputs = [], [], [], {}
    for split in ['train', 'val']:
        selection_path = data / 'selection' / f'{split}.npz'
        if sha256(selection_path) != selection_report[split]['sha256']:
            raise AssertionError('Frozen selection hash mismatch')
        with np.load(selection_path) as source:
            selected = {key: source[key] for key in ['label', 'group_id', 'log_id']}
        rows, groups = selection_report[split]['rows'], selection_report[split]['groups']
        if len(selected['label']) != rows or len(np.unique(selected['group_id'])) != groups:
            raise AssertionError('Frozen sample size mismatch')
        _, metadata_path = _source_paths(split)
        current = read_metadata(metadata_path, selected['log_id'])
        for key in ['label', 'group_id', 'log_id']:
            np.testing.assert_array_equal(current[key], selected[key])
        session_path = data / 'manual_sessions' / f'{split}_session_X.npy'
        session = np.load(session_path, mmap_mode='r')
        if session.shape != (rows, 7):
            raise AssertionError('Unexpected frozen session schema')
        summary, phases, sensitivity = split_summary(current, session, borders, dfs_borders, split)
        summaries.append(summary)
        phase_tables.append(phases)
        sensitivity_tables.append(sensitivity)
        metadata[split] = {key: current[key] for key in ['time', 'track_length_seconds', 'label']}
        inputs[split] = {'selection': str(selection_path), 'selection_sha256': sha256(selection_path),
                         'metadata': str(metadata_path), 'metadata_size_bytes': metadata_path.stat().st_size,
                         'session': str(session_path),
                         'session_recorded_sha256': json.loads((result / f'manual_sessions_{split}.json').read_text())['session'].get('sha256'),
                         'session_actual_sha256': sha256(session_path)}
        print(json.dumps(summary), flush=True)
        del current, selected, session
        gc.collect()
    lookup = lookup_diagnostics(metadata['train'], metadata['val'])
    destination.mkdir(parents=True)
    pd.concat(phase_tables, ignore_index=True).to_csv(destination / 'phase_label_counts.csv', index=False)
    pd.concat(sensitivity_tables, ignore_index=True).to_csv(destination / 'quantizer_phase_sensitivity.csv', index=False)
    singleton_integer_bins(borders).to_csv(destination / 'freshness_integer_bins.csv', index=False)
    pd.DataFrame({'feature_index': 48, 'border_seconds': borders}).to_csv(destination / 'freshness_borders.csv', index=False)
    lookup.to_csv(destination / 'lookup_validation_diagnostics.csv', index=False)
    pd.DataFrame({'feature_index': dfs_column, 'border_days': dfs_borders,
                  'border_seconds': dfs_borders * 86400}).to_csv(destination / 'dfs_freshness_borders.csv', index=False)
    report = {'completed': True, 'test_read': False, 'model_fit': False, 'model_predict': False,
              'diagnostic_not_ranking_result': True, 'smoothing_alpha': 1.0,
              'phase_formula': 'start=floor(end-length*ratio/100-5); for end%5=0: start%5=-ceil(length*ratio/100)%5',
              'caveat': 'Target-derived phase association does not identify the causal fraction of the CatBoost gain; true-start timing and legal behavior also matter.',
              'source_sha256': sha256(__file__), 'borders_sha256': sha256(border_path),
              'dfs_borders_path': str(dfs_border_path), 'dfs_borders_sha256': sha256(dfs_border_path),
              'dfs_quantizer_rule': f'searchsorted fixed column{dfs_column} borders on float32(F_seconds / 86400)',
              'inputs': inputs, 'splits': summaries, 'lookup_diagnostics': lookup.to_dict('records'),
              'elapsed_seconds': time.perf_counter() - started}
    (destination / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'destination': str(destination), 'lookup': lookup.to_dict('records')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--controls-run', default=CONTROL_NAME)
    parser.add_argument('--plan')
    parser.add_argument('--legacy-run')
    args = parser.parse_args()
    configure_legacy(args.legacy_run)
    main(args.output, args.controls_run, args.plan)
