"""Фиксированная validation-диагностика секундной фазы: без обучения и test."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

from block_bootstrap import aggregate_days, circular_day_indices
from compare_temporal_controls_pairs import absolute_delta_draws
from optuna_evaluation import _group_scores, _validate_targets
from train_dfs_catboost import ranking_metrics

from replay_paths import project_path, run_name

ROOT = Path(__file__).resolve().parents[1]
RUN = 'dfs462_controls_20261023_v1'


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def permute_phase(freshness, length, seed=42):
    """Перемешать F внутри одинаковых длины трека и floor(F/5), без меток."""
    freshness, length = np.asarray(freshness), np.asarray(length)
    if freshness.shape != length.shape or freshness.ndim != 1:
        raise ValueError('Нужны два одномерных массива одинаковой длины')
    result = freshness.copy()
    positions = np.flatnonzero(np.isfinite(freshness))
    bucket = np.floor(freshness[positions] / 5).astype('int64')
    track_length = length[positions]
    canonical = np.lexsort((track_length, bucket))
    random = np.random.default_rng(seed).random(len(positions))
    shuffled = np.lexsort((random, track_length, bucket))
    np.testing.assert_array_equal(bucket[canonical], bucket[shuffled])
    np.testing.assert_array_equal(track_length[canonical], track_length[shuffled])
    result[positions[canonical]] = freshness[positions[shuffled]]
    return result


def main(output=None, controls_run=RUN):
    controls_run = run_name(controls_run)
    output = project_path(output) if output else ROOT / 'results/analysis/start_ts_phase_permutation_20261024_v1'
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    original_run = ROOT / 'results/experiment-1/temporal_controls' / controls_run
    data = ROOT / 'data/experiment-1/temporal_controls' / controls_run
    model_record = json.loads((original_run / 'models/C2b/control.json').read_text())
    features = json.loads((original_run / 'protocol.json').read_text())
    names = features['manual_features'] + features['session_features']
    model_path = original_run / 'models/C2b/model.cbm'
    expected_hash = model_record['model_sha256']
    if hash_file(model_path) != expected_hash:
        raise ValueError('Модель изменилась')
    with np.load(data / 'selection/val.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ['label', 'group_id', 'day', 'log_id']}
    raw_path = data / 'manual_sessions/val_X.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    if raw.shape != (len(arrays['label']), 49):
        raise ValueError('Неожиданная validation-схема')
    model = CatBoostRanker().load_model(str(model_path))
    if model.feature_names_ != names:
        raise ValueError('Порядок признаков изменился')
    raw_pool = Pool(raw, feature_names=names, thread_count=4)
    baseline_score = np.asarray(model.predict(raw_pool, thread_count=4))
    expected_score = np.load(original_run / 'models/C2b/val_score.npy', mmap_mode='r')
    np.testing.assert_allclose(baseline_score, expected_score, rtol=0, atol=1e-12)
    del raw_pool, expected_score
    original_freshness = np.asarray(raw[:, names.index('user_time_since_last')]).copy()
    changed = permute_phase(original_freshness, raw[:, names.index('track_length_seconds')])
    present = np.isfinite(original_freshness)
    modified = np.array(raw, dtype='float32', order='F', copy=True)
    modified[:, names.index('user_time_since_last')] = changed
    changed_pool = Pool(modified, feature_names=names, thread_count=4)
    score = np.asarray(model.predict(changed_pool, thread_count=4))
    del changed_pool, modified, raw
    gc.collect()
    group_starts = np.flatnonzero(np.r_[True, arrays['group_id'][1:] != arrays['group_id'][:-1]])
    base_metrics = ranking_metrics(arrays['label'], arrays['group_id'], baseline_score)
    metrics = ranking_metrics(arrays['label'], arrays['group_id'], score)
    base_groups = _group_scores(arrays, group_starts, baseline_score, base_metrics['NDCG@10'])
    groups = _group_scores(arrays, group_starts, score, metrics['NDCG@10'])
    daily = aggregate_days(groups['day'], base_groups['ndcg'], groups['ndcg'])
    intervals = []
    for block in [3, 4]:
        indices = circular_day_indices(len(daily), block, 20_000, seed=42)
        low, high = np.quantile(absolute_delta_draws(daily, indices), [.025, .975])
        intervals.append({'block_days': block, 'ci_low': float(low), 'ci_high': float(high)})
    if hash_file(model_path) != expected_hash:
        raise ValueError('Исходная модель изменена')
    record = {'completed': True, 'split': 'val', 'test_used': False,
        'training_performed': False, 'model': 'C2b frozen',
        'model_sha256_unchanged': expected_hash,
        'rows': len(score), 'groups': len(groups),
        'intervention': 'permutation within (current track_length, floor(user freshness/5)), seed42',
        'labels_used_for_permutation': False,
        'changed_fraction_present': float(np.mean(original_freshness[present] != changed[present])),
        'max_freshness_change_seconds': float(np.max(np.abs(original_freshness[present] - changed[present]))),
        'original_metrics': {key: float(value) for key,value in base_metrics.items()},
        'permuted_metrics': {key: float(value) for key,value in metrics.items()},
        'delta_ndcg': float(metrics['NDCG@10'] - base_metrics['NDCG@10']),
        'paired_exploratory_intervals': intervals,
        'source_sha256': hash_file(Path(__file__)), 'raw_validation_sha256': hash_file(raw_path),
        'elapsed_seconds': time.perf_counter()-started,
        'limitations': ['post-hoc validation sensitivity, not refitted-model quality',
            'other feature relationships can be disrupted',
            'does not prove inaccurate reconstruction against independently observed true start',
            'does not remove cutoff dependence from other features or group construction']}
    output.mkdir(parents=True)
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    daily.to_csv(output/'daily_ndcg.csv',index=False)
    groups.to_parquet(output/'group_ndcg_permuted.parquet',index=False)
    (output/'audit_start_time_phase_permutation.py').write_text(Path(__file__).read_text())
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-run', default=RUN)
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.output, args.controls_run)
