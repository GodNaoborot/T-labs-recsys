"""Тот же контроль секундной фазы для готовой DFS-модели, порциями validation."""
from pathlib import Path
import argparse
import gc
import json
import time

from catboost import CatBoostRanker, Pool
import numpy as np

from audit_start_time_phase_permutation import (
    ROOT, RUN, hash_file, permute_phase, ranking_metrics,
    _group_scores, aggregate_days, circular_day_indices, absolute_delta_draws)

from replay_paths import project_path, saved_path, run_name


def main(output=None, controls_run=RUN, full_run='temporal_full462_20261023_v5'):
    controls_run, full_run = run_name(controls_run), run_name(full_run)
    output = project_path(output) if output else ROOT / 'results/analysis/start_ts_dfs_phase_permutation_20261024_v1'
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    run = ROOT / 'results/experiment-1/temporal_controls' / controls_run
    data = ROOT / 'data/experiment-1/temporal_controls' / controls_run
    control = json.loads((run / 'models/D462-sub/control.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    record = {'model': {'path': str(run / 'models/D462-sub/model.cbm'),
                        'sha256': control['model_sha256']},
              'tree_count': control['evaluation']['tree_count']}
    model_path = saved_path(record['model']['path'])
    if hash_file(model_path) != record['model']['sha256']:
        raise ValueError('Модель изменилась')
    names = protocol['dfs_features']
    column = names.index('Users__TIME_SINCE_LAST(Interactions.end_ts, unit=days)')
    val_full = ROOT / 'results/experiment-1/temporal_runs' / full_run
    quantization = json.loads((val_full / 'quantization_val.json').read_text())
    with np.load(data / 'selection/val.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ['label', 'group_id', 'log_id', 'day']}
    with np.load(saved_path(quantization['targets_path']), allow_pickle=False) as archive:
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], archive[name])
    raw = np.load(saved_path(quantization['matrix_path']), mmap_mode='r', allow_pickle=False)
    manual = np.load(data / 'manual_sessions/val_X.npy', mmap_mode='r', allow_pickle=False)
    manual_names = protocol['manual_features'] + protocol['session_features']
    freshness = np.asarray(manual[:, manual_names.index('user_time_since_last')]).copy()
    changed = permute_phase(freshness, manual[:, manual_names.index('track_length_seconds')])
    altered = np.isfinite(freshness) & (freshness != changed)
    np.testing.assert_array_equal(raw[altered, column],
                                  (freshness[altered].astype('float64') / 86400).astype('float32'))
    model = CatBoostRanker().load_model(str(model_path))
    if model.feature_names_ != names or model.tree_count_ != record['tree_count']:
        raise ValueError('Схема модели отличается от зафиксированной')
    baseline = np.empty(len(freshness), dtype='float64')
    modified_score = np.empty_like(baseline)
    expected_score = np.load(run / 'models/D462-sub/val_score.npy', mmap_mode='r')
    for left in range(0, len(freshness), 100_000):
        right = min(left + 100_000, len(freshness))
        block = np.array(raw[left:right], dtype='float32', order='F', copy=True)
        pool = Pool(block, feature_names=names, thread_count=4)
        baseline[left:right] = model.predict(pool, thread_count=4)
        np.testing.assert_allclose(baseline[left:right], expected_score[left:right], rtol=0, atol=1e-12)
        del pool
        modified = block.copy(order='F')
        del block
        local = altered[left:right]
        modified[local, column] = (changed[left:right][local].astype('float64') / 86400).astype('float32')
        pool = Pool(modified, feature_names=names, thread_count=4)
        modified_score[left:right] = model.predict(pool, thread_count=4)
        del pool, modified
        gc.collect()
    before = ranking_metrics(arrays['label'], arrays['group_id'], baseline)
    after = ranking_metrics(arrays['label'], arrays['group_id'], modified_score)
    starts = np.flatnonzero(np.r_[True, arrays['group_id'][1:] != arrays['group_id'][:-1]])
    bg = _group_scores(arrays, starts, baseline, before['NDCG@10'])
    mg = _group_scores(arrays, starts, modified_score, after['NDCG@10'])
    daily = aggregate_days(bg['day'], bg['ndcg'], mg['ndcg'])
    intervals = []
    for block in [3, 4]:
        indices = circular_day_indices(len(daily), block, 20_000, seed=42)
        low, high = np.quantile(absolute_delta_draws(daily, indices), [.025, .975])
        intervals.append({'block_days':block,'ci_low':float(low),'ci_high':float(high)})
    if hash_file(model_path) != record['model']['sha256']:
        raise ValueError('Модель изменилась во время проверки')
    result = {'completed':True,'split':'val','test_used':False,'training_performed':False,
        'model':'D462-sub frozen','feature_index':column,
        'intervention':'same conditional phase permutation as C2b; only User recency column changed',
        'changed_rows':int(altered.sum()),'max_freshness_change_seconds':4.0,
        'original_metrics':{k:float(v) for k,v in before.items()},
        'permuted_metrics':{k:float(v) for k,v in after.items()},
        'delta_ndcg':float(after['NDCG@10']-before['NDCG@10']),
        'paired_exploratory_intervals':intervals,
        'original_raw_predictions_match_saved':True,'model_sha256_unchanged':record['model']['sha256'],
        'source_sha256':hash_file(Path(__file__)),
        'elapsed_seconds':time.perf_counter()-started,
        'limitations':['post-hoc validation sensitivity, not refitted-model quality',
            'other feature relationships can be disrupted',
            'true start unobserved; no causal fraction of leakage',
            'other start-derived features and groups were not changed']}
    output.mkdir(parents=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    daily.to_csv(output/'daily_ndcg.csv',index=False)
    mg.to_parquet(output/'group_ndcg_permuted.parquet',index=False)
    (output/'audit_start_time_dfs_phase_permutation.py').write_text(Path(__file__).read_text())
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-run', default=RUN)
    parser.add_argument('--full-run', default='temporal_full462_20261023_v5')
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.output, args.controls_run, args.full_run)
