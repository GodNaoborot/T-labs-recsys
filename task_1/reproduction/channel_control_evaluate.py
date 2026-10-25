"""Portable publication of the original manual48 audit evaluation.

Default: print the plan. --execute explicitly predicts on the previously opened
test using the fixed model. Run names are set through TLAB_CHANNEL_TRAINING_RUN,
TLAB_CHANNEL_REFERENCE_TRAINING_RUN, TLAB_CHANNEL_TEST_RUN and
TLAB_CHANNEL_CONTROL_RUN. No model fitting or tuning is performed here.
"""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'task_1'


def run_name(variable, default):
    value = os.environ.get(variable, default)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Invalid run name in {variable}: {value!r}')
    return value


TRAIN_RUN = run_name('TLAB_CHANNEL_TRAINING_RUN', 'c4_packet_20261024_v1')
REFERENCE_RUN = run_name('TLAB_CHANNEL_REFERENCE_TRAINING_RUN', 'c4_packet_20261024_v1')
TEST_RUN = run_name('TLAB_CHANNEL_TEST_RUN', 'c4_test_20261024_v1')
CONTROL_RUN = run_name('TLAB_CHANNEL_CONTROL_RUN', 'channel_source_repeat')
if CONTROL_RUN == 'no_current_channel':
    raise ValueError('The imported no_current_channel reports cannot be overwritten')
OUT = TASK / 'results/c4' / CONTROL_RUN
DATA = TASK / 'data/c4' / TRAIN_RUN
RESULTS = TASK / 'results/c4' / TRAIN_RUN
REFERENCE = TASK / 'results/c4' / REFERENCE_RUN
TEST_RESULTS = TASK / 'results/c4_test' / TEST_RUN
TEST_DATA = TASK / 'data/c4_test' / TEST_RUN
PY = Path(sys.executable)


def intervals(candidate, reference):
    pd.testing.assert_frame_equal(candidate[['group_id', 'day', 'group_size']],
                                  reference[['group_id', 'day', 'group_size']])
    frame = candidate[['day']].copy()
    frame['delta'] = candidate.ndcg.to_numpy() - reference.ndcg.to_numpy()
    daily = frame.groupby('day').agg(total=('delta', 'sum'), count=('delta', 'size'))
    daily = daily.reindex(range(int(frame.day.min()), int(frame.day.max()) + 1), fill_value=0)
    records = []
    for block in [3, 4]:
        n = len(daily)
        starts = np.random.default_rng(42).integers(n, size=(20000, (n + block - 1) // block))
        ids = ((starts[..., None] + np.arange(block)) % n).reshape(20000, -1)[:, :n]
        draws = daily.total.to_numpy()[ids].sum(1) / daily['count'].to_numpy()[ids].sum(1)
        lo, hi = np.quantile(draws, [.025, .975])
        records.append({'block_days': block, 'delta_ndcg': float(frame.delta.mean()),
                        'ci_low': float(lo), 'ci_high': float(hi), 'replicates': 20000, 'seed': 42})
    return records


def main():
    while not (OUT / 'result.json').exists():
        state = json.loads((OUT / 'status.json').read_text())
        if state.get('stage') == 'failed':
            raise RuntimeError('Control training failed')
        time.sleep(3)
    fit = json.loads((OUT / 'result.json').read_text())
    model = CatBoostRanker().load_model(str(OUT / 'model/model.cbm'))
    names = model.feature_names_
    assert model.get_all_params()['ignored_features'] == [names.index('is_organic')]
    fresh_borders = RESULTS / 'quantized/manual49/train_borders.tsv'
    old_borders = REFERENCE / 'quantized/manual49/train_borders.tsv'
    assert fresh_borders.read_bytes() == old_borders.read_bytes()
    # The already-opened original test qbin uses these exact borders and rows.
    folder = TEST_RESULTS / 'quantized/manual49'
    pool = Pool('quantized://' + str(folder / 'test.qbin'), thread_count=4)
    with np.load(TEST_DATA / 'targets.npz') as source:
        targets = {k: source[k] for k in ['label', 'group_id', 'log_id', 'day']}
    np.testing.assert_array_equal(np.asarray(pool.get_label(), dtype='float64'), targets['label'])
    assert pool.get_feature_names() == names
    digest = file_hash(OUT / 'model/model.cbm')
    score = model.predict(pool, thread_count=4)
    metrics = ranking_metrics(targets['label'], targets['group_id'], score)
    starts = np.r_[0, np.flatnonzero(np.diff(targets['group_id'])) + 1]
    test_groups = _group_scores(targets, starts, score, metrics['NDCG@10'])
    np.save(OUT / 'test_score.npy', score)
    test_groups.to_parquet(OUT / 'test_group_ndcg.parquet', index=False)
    assert file_hash(OUT / 'model/model.cbm') == digest
    val_groups = pd.read_parquet(OUT / 'model/group_ndcg.parquet')
    comparisons = {}
    for split, candidate, reference in [
        ('validation', val_groups, REFERENCE / 'models/D462-t/group_ndcg.parquet'),
        ('test', test_groups, TEST_RESULTS / 'models/D462-t/group_ndcg.parquet'),
    ]:
        comparisons[split] = intervals(candidate, pd.read_parquet(reference))
    result = {'completed': True, 'active_features': 48, 'excluded': 'current target is_organic',
        'validation_metrics': fit['evaluation']['metrics'], 'test_metrics': metrics,
        'tree_count': int(model.tree_count_), 'fit_seconds': fit['fit_seconds'],
        'manual_minus_published_dfs462': comparisons,
        'test_already_open': True, 'new_blind_test': False,
        'test_pool_reused_from_original_run': True, 'same_train_borders_verified': True,
        'model_hash_unchanged': True, 'new_hyperparameter_search': False,
        'note': 'Independent audit control; original test features/pool reused only after matching borders. Main refit still verified separately.'}
    (OUT / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode': 'plan', 'training_run': TRAIN_RUN,
                          'reference_run': REFERENCE_RUN, 'test_run': TEST_RUN,
                          'output': str((OUT / 'evaluation.json').relative_to(ROOT)),
                          'block_days': [3, 4], 'replicates': 20000, 'seed': 42,
                          'execute_required': True}, indent=2))
        return
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists; choose a fresh control run')
    global np, pd, CatBoostRanker, Pool, file_hash, ranking_metrics, _group_scores
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRanker, Pool
    sys.path.insert(0, str(TASK / 'src'))
    from project_artifacts import file_hash
    from train_dfs_catboost import ranking_metrics
    from optuna_evaluation import _group_scores
    main()


if __name__ == '__main__':
    cli()
