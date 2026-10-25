"""Сквозная проверка основного расчёта на искусственном журнале в новой копии."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from prepare_history import file_hash


TASK = Path(__file__).resolve().parents[1]


def make_raw(output):
    """Небольшой журнал с теми же полями и календарём, тремя метками и реакциями."""
    rows = []
    for user in range(6):
        for day in range(301):
            for slot in range(4):
                item = 1 + (day + slot + user) % 12
                timestamp = day * 86400 + 2000 + 400 * slot
                ratio = [10., 70., 100.][(day + slot + user) % 3]
                rows.append((user, item, timestamp, 'listen', ratio, 100. + 50 * (item % 4), slot % 2))
                if (day + slot + user) % 7 == 0:
                    rows.append((user, item, timestamp - 5, 'like', np.nan, np.nan, slot % 2))
    frame = pd.DataFrame(rows, columns=['uid', 'item_id', 'timestamp', 'event_type',
                                      'played_ratio_pct', 'track_length_seconds', 'is_organic'])
    frame = frame.sort_values(['uid', 'timestamp'], kind='stable').reset_index(drop=True)
    raw = output / 'multi_event.parquet'
    mapping = output / 'artist_item_mapping.parquet'
    frame.to_parquet(raw, index=False)
    pd.DataFrame({'item_id': range(1, 13), 'artist_id': [item // 3 for item in range(1, 13)]}).to_parquet(mapping, index=False)
    return raw, mapping, len(frame)


def copy_inputs(destination):
    """Копировать только код и малые спецификации, без истории и старых моделей."""
    target = destination / 'task_1'
    shutil.copytree(TASK / 'src', target / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(TASK / 'dataset_source.json', target / 'dataset_source.json')
    specifications = [
        Path('results/experiment-1/temporal_controls/dfs462_controls_20261023_v1/protocol.json'),
        Path('results/experiment-1/temporal_runs/temporal_used_20261023_v1/protocol.json'),
    ]
    specifications += list(path.relative_to(TASK) for path in
                           (TASK / 'data/experiment-1/temporal_runs/temporal_topk_20261023_v2/definitions').glob('*.json'))
    for name in specifications:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TASK / name, path)
    return target


def run_check(output_dir):
    """Подготовка, EDA, признаки, три fit, synthetic test и разбор важностей."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Нужна новая папка проверки: ' + str(output_dir))
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    records = []
    with tempfile.TemporaryDirectory(prefix='recsys-reproduction-') as temporary:
        workspace = Path(temporary)
        target = copy_inputs(workspace)
        raw, mapping, rows = make_raw(workspace)
        commands = [
            ('history', ['src/prepare_history.py', '--raw-events', str(raw), '--artist-mapping', str(mapping)]),
            ('eda', ['src/analyze_dataset.py', '--raw-events', str(raw), '--output-dir', str(target / 'eda')]),
            ('train', ['src/run_c4.py', '--run-name', 'synthetic', '--train-groups', '12', '--max-training-seconds', '300']),
            ('synthetic-test', ['src/run_c4_test.py', '--run-name', 'synthetic_test', '--training-run', 'synthetic']),
            ('features', ['src/summarize_selected_features.py', '--training-run', 'synthetic',
                          '--output-dir', str(target / 'feature-analysis')]),
        ]
        model_hashes = None
        for stage, arguments in commands:
            print('Проверка:', stage, flush=True)
            stage_started = time.monotonic()
            log = output_dir / (stage + '.log')
            with log.open('w') as stream:
                result = subprocess.run([sys.executable, *arguments], cwd=target,
                                        stdout=stream, stderr=subprocess.STDOUT)
            record = {'stage': stage, 'exit_code': result.returncode,
                      'seconds': time.monotonic() - stage_started}
            records.append(record)
            if result.returncode:
                raise RuntimeError(f'Проверка {stage} завершилась с ошибкой; {log}')
            if stage == 'eda':
                snapshot = json.loads((target / 'eda/snapshot.json').read_text())
                assert snapshot['rows'] == rows
                assert all((target / 'eda' / image['file']).is_file() for image in snapshot['images'])
            if stage == 'train':
                model_hashes = {p.parent.name: file_hash(p) for p in (target / 'results/c4/synthetic/models').glob('*/model.cbm')}
                assert len(model_hashes) == 3
        after = {p.parent.name: file_hash(p) for p in (target / 'results/c4/synthetic/models').glob('*/model.cbm')}
        assert model_hashes == after
        training = target / 'results/c4/synthetic'
        evaluation = target / 'results/c4_test/synthetic_test'
        for folder in [training, evaluation]:
            assert all(json.loads((folder / name).read_text()).get('completed') is True
                       for name in ['status.json', 'result.json'])
        train_table = pd.read_csv(training / 'comparison.csv')
        test_table = pd.read_csv(evaluation / 'comparison.csv')
        assert train_table['variant'].tolist() == test_table['variant'].tolist() == ['M-t', 'C2b-t', 'D462-t']
        assert train_table['active_features'].tolist() == [42, 49, 462]
        assert np.isfinite(train_table[['NDCG@10', 'QueryAUC']]).all().all()
        assert np.isfinite(test_table[['NDCG@10', 'QueryAUC']]).all().all()
        assert len(pd.read_csv(target / 'feature-analysis/selected_features.csv')) == 462
        for name, folder in [('train', training), ('synthetic-test', evaluation)]:
            shutil.copy2(folder / 'comparison.csv', output_dir / (name + '-comparison.csv'))
            shutil.copy2(folder / 'bootstrap_summary.csv', output_dir / (name + '-intervals.csv'))
        proof = {'completed': True, 'profile': 'synthetic', 'raw_rows': rows,
                 'real_dataset_read': False, 'research_test_read': False,
                 'fresh_training_performed': True, 'model_hashes_unchanged_after_test': True,
                 'feature_counts': [42, 49, 462], 'stages': records,
                 'seconds': time.monotonic() - started,
                 'scope': 'actual production CLI chain on small artificial data; not a full Yambda rerun'}
        (output_dir / 'result.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
        return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_check(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
