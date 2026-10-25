"""Переобучить временной DFS на признаках, использованных широкой моделью."""
from pathlib import Path
import argparse
import json
import re
import resource
import tempfile
import time

from catboost import CatBoostRanker
import numpy as np
import pandas as pd

from temporal_feature_selection import (
    _fit_and_save, _pool, _sha256, _target_fingerprint, _write_json)
from run_temporal_selection import start_guard
from replay_paths import saved_path

ROOT = Path(__file__).resolve().parents[1]


def used_feature_indices(model_json, feature_names):
    """Перевести числовые индексы сплитов в исходные столбцы, проверяя имена."""
    floats = {feature['feature_index']: feature
              for feature in model_json['features_info']['float_features']}
    indices = set()
    for tree in model_json['oblivious_trees']:
        for split in tree.get('splits', []):
            if split['split_type'] != 'FloatFeature':
                raise NotImplementedError('Эта проверка поддерживает только числовые признаки')
            feature = floats[split['float_feature_index']]
            flat = feature['flat_feature_index']
            if not 0 <= flat < len(feature_names) or feature['feature_id'] != feature_names[flat]:
                raise ValueError('Индекс или имя признака не соответствует исходной матрице')
            indices.add(flat)
    if not indices:
        raise ValueError('В модели нет признаков, использованных в сплитах')
    return sorted(indices)


def read_source(source_name):
    """Проверить исходный запуск и прочитать его матрицы, цели и модель."""
    relative = Path('experiment-1/temporal_runs') / source_name
    results = ROOT / 'results' / relative
    data = ROOT / 'data' / relative
    if not json.loads((results / 'status.json').read_text())['completed']:
        raise ValueError('Исходный запуск еще не завершен')
    prepared = json.loads((results / 'preparation.json').read_text())
    for name, checksum in prepared['data_artifacts'].items():
        if _sha256(data / name) != checksum:
            raise ValueError('Изменился исходный артефакт: ' + name)
    config = json.loads((results / 'selection/protocol.json').read_text())
    reference = json.loads((results / 'selection/models/all_features/metrics.json').read_text())
    reference['model_path'] = str(saved_path(reference['model_path']))
    if _sha256(reference['model_path']) != reference['model_sha256']:
        raise ValueError('Изменилась исходная широкая модель')
    model = CatBoostRanker().load_model(reference['model_path'])
    names = prepared['feature_names']
    if model.feature_names_ != names:
        raise ValueError('Порядок признаков модели и матрицы различается')
    targets = {}
    for split in ['train', 'val']:
        with np.load(data / f'{split}_targets.npz') as arrays:
            targets[split] = {name: arrays[name] for name in arrays.files}
        if _target_fingerprint(targets[split]) != config[f'{split}_targets_sha256']:
            raise ValueError('Изменились целевые строки или их порядок')
    matrices = {split: np.load(data / f'{split}_X.npy', mmap_mode='r')
                for split in ['train', 'val']}
    return results, prepared, config, reference, model, matrices, targets


def compare_used_features(source_name, run_name):
    """Сохранить отдельное сравнение с новым fit при прежних строках и параметрах."""
    for name in [source_name, run_name]:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
            raise ValueError('Неверное имя запуска')
    output = ROOT / 'results/experiment-1/temporal_runs' / run_name
    if output.exists():
        raise FileExistsError('Результат уже существует: нужно новое имя запуска')
    output.mkdir(parents=True)
    started = time.monotonic()
    stop = start_guard(output, started, 5)
    _write_json(output / 'status.json', {'completed': False, 'phase': 'loading'})
    try:
        source, prepared, config, reference, model, matrices, targets = read_source(source_name)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            model.save_model(str(path), format='json')
            indices = used_feature_indices(json.loads(path.read_text()), prepared['feature_names'])
        names = [prepared['feature_names'][index] for index in indices]
        parameters = config['model_parameters']
        protocol = {'source_run': source_name, 'source_model_sha256': reference['model_sha256'],
                    'rule': 'all unique features used in the saved wide model splits',
                    'original_feature_count': len(prepared['feature_names']),
                    'retained_feature_count': len(names), 'indices': indices, 'features': names,
                    'feature_order': 'original flat column order', 'model_parameters': parameters,
                    'train_targets_sha256': _target_fingerprint(targets['train']),
                    'val_targets_sha256': _target_fingerprint(targets['val']),
                    'data_artifacts': prepared['data_artifacts'],
                    'source_sha256': {name: _sha256(ROOT / 'src' / name) for name in
                        ['check_used_temporal_features.py', 'temporal_feature_selection.py',
                         'run_temporal_selection.py', 'train_dfs_catboost.py']},
                    'test_used': False,
                    'limitations': ['Same small validation reused during exploratory selection.',
                                    'Only one training seed; no full-data refit.']}
        _write_json(output / 'protocol.json', protocol)
        train_pool = _pool(matrices['train'], targets['train'], names,
                           parameters['thread_count'], indices)
        val_pool = _pool(matrices['val'], targets['val'], names,
                         parameters['thread_count'], indices)
        # Проверяем извлечение отдельно от нового обучения: старая модель видит те же входы.
        reference_scores = np.load(source / 'selection/models/all_features/val_score.npy')
        np.testing.assert_array_equal(
            reference_scores, model.predict(val_pool, thread_count=parameters['thread_count']))
        _write_json(output / 'status.json', {'completed': False, 'phase': 'refit',
                                             'retained_feature_count': len(names)})
        record, refitted = _fit_and_save(
            train_pool, val_pool, targets['val'], names, indices,
            output / 'models/used_in_trees', parameters)
        old_ndcg = reference['metrics']['NDCG@10']
        new_ndcg = record['metrics']['NDCG@10']
        rows = [{'model': 'all_features', 'features': reference['k'], **reference['metrics'],
                 'fit_seconds': reference['fit_seconds'], 'tree_count': reference['tree_count']},
                {'model': 'used_in_trees_refit', 'features': len(names), **record['metrics'],
                 'fit_seconds': record['fit_seconds'], 'tree_count': record['tree_count']}]
        pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
        summary = {'completed': True, 'source_run': source_name, 'features': len(names),
                   'reference_features': reference['k'], 'metrics': record['metrics'],
                   'reference_metrics': reference['metrics'],
                   'NDCG_relative_change': new_ndcg / old_ndcg - 1,
                   'NDCG_absolute_change': new_ndcg - old_ndcg,
                   'fit_seconds': record['fit_seconds'], 'reference_fit_seconds': reference['fit_seconds'],
                   'best_iteration': record['best_iteration'], 'tree_count': record['tree_count'],
                   'original_model_subset_predictions_exact': True,
                   'reload_predictions_exact': record['reload_predictions_exact'],
                   'model_path': record['model_path'], 'model_sha256': record['model_sha256'],
                   'elapsed_seconds': time.monotonic() - started,
                   'max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                   'test_used': False}
        _write_json(output / 'result.json', summary)
        _write_json(output / 'status.json', {**summary, 'phase': 'finished'})
        return summary
    except Exception as error:
        _write_json(output / 'status.json', {'completed': False, 'phase': 'failed',
                                             'error': repr(error)})
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default='temporal_topk_20261023_v2')
    parser.add_argument('--run-name', default='temporal_used_20261023_v1')
    args = parser.parse_args()
    print(json.dumps(compare_used_features(args.source_run, args.run_name),
                     ensure_ascii=False, indent=2))
