"""Производные таблицы Optuna и парные интервалы без обучения и чтения test.

Относительный прирост равен (NDCG_DFS - NDCG_manual) / NDCG_manual.
Положительный знак означает улучшение DFS; интервалы условны на обученных моделях.
"""
from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd

from block_bootstrap import aggregate_days, circular_day_indices, relative_loss_draws
from bootstrap_intervals import compare_saved_intervals, _interval_rows
from project_artifacts import file_hash, load_current_artifacts

CONFIGURATIONS = [('fixed_600', None), ('control_1600', 'control'), ('tuned_1600', 'tuned')]
OBJECTIVES = ('ranker', 'pointwise')
METHODS = ('manual', 'dfs')
BLOCK_DAYS, RESAMPLES, SEED = (3, 4), 20_000, 42
GAIN_RULE = '(NDCG_DFS - NDCG_manual) / NDCG_manual; positive means DFS improves'
PAIR_FILES = ('groups.parquet', 'daily.csv', 'bootstrap_losses.npz', 'summary.csv')


def _write_json(path, value):
    """Атомарно заменить только собственный производный JSON без NaN."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _write_csv(path, frame):
    """Атомарно обновить собственную производную CSV-таблицу."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _expired(deadline):
    """Проверить абсолютный срок перед началом нового bootstrap."""
    return deadline is not None and time.time() >= float(deadline)


def _finite(value):
    """Оставить конечное число или None для отсутствующей метрики."""
    return float(value) if value is not None and np.isfinite(value) else None


def _recorded_models():
    """Прочитать четыре исходные модели 600 из канонических config/metrics без predict."""
    records = {}
    name = os.environ.get('TLAB_EXPERIMENT_RUN')
    sources = load_current_artifacts()
    if name is not None:
        root = Path(__file__).resolve().parents[1]
        folders = {'manual': root / 'results/baselines/runs' / name,
                   'dfs': root / 'results/experiment-1/runs' / name / 'dfs'}
        sources = {method: {'config': folder / 'run_config.json',
                            'metrics': folder / 'validation_metrics.csv'}
                   for method, folder in folders.items()}
    for method, files in sources.items():
        config = json.loads(files['config'].read_text())
        metrics = pd.read_csv(files['metrics'])
        for objective, token in [('ranker', 'YetiRank'), ('pointwise', 'MultiClass')]:
            row = metrics.loc[metrics['model'].str.contains(token, regex=False)].iloc[0]
            key = token if method == 'manual' else objective
            info = config['models'][key]
            values = {name: _finite(row.get(name)) for name in ('NDCG@10', 'QueryAUC', 'logloss')}
            result = {**info, 'metrics': values,
                      'params': info.get('actual_params', info.get('params', {}))}
            records[(method, objective)] = {'status': 'completed', 'result': result}
    return records


def _model_row(configuration, method, objective, stage):
    """Представить один этап, не принимая метрики незавершенного fit за результат."""
    status = stage.get('status', 'not_run')
    result = stage.get('result', {}) if status == 'completed' else {}
    metrics = result.get('metrics', {})
    row = {'configuration': configuration, 'method': method, 'objective': objective,
           'status': status, 'reused_from': stage.get('reused_from', result.get('reused_from')),
           'best_iteration': result.get('best_iteration'), 'tree_count': result.get('tree_count'),
           'fit_seconds': _finite(result.get('fit_seconds')),
           'params': json.dumps(result.get('params', {}), sort_keys=True, allow_nan=False)}
    row.update({name: _finite(metrics.get(name)) for name in ('NDCG@10', 'QueryAUC', 'logloss')})
    return row


def _model_rows(stages, recorded):
    """Собрать исходные, контрольные и подобранные модели, включая пропуски этапов."""
    rows = []
    for configuration, prefix in CONFIGURATIONS:
        for objective in OBJECTIVES:
            for method in METHODS:
                stage = (recorded[(method, objective)] if prefix is None
                         else stages.get(f'{prefix}_{method}_{objective}', {}))
                rows.append(_model_row(configuration, method, objective, stage))
    return rows


def _comparison_row(manual, dfs):
    """Показать D−M в пунктах NDCG и относительный прирост со знаком улучшения."""
    row = {'configuration': manual['configuration'], 'objective': manual['objective']}
    for method, model in [('manual', manual), ('dfs', dfs)]:
        for name in ('status', 'NDCG@10', 'QueryAUC', 'logloss', 'best_iteration', 'tree_count', 'fit_seconds', 'params'):
            row[f'{method}_{name}'] = model[name]
    base, value = manual['NDCG@10'], dfs['NDCG@10']
    row['ndcg_difference'] = None if base is None or value is None else value - base
    row['relative_gain'] = None if base is None or value is None or base <= 0 else value / base - 1.0
    return row


def _comparison_rows(models, intervals):
    """Добавить отдельные L3/L4 интервалы к сопоставлению полных моделей."""
    lookup = {(row['configuration'], row['objective'], row['method']): row for row in models}
    rows = []
    for configuration, _ in CONFIGURATIONS:
        for objective in OBJECTIVES:
            row = _comparison_row(lookup[(configuration, objective, 'manual')],
                                  lookup[(configuration, objective, 'dfs')])
            selected = [item for item in intervals if item['configuration'] == configuration and item['objective'] == objective]
            for item in selected:
                suffix = f"L{item['block_days']}"
                row[f'ci95_{suffix}_lower_relative_gain'] = item['ci_lower_relative_gain']
                row[f'ci95_{suffix}_upper_relative_gain'] = item['ci_upper_relative_gain']
            rows.append(row)
    return rows


def _recorded_intervals():
    """Прочитать старые реплики исходного val и обратить знак потери в прирост."""
    name = os.environ.get('TLAB_EXPERIMENT_RUN')
    if name is None:
        frame = compare_saved_intervals()
    else:
        root = Path(__file__).resolve().parents[1]
        folder = root / 'results/experiment-1/runs' / name / 'dfs/bootstrap_noninferiority'
        frame = pd.DataFrame(_interval_rows('original_val', folder))
    rows = []
    for source in frame.loc[frame['period'] == 'original_val'].to_dict('records'):
        rows.append({'configuration': 'fixed_600', 'objective': source['objective'],
                     'block_days': int(source['block_days']), 'relative_gain': -source['relative_loss'],
                     'ci_lower_relative_gain': -source['ci_upper'],
                     'ci_upper_relative_gain': -source['ci_lower'],
                     'upper95_relative_loss': source['upper_bound'],
                     'n_resamples': int(source['n_resamples']), 'source': source['source']})
    return rows


def _group_source(result):
    """Проверить хеш сохраненных групп и вернуть путь с его текущим SHA256."""
    path = Path(result['artifacts']['groups'])
    digest = file_hash(path)
    expected = result.get('artifact_sha256', {}).get('groups')
    if expected is not None and digest != expected:
        raise ValueError(f'Изменился файл групп: {path}')
    return {'path': str(path), 'sha256': digest, 'ndcg': float(result['metrics']['NDCG@10'])}


def _read_groups(source, metric):
    """Проверить уникальный порядок ranking-групп и совпадение среднего с метрикой."""
    frame = pd.read_parquet(source['path'])
    required = {'group_id', 'day', 'ndcg'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError('Нужна непустая таблица group_id/day/ndcg')
    if frame['group_id'].duplicated().any() or not frame['group_id'].is_monotonic_increasing:
        raise ValueError('group_id должен быть уникальным и отсортированным')
    if not np.isfinite(frame[['group_id', 'day', 'ndcg']].to_numpy()).all():
        raise ValueError('В групповых метриках должны быть конечные числа')
    np.testing.assert_allclose(frame['ndcg'].mean(), metric, rtol=0, atol=1e-12)
    return frame


def _paired_groups(sources, manual, dfs):
    """Проверить строгое соответствие групп и дней перед парным bootstrap."""
    base = _read_groups(sources['manual'], manual['metrics']['NDCG@10'])
    generated = _read_groups(sources['dfs'], dfs['metrics']['NDCG@10'])
    for name in ('group_id', 'day'):
        np.testing.assert_array_equal(base[name].to_numpy(), generated[name].to_numpy())
    if 'group_size' in base and 'group_size' in generated:
        np.testing.assert_array_equal(base['group_size'].to_numpy(), generated['group_size'].to_numpy())
    return pd.DataFrame({'group_id': base['group_id'].to_numpy(), 'day': base['day'].to_numpy(),
                         'manual_ndcg': base['ndcg'].to_numpy(), 'dfs_ndcg': generated['ndcg'].to_numpy()})


def _bootstrap_rows(configuration, objective, daily):
    """Использовать одни дневные индексы для M/D и обратить границы потери в прирост."""
    rows, draws = [], {}
    base_sum = float(daily['base_sum'].sum())
    if base_sum <= 0:
        raise ValueError('Сумма NDCG manual должна быть положительной')
    gain = float(daily['dfs_sum'].sum() / base_sum - 1.0)
    for block in BLOCK_DAYS:
        indices = circular_day_indices(len(daily), block, RESAMPLES, seed=SEED)
        loss = relative_loss_draws(daily, indices)
        low, high, upper = np.quantile(loss, [.025, .975, .95], method='linear')
        draws[f'loss_L{block}'] = loss
        rows.append({'configuration': configuration, 'objective': objective, 'block_days': block,
                     'relative_gain': gain, 'ci_lower_relative_gain': float(-high),
                     'ci_upper_relative_gain': float(-low), 'upper95_relative_loss': float(upper),
                     'n_resamples': RESAMPLES, 'groups': int(daily['group_count'].sum()), 'days': len(daily)})
    return rows, draws


def _bootstrap_settings():
    """Включить параметры ресемплирования в проверку кеша производного отчета."""
    return {'block_days': list(BLOCK_DAYS), 'n_resamples': RESAMPLES, 'seed': SEED,
            'interval': 'percentile q.025/q.975 of relative gain'}


def _cached_pair(folder, sources):
    """Повторно использовать только целые производные файлы тех же исходных групп."""
    path = folder / 'cache.json'
    if not path.is_file():
        return None
    cache = json.loads(path.read_text())
    if cache['sources'] != sources or cache.get('settings') != _bootstrap_settings():
        return None
    for filename, digest in cache['artifact_sha256'].items():
        file = folder / filename
        if not file.is_file() or file_hash(file) != digest:
            return None
    return pd.read_csv(folder / 'summary.csv').to_dict('records')


def _save_pair(folder, sources, groups, daily, rows, draws):
    """Сохранить собственные парные группы, дневные суммы и bootstrap-реплики."""
    folder.mkdir(parents=True, exist_ok=True)
    groups.to_parquet(folder / 'groups.parquet', index=False)
    _write_csv(folder / 'daily.csv', daily)
    temporary = folder / 'bootstrap_losses.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **draws)
    temporary.replace(folder / 'bootstrap_losses.npz')
    _write_csv(folder / 'summary.csv', pd.DataFrame(rows))
    _write_json(folder / 'cache.json', {'sources': sources, 'settings': _bootstrap_settings(),
                'artifact_sha256': {name: file_hash(folder / name) for name in PAIR_FILES}})


def _new_pair(folder, configuration, objective, manual, dfs, deadline):
    """Прочитать готовый кеш либо начать новый парный расчет до истечения срока."""
    sources = {'manual': _group_source(manual), 'dfs': _group_source(dfs)}
    cached = _cached_pair(folder, sources)
    if cached is not None:
        return cached, 'cached'
    if _expired(deadline):
        return [], 'skipped_deadline'
    groups = _paired_groups(sources, manual, dfs)
    daily = aggregate_days(groups['day'], groups['manual_ndcg'], groups['dfs_ndcg'])
    if _expired(deadline):
        return [], 'skipped_deadline'
    rows, draws = _bootstrap_rows(configuration, objective, daily)
    _save_pair(folder, sources, groups, daily, rows, draws)
    return rows, 'completed'


def _new_intervals(output, stages, deadline):
    """Публиковать CI только для завершенных пар; ошибки соответствия сохранять явно."""
    rows, statuses = [], {}
    for prefix in ('control', 'tuned'):
        for objective in OBJECTIVES:
            name = f'{prefix}_{objective}'
            pair = [stages.get(f'{prefix}_{method}_{objective}', {}) for method in METHODS]
            if not all(item.get('status') == 'completed' for item in pair):
                statuses[name] = {'status': 'incomplete_pair'}
                continue
            try:
                items, status = _new_pair(output / name, f'{prefix}_1600', objective,
                                          pair[0]['result'], pair[1]['result'], deadline)
                rows.extend(items)
                statuses[name] = {'status': status}
            except (ValueError, AssertionError, OSError, KeyError) as error:
                statuses[name] = {'status': 'invalid_pair', 'error': str(error)}
    return rows, statuses


def _report_manifest(output, statuses, deadline, rows):
    """Описать знаки, ограничения и файлы без объявления независимости val после поиска."""
    files = {'models': 'models.csv', 'comparison': 'summary.csv',
             'bootstrap': 'bootstrap_summary.csv', 'manifest': 'manifest.json'}
    return {'artifacts': {name: str(output / file) for name, file in files.items()},
            'comparison_rows': len(rows), 'pairs': statuses, 'deadline_expired': _expired(deadline),
            'relative_gain_definition': GAIN_RULE, 'test_used': False, 'training_started': False,
            'limitations': ['Validation reused for early stopping and parameter selection; exploratory CI.',
                           'Bootstrap conditional on fitted models; training seed uncertainty omitted.',
                           '600→1600 also changes patience 50→80: extended training policy, not isolated cap.',
                           'Fit seconds describe observed runs; tree counts and runtime conditions may differ.']}


def publish_reports(paths, stages, deadline_unix=None):
    """Обновить только comparison/ отдельного tuning-запуска без fit и test."""
    output = Path(paths['results']) / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    models = _model_rows(stages, _recorded_models())
    intervals = _recorded_intervals()
    new_intervals, statuses = _new_intervals(output, stages, deadline_unix)
    intervals.extend(new_intervals)
    rows = _comparison_rows(models, intervals)
    _write_csv(output / 'models.csv', pd.DataFrame(models))
    _write_csv(output / 'summary.csv', pd.DataFrame(rows))
    _write_csv(output / 'bootstrap_summary.csv', pd.DataFrame(intervals))
    manifest = _report_manifest(output, statuses, deadline_unix, rows)
    _write_json(output / 'manifest.json', manifest)
    return manifest
