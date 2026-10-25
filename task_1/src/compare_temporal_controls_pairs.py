"""Прямые парные интервалы по сохранённым validation NDCG, без чтения test."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from block_bootstrap import aggregate_days, circular_day_indices

TASK_ROOT = Path(__file__).resolve().parents[1]
RUN = TASK_ROOT / 'results/experiment-1/temporal_controls/dfs462_controls_20261023_v1'
PAIRS = [('C2b', 'D462-sub'), ('C2b', 'C2a')]
BLOCKS = (3, 4)
REPLICATES = 20_000
SEED = 42


def file_hash(path):
    """Посчитать SHA256 небольшого отчёта или исходника."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_groups(run, variant):
    """Прочитать только validation-группы и сверить их с готовой оценкой."""
    folder = run / 'models' / variant
    groups = pd.read_parquet(folder / 'group_ndcg.parquet')
    required = ['group_id', 'day', 'group_size', 'ndcg']
    if list(groups.columns) != required:
        raise ValueError('Изменилась схема сохранённых групп')
    if not groups['group_id'].is_unique or not groups['group_id'].is_monotonic_increasing:
        raise ValueError('Нужны уникальные группы в canonical-порядке')
    if (groups['group_size'] < 2).any():
        raise ValueError('Неожиданная одноэлементная группа')
    values = groups['ndcg'].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1 + 1e-12)).any():
        raise ValueError('Некорректный групповой NDCG')
    evaluation = json.loads((folder / 'evaluation.json').read_text())
    if len(groups) != evaluation['groups'] or int(groups['group_size'].sum()) != evaluation['rows']:
        raise ValueError('Размеры групп не совпали с evaluation.json')
    np.testing.assert_allclose(values.mean(), evaluation['metrics']['NDCG@10'], rtol=0, atol=1e-11)
    return groups


def pair_daily(candidate, reference):
    """Сначала проверить одинаковые группы, затем собрать дневные суммы."""
    keys = ['group_id', 'day', 'group_size']
    pd.testing.assert_frame_equal(candidate[keys], reference[keys])
    return aggregate_days(reference['day'].to_numpy(),
                          reference['ndcg'].to_numpy(), candidate['ndcg'].to_numpy())


def absolute_delta_draws(daily, indices):
    """Сохранить исходный вес групп при общем для обеих моделей выборе дней."""
    counts = daily['group_count'].to_numpy()[indices].sum(axis=1)
    if (counts <= 0).any():
        raise ValueError('Реплика без групп')
    candidate = daily['dfs_sum'].to_numpy()[indices].sum(axis=1)
    reference = daily['base_sum'].to_numpy()[indices].sum(axis=1)
    return (candidate - reference) / counts


def calculate(run, output):
    """Пересчитать две validation-пары; исторические отчёты не перезаписывать."""
    for name in ['status.json', 'result.json']:
        if not json.loads((run / name).read_text())['completed']:
            raise ValueError('Основной запуск не завершён')
    if output.exists():
        raise FileExistsError('Для нового расчёта нужна новая папка вывода')
    names = sorted({name for pair in PAIRS for name in pair})
    use_groups = all((run / 'models' / name / 'group_ndcg.parquet').exists() for name in names)
    groups = {name: read_groups(run, name) for name in names} if use_groups else {}
    daily_sources = {}
    if not use_groups:
        daily_sources = {name: pd.read_csv(run / f'daily_{name}.csv') for name in names}
        for name, table in daily_sources.items():
            required = ['day', 'group_count', 'base_sum', 'dfs_sum']
            if any(column not in table for column in required):
                raise ValueError('Неполная дневная таблица: ' + name)
            counts = table['group_count'].to_numpy()
            values = table['dfs_sum'].to_numpy()
            if (counts < 0).any() or not np.isfinite(values).all():
                raise ValueError('Некорректные дневные суммы: ' + name)
            if ((values < 0) | (values > counts + 1e-9)).any():
                raise ValueError('Дневной NDCG вне границ: ' + name)
    indices = {block: circular_day_indices(15, block, REPLICATES, seed=SEED)
               for block in BLOCKS}
    rows, daily_tables = [], {}
    for candidate, reference in PAIRS:
        if use_groups:
            daily = pair_daily(groups[candidate], groups[reference])
        else:
            candidate_daily, reference_daily = daily_sources[candidate], daily_sources[reference]
            pd.testing.assert_frame_equal(candidate_daily[['day', 'group_count']],
                                          reference_daily[['day', 'group_count']])
            daily = reference_daily[['day', 'group_count']].copy()
            daily['base_sum'] = reference_daily['dfs_sum'].to_numpy()
            daily['dfs_sum'] = candidate_daily['dfs_sum'].to_numpy()
        if daily['day'].tolist() != list(range(270, 285)):
            raise ValueError('Ожидались validation-дни 270–284')
        point = float((daily['dfs_sum'].sum() - daily['base_sum'].sum()) / daily['group_count'].sum())
        daily_tables[candidate + '_minus_' + reference] = daily
        for block in BLOCKS:
            draws = absolute_delta_draws(daily, indices[block])
            low, high = np.quantile(draws, [.025, .975], method='linear')
            rows.append({'candidate': candidate, 'reference': reference,
                         'block_days': block, 'replicates': REPLICATES,
                         'seed': SEED, 'delta_ndcg': point,
                         'ci_low': float(low), 'ci_high': float(high),
                         'confidence': .95, 'two_sided': True,
                         'validation_reused': True})
    source_files = ([run / 'models' / name / 'group_ndcg.parquet' for name in names]
                    if use_groups else [run / f'daily_{name}.csv' for name in names])
    sources = {str(path): file_hash(path) for path in source_files}
    sources[str(run / 'comparison.csv')] = file_hash(run / 'comparison.csv')
    for path in [Path(__file__), Path(__file__).with_name('block_bootstrap.py')]:
        sources[str(path)] = file_hash(path)
    selection_path = run / 'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    source_rows = (int(groups[names[0]]['group_size'].sum()) if use_groups
                   else selection.get('val', {}).get('rows'))
    report = {'completed': True, 'split': 'val', 'test_opened': False, 'rows': source_rows,
              'run': str(run), 'groups': int(daily['group_count'].sum()),
              'input_level': 'groups' if use_groups else 'daily_sums',
              'calendar': [270, 284], 'pairs': rows, 'source_sha256': sources,
              'method': 'paired circular day-block percentile bootstrap; group-weighted',
              'limitations': ['validation reused', 'fixed fitted models',
                              'no multiplicity correction', 'not an equivalence test']}
    output.mkdir()
    pd.DataFrame(rows).to_csv(output / 'bootstrap_summary.csv', index=False)
    for name, daily in daily_tables.items():
        daily.to_csv(output / ('daily_' + name + '.csv'), index=False)
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'compare_temporal_controls_pairs.py').write_text(Path(__file__).read_text())
    lines = ['# Прямые интервалы Temporal DFS462', '',
             f"Полная validation: {int(daily['group_count'].sum())} групп, дни 271–285 (с единицы).",
             'Разность абсолютная: NDCG кандидата минус NDCG сравнения.', '',
             '| Пара | Δ NDCG | Блок, дней | Двусторонний 95% интервал |',
             '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['candidate']} − {row['reference']} | {row['delta_ndcg']:+.6f} | "
                     f"{row['block_days']} | [{row['ci_low']:+.6f}; {row['ci_high']:+.6f}] |")
    lines += ['', 'Парный циклический bootstrap по дням: 20 000 реплик, seed 42, '
              'блоки 3 и 4 дня. Сохранён исходный вес групп.',
              'Интервалы относятся к фиксированным моделям и повторно использованной validation.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default=RUN.name)
    parser.add_argument('--output-name', default='direct_pairs_20261023_v1')
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ['.', '..']:
        raise ValueError('Нужно имя подпапки без разделителей пути')
    if Path(args.run_name).name != args.run_name or args.run_name in ['.', '..']:
        raise ValueError('Нужно имя запуска без компонентов пути')
    run = RUN.parent / args.run_name
    result = calculate(run, run / args.output_name)
    print(json.dumps({'completed': result['completed'], 'pairs': result['pairs'],
                      'test_opened': False}, indent=2))


if __name__ == '__main__':
    main()
