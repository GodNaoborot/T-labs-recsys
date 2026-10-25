"""Повторить исходный парный bootstrap по новым моделям или дневным суммам."""
from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd

from block_bootstrap import (aggregate_days, circular_day_indices,
                             relative_loss_draws, summarize_bootstrap)
from bootstrap_validation import prepare_validation_groups

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_daily(daily, seed=42, n_resamples=20000):
    """Рассчитать обе длины блока отдельно для каждого objective."""
    rows, draws = [], {}
    for objective in ('ranker', 'pointwise'):
        part = daily.loc[daily['objective'] == objective].reset_index(drop=True)
        for block_days in (3, 4):
            indices = circular_day_indices(len(part), block_days, n_resamples, seed)
            losses = relative_loss_draws(part, indices)
            rows.append({'objective': objective, 'block_days': block_days,
                         **summarize_bootstrap(part, losses, margin=.03, alpha_sig=.05)})
            draws[f'{objective}_L{block_days}'] = losses
    return pd.DataFrame(rows), draws


def groups_to_daily(groups):
    """Сохранить вес каждой user-day группы в суммах соответствующего дня."""
    pieces = []
    for objective in ('ranker', 'pointwise'):
        daily = aggregate_days(groups['day'], groups[f'base_{objective}'],
                               groups[f'dfs_{objective}'])
        daily['objective'] = objective
        daily['base_ndcg'] = daily['base_sum'] / daily['group_count']
        daily['dfs_ndcg'] = daily['dfs_sum'] / daily['group_count']
        pieces.append(daily)
    return pd.concat(pieces, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--daily', type=Path, help='Пересчитать только статистику готовых прогнозов')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if (output / 'summary.csv').exists():
        raise FileExistsError('Результат уже существует; нужен новый --output')
    output.mkdir(parents=True, exist_ok=True)
    if args.daily:
        daily = pd.read_csv(args.daily)
    else:
        if not args.run_name:
            parser.error('Нужен --run-name либо --daily')
        os.environ['TLAB_EXPERIMENT_RUN'] = args.run_name
        daily = groups_to_daily(prepare_validation_groups(ROOT, output, args.seed))
    summary, draws = bootstrap_daily(daily, args.seed)
    daily.to_csv(output / 'daily_ndcg.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    np.savez_compressed(output / 'bootstrap_losses.npz', **draws)
    (output / 'run_config.json').write_text(json.dumps({
        'completed': True, 'run_name': args.run_name, 'seed': args.seed,
        'n_resamples': 20000, 'block_days': [3, 4], 'margin': .03,
        'test_used': False, 'daily_source': str(args.daily) if args.daily else None,
    }, ensure_ascii=False, indent=2) + '\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
