"""Построить интервалы итогового сравнения из опубликованных CSV."""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAIRS = (
    ('D462-t', 'M-t', 'DFS − исходные ручные'),
    ('C2b-t', 'M-t', 'Расширенные − исходные ручные'),
    ('C2b-t', 'D462-t', 'Расширенные ручные − DFS'),
)


def plot_intervals(validation, test=None, pairs=None):
    """У каждой пары свой масштаб, чтобы была видна и небольшая разница."""
    pairs = PAIRS if pairs is None else tuple(pairs)
    splits = [('Validation', validation)]
    if test is not None:
        splits.append(('Повторный test', test))
    height = 7.3 if len(pairs) == 3 else 2.3 * len(pairs) + 0.4
    figure, axes = plt.subplots(len(pairs), len(splits), figsize=(6 * len(splits), height), squeeze=False)
    for column, (split, table) in enumerate(splits):
        required = {'candidate', 'reference', 'block_days', 'delta_ndcg', 'ci_low', 'ci_high'}
        if not required.issubset(table.columns):
            raise ValueError(f'{split}: не хватает столбцов интервалов')
        for row, (candidate, reference, title) in enumerate(pairs):
            selected = table[(table.candidate == candidate) & (table.reference == reference)]
            if len(selected) != 2 or set(selected.block_days) != {3, 4}:
                raise ValueError(f'{split}: нужны блоки 3 и 4 дня для {candidate} − {reference}')
            selected = selected.sort_values('block_days')
            axis = axes[row, column]
            for position, record in enumerate(selected.itertuples(index=False)):
                delta = record.delta_ndcg
                if not record.ci_low <= delta <= record.ci_high:
                    raise ValueError(f'{split}: точечная разница вне интервала')
                axis.errorbar(delta, position,
                              xerr=[[delta - record.ci_low], [record.ci_high - delta]],
                              fmt='o', capsize=5, color=('#4e79a7', '#f28e2b')[position])
            if abs(selected.delta_ndcg.iloc[0]) < 0.002:
                axis.axvline(0, color='gray', linestyle='--', linewidth=1)
            axis.set_yticks([0, 1], ['Блок 3 дня', 'Блок 4 дня'])
            axis.set_ylim(-0.5, 1.5)
            axis.set_title(f'{split}: {title}', fontsize=10)
            axis.set_xlabel('Абсолютная разница NDCG@10')
            axis.ticklabel_format(axis='x', style='plain', useOffset=False)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.grid(axis='x', alpha=0.2)
            axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Парные 95% интервалы: условно на обученных моделях', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    reports = ROOT / 'docs/results/c4'
    parser.add_argument('--validation', type=Path, default=reports / 'validation_intervals.csv')
    parser.add_argument('--test', type=Path, default=reports / 'test_intervals.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/figures/c4_intervals.png')
    args = parser.parse_args(argv)
    validation = pd.read_csv(args.validation, float_precision='round_trip')
    test = pd.read_csv(args.test, float_precision='round_trip')
    figure = plot_intervals(validation, test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == '__main__':
    main()
