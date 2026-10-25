"""Двусторонние интервалы из сохраненных bootstrap-реплик без изменения экспериментов."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'results/analysis/20261023'


def _saved_periods(root):
    """Найти исходный val и завершенные rolling-окна по сохраненным файлам."""
    results = Path(root) / 'results/experiment-1'
    yield 'original_val', results / 'dfs/bootstrap_noninferiority'
    for folder in sorted((results / 'rolling_windows').glob('train_*')):
        if (folder / 'bootstrap_losses.npz').is_file():
            yield folder.name, folder


def _period_summary(period, folder):
    """Прочитать локальную сводку val или строки окна из общей rolling-сводки."""
    if period == 'original_val':
        return pd.read_csv(folder / 'summary.csv')
    summary = pd.read_csv(folder.parent / 'summary.csv')
    rows = summary.loc[summary['window'] == period]
    if rows.empty:
        raise ValueError(f'В общей сводке нет окна {period}')
    return rows


def _interval_rows(period, folder):
    """Добавить q2.5/q97.5 к прежней точечной оценке и проверить прежнюю q95."""
    summary = _period_summary(period, folder)
    with np.load(folder / 'bootstrap_losses.npz') as saved:
        for row in summary.to_dict('records'):
            key = f"{row['objective']}_L{row['block_days']}"
            losses = saved[key]
            if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
                raise ValueError(f'{period}/{key}: нужны конечные одномерные реплики')
            low, high, upper = np.quantile(losses, [.025, .975, .95], method='linear')
            np.testing.assert_allclose(upper, row['upper_bound'], rtol=0, atol=1e-14)
            if int(row['n_resamples']) != len(losses):
                raise ValueError(f'{period}/{key}: число реплик не совпало с summary')
            yield {'period': period, 'objective': row['objective'], 'block_days': row['block_days'],
                   'relative_loss': row['relative_loss'], 'ci_lower': float(low), 'ci_upper': float(high),
                   'upper_bound': float(upper), 'n_resamples': len(losses),
                   'source': str(folder / 'bootstrap_losses.npz')}


def compare_saved_intervals(root=ROOT):
    """Прочитать готовые реплики и вернуть интервалы; исходные файлы не изменяются."""
    return pd.DataFrame([row for period, folder in _saved_periods(root)
                         for row in _interval_rows(period, folder)])


def write_analysis(frame, output_dir):
    """Записать отдельный аналитический отчет, отказавшись от перезаписи файлов."""
    output_dir = Path(output_dir)
    csv_path, config_path = output_dir / 'bootstrap_intervals.csv', output_dir / 'bootstrap_intervals.json'
    if csv_path.exists() or config_path.exists():
        raise FileExistsError('Аналитический отчет уже существует; выберите другой --output')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    config = {'rows': len(frame), 'interval': 'percentile q.025/q.975',
              'upper_bound': 'one-sided q.95; original NI results unchanged',
              'scope': 'exploratory; conditional on fixed fitted models; periods are dependent',
              'test_used': False, 'training_started': False}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    return csv_path


def main(argv=None):
    """Создать новый отчет по завершенным val-экспериментам без обучения и test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(write_analysis(compare_saved_intervals(), args.output))


if __name__ == '__main__':
    main()
