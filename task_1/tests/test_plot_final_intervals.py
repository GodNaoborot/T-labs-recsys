"""Проверить, что рисунок строится из интервалов, а не из новых предсказаний."""
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from plot_final_intervals import PAIRS, plot_intervals


class FinalIntervalPlotTests(unittest.TestCase):
    def test_each_pair_and_block_is_plotted(self):
        table = pd.DataFrame([
            {'candidate': candidate, 'reference': reference, 'block_days': block,
             'delta_ndcg': 0.001, 'ci_low': -0.001, 'ci_high': 0.003}
            for candidate, reference, _ in PAIRS for block in (3, 4)
        ])
        figure = plot_intervals(table, table)
        try:
            self.assertEqual(len(figure.axes), 6)
            for axis in figure.axes:
                self.assertEqual(len(axis.containers), 2)
                for container in axis.containers:
                    self.assertAlmostEqual(container.lines[0].get_xdata()[0], 0.001)
        finally:
            plt.close(figure)

    def test_missing_pair_is_not_silently_omitted(self):
        table = pd.DataFrame(columns=['candidate', 'reference', 'block_days',
                                      'delta_ndcg', 'ci_low', 'ci_high'])
        try:
            with self.assertRaisesRegex(ValueError, 'нужны блоки'):
                plot_intervals(table)
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
