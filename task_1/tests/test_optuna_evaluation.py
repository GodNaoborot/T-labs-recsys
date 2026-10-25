"""Небольшие проверки оценки без обучения и без настоящих датасетов."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from optuna_evaluation import evaluate_and_save
from train_dfs_catboost import ranking_metrics
from bootstrap_intervals import compare_saved_intervals, write_analysis


class FakePool:
    def __init__(self, size, ids=None, slices=None):
        self.ids = np.arange(size) if ids is None else ids
        self.slices = [] if slices is None else slices

    def num_row(self):
        return len(self.ids)

    def slice(self, ids):
        selected = self.ids[ids]
        self.slices.append(selected.copy())
        return FakePool(len(selected), ids=selected, slices=self.slices)


class FakeRanker:
    loaded_offset = 0.
    tree_count_ = 4

    def __init__(self, scores=None, probabilities=None):
        self.scores = np.array([] if scores is None else scores, dtype=float)
        self.probabilities = np.array([] if probabilities is None else probabilities, dtype=float)
        self.classes_ = np.array([0, 1, 2])
        self.loaded = False

    def predict(self, pool, thread_count=None):
        return self.scores[pool.ids] + (self.loaded_offset if self.loaded else 0.)

    def predict_proba(self, pool, thread_count=None):
        return self.probabilities[pool.ids]

    def save_model(self, path):
        Path(path).write_text(json.dumps({'scores': self.scores.tolist(),
                                         'probabilities': self.probabilities.tolist()}))

    def load_model(self, path):
        saved = json.loads(Path(path).read_text())
        self.scores = np.asarray(saved['scores'])
        self.probabilities = np.asarray(saved['probabilities'])
        self.loaded = True
        return self

    def get_all_params(self):
        return {'depth': 4, 'random_seed': 42}

    def get_best_iteration(self):
        return 3

    def get_evals_result(self):
        return {'validation': {'loss': [1., .5]}}

    def get_feature_importance(self, **kwargs):
        return pd.DataFrame({'Feature Id': ['a'], 'Importances': [100.]})


class FakeClassifier(FakeRanker):
    pass


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='optuna-evaluation-test-')
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / 'evaluation'
        self.targets = {'label': np.array([0, 2, 0, 1, 2, 2, 0]),
                        'group_id': np.array([10, 10, 20, 20, 20, 30, 30]),
                        'log_id': np.arange(100, 107), 'day': np.array([4, 4, 4, 4, 4, 6, 6])}
        self.pool = FakePool(7)
        self.scores = np.array([0., 0., .2, .5, .5, .1, .9])

    def evaluate(self, model=None, kind='ranker', targets=None):
        return evaluate_and_save(FakeRanker(self.scores) if model is None else model, kind,
                                 self.pool, self.targets if targets is None else targets, self.output)

    def test_ranker_ties_and_group_weighting(self):
        record = self.evaluate()
        self.assertEqual(record['metrics'], ranking_metrics(self.targets['label'], self.targets['group_id'], self.scores))
        np.testing.assert_array_equal(np.load(self.output / 'val_score.npy'), self.scores)
        groups = pd.read_parquet(self.output / 'group_ndcg.parquet')
        daily = pd.read_csv(self.output / 'daily_ndcg.csv')
        np.testing.assert_array_equal(groups.group_size, [2, 3, 2])
        np.testing.assert_array_equal(daily.day, [4, 5, 6])
        np.testing.assert_array_equal(daily.group_count, [2, 0, 1])
        self.assertAlmostEqual(daily.ndcg_sum.sum() / daily.group_count.sum(), record['metrics']['NDCG@10'])
        self.assertNotAlmostEqual(daily.ndcg_mean.mean(), record['metrics']['NDCG@10'])
        self.assertTrue(record['reload_predictions_exact'])

    def test_classifier_uses_expected_label_and_logloss(self):
        probabilities = np.array([[.6, .3, .1], [.1, .1, .8], [.8, .1, .1], [.2, .6, .2],
                                  [.1, .2, .7], [.2, .2, .6], [.7, .2, .1]])
        record = self.evaluate(FakeClassifier(probabilities=probabilities), 'pointwise')
        expected = probabilities @ np.array([0., 1., 2.])
        np.testing.assert_array_equal(np.load(self.output / 'val_score.npy'), expected)
        np.testing.assert_array_equal(np.load(self.output / 'val_proba.npy'), probabilities)
        self.assertAlmostEqual(record['metrics']['logloss'], log_loss(self.targets['label'], probabilities, labels=[0, 1, 2]))
        self.assertEqual(record['metrics']['NDCG@10'], ranking_metrics(
            self.targets['label'], self.targets['group_id'], expected)['NDCG@10'])

    def test_completed_artifacts_are_not_overwritten(self):
        self.evaluate()
        path = self.output / 'evaluation.json'
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.evaluate()
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_pairing_metadata_is_rejected(self):
        bad_cases = [('group_id', [10, 20, 10, 20, 20, 30, 30]),
                     ('log_id', [1, 1, 2, 3, 4, 5, 6]), ('day', [4, 5, 4, 4, 4, 6, 6]),
                     ('label', [0, 0, 0, 1, 2, 2, 0])]
        for key, values in bad_cases:
            with self.subTest(key=key):
                targets = {**self.targets, key: np.array(values)}
                with self.assertRaises(ValueError):
                    self.evaluate(targets=targets)
        self.assertFalse(self.output.exists())

    def test_nonfinite_score_and_wrong_kind_are_rejected(self):
        self.scores[1] = np.nan
        with self.assertRaises(ValueError):
            self.evaluate()
        with self.assertRaises(ValueError):
            self.evaluate(kind='other')

    def test_reload_difference_is_not_marked_completed(self):
        FakeRanker.loaded_offset = 1.
        self.addCleanup(setattr, FakeRanker, 'loaded_offset', 0.)
        with self.assertRaises(AssertionError):
            self.evaluate()
        self.assertFalse((self.output / 'evaluation.json').exists())

    def test_reload_probe_preserves_entire_large_group(self):
        size = 2500
        targets = {'label': np.arange(size) % 3, 'group_id': np.r_[np.full(2100, 11), np.full(400, 22)],
                   'log_id': np.arange(size), 'day': np.r_[np.full(2100, 4), np.full(400, 6)]}
        pool = FakePool(size)
        evaluate_and_save(FakeRanker(np.arange(size) % 7), 'ranker', pool, targets, self.output)
        self.assertEqual(len(pool.slices[-1]), 2100)
        self.assertEqual(pool.slices[-1][-1], 2099)


class SavedIntervalsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='saved-intervals-test-')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'results/experiment-1/dfs/bootstrap_noninferiority'
        self.source.mkdir(parents=True)
        values = np.linspace(.01, .02, 21)
        np.savez(self.source / 'bootstrap_losses.npz', ranker_L3=values)
        pd.DataFrame([{'objective': 'ranker', 'block_days': 3, 'relative_loss': .015,
                       'upper_bound': np.quantile(values, .95), 'n_resamples': len(values)}]).to_csv(
                           self.source / 'summary.csv', index=False)

    def test_quantiles_are_separate_from_ni_and_sources_unchanged(self):
        before = (self.source / 'bootstrap_losses.npz').read_bytes()
        frame = compare_saved_intervals(self.root)
        self.assertAlmostEqual(frame.ci_lower.iloc[0], .01025)
        self.assertAlmostEqual(frame.ci_upper.iloc[0], .01975)
        self.assertAlmostEqual(frame.upper_bound.iloc[0], .0195)
        self.assertNotEqual(frame.ci_upper.iloc[0], frame.upper_bound.iloc[0])
        self.assertEqual((self.source / 'bootstrap_losses.npz').read_bytes(), before)
        path = write_analysis(frame, self.root / 'analysis')
        self.assertTrue(path.is_file())
        with self.assertRaises(FileExistsError):
            write_analysis(frame, self.root / 'analysis')

    def test_rolling_uses_global_summary(self):
        folder = self.root / 'results/experiment-1/rolling_windows/train_121_165'
        folder.mkdir(parents=True)
        (folder / 'bootstrap_losses.npz').write_bytes((self.source / 'bootstrap_losses.npz').read_bytes())
        summary = pd.read_csv(self.source / 'summary.csv')
        summary['window'] = 'train_121_165'
        summary.to_csv(folder.parent / 'summary.csv', index=False)
        frame = compare_saved_intervals(self.root)
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame.period), {'original_val', 'train_121_165'})

    def test_original_upper_bound_mismatch_is_rejected(self):
        frame = pd.read_csv(self.source / 'summary.csv')
        frame['upper_bound'] = 0.
        frame.to_csv(self.source / 'summary.csv', index=False)
        with self.assertRaises(AssertionError):
            compare_saved_intervals(self.root)


if __name__ == '__main__':
    unittest.main()
