"""Проверки чередования trials и matched prefix без CatBoost и данных."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import optuna
from optuna.trial import TrialState, create_trial
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from optuna_config import SEARCH_SPACE, baseline_parameters, fixed_parameters, suggest_parameters
from optuna_selection import create_studies, choose_next, matched_selection
from optuna_selection import save_search_summary, trial_counts

optuna.logging.set_verbosity(optuna.logging.ERROR)


class OptunaSelectionTests(unittest.TestCase):
    def studies(self):
        return {(method, objective): optuna.create_study(direction='maximize')
                for objective in ('pointwise', 'ranker') for method in ('manual', 'dfs')}

    def add(self, studies, method, objective, value=None, state=TrialState.COMPLETE):
        params = baseline_parameters(objective) if state == TrialState.COMPLETE else {}
        distributions = {'depth': IntDistribution(4, 8),
                         'learning_rate': FloatDistribution(.03, .2, log=True),
                         'l2_leaf_reg': CategoricalDistribution(SEARCH_SPACE['l2_leaf_reg'])}
        studies[(method, objective)].add_trial(create_trial(
            state=state, value=value, params=params,
            distributions=distributions if params else {},
        ))

    def test_three_parameters_and_fixed_randomness(self):
        self.assertEqual(set(SEARCH_SPACE), {'depth', 'learning_rate', 'l2_leaf_reg'})
        self.assertEqual(SEARCH_SPACE['l2_leaf_reg'], [1e-20, .1, 1, 3, 10, 30])
        for objective in ('ranker', 'pointwise'):
            baseline = baseline_parameters(objective)
            self.assertEqual(set(baseline), set(SEARCH_SPACE))
            self.assertEqual(suggest_parameters(optuna.trial.FixedTrial(baseline)), baseline)
            self.assertEqual(fixed_parameters(objective)['random_strength'], 1.)
            self.assertEqual(fixed_parameters(objective)['rsm'], 1.)

    def test_failed_pruned_trials_do_not_enter_complete_prefix(self):
        studies = self.studies()
        self.add(studies, 'manual', 'ranker', state=TrialState.FAIL)
        self.add(studies, 'manual', 'ranker', .70)
        self.add(studies, 'manual', 'ranker', state=TrialState.PRUNED)
        self.add(studies, 'manual', 'ranker', .95)
        self.add(studies, 'dfs', 'ranker', .80)
        result = matched_selection(studies)['ranker']
        self.assertEqual(result['k'], 1)
        self.assertEqual(result['manual']['eligible_trial_numbers'], [1])
        self.assertEqual(result['manual']['trial_number'], 1)
        self.assertEqual(result['manual']['value'], .70)
        self.assertEqual(result['counts']['manual']['attempted'], 4)
        self.assertEqual(result['counts']['manual']['complete'], 2)
        self.assertEqual(result['counts']['manual']['fail'], 1)
        self.assertEqual(result['counts']['manual']['pruned'], 1)

    def test_last_unmatched_winner_is_excluded(self):
        studies = self.studies()
        for value in [.60, .70, .99]:
            self.add(studies, 'manual', 'pointwise', value)
        for value in [.65, .75]:
            self.add(studies, 'dfs', 'pointwise', value)
        result = matched_selection(studies)['pointwise']
        self.assertEqual(studies[('manual', 'pointwise')].best_trial.number, 2)
        self.assertEqual(result['k'], 2)
        self.assertEqual(result['manual']['trial_number'], 1)
        self.assertEqual(result['manual']['eligible_trial_numbers'], [0, 1])
        self.assertEqual(result['dfs']['trial_number'], 1)

    def test_zero_matched_trials_returns_no_candidate_on_either_side(self):
        studies = self.studies()
        self.add(studies, 'manual', 'ranker', .9)
        self.add(studies, 'dfs', 'ranker', state=TrialState.FAIL)
        result = matched_selection(studies)['ranker']
        self.assertEqual(result['k'], 0)
        for method in ('manual', 'dfs'):
            self.assertIsNone(result[method]['trial_number'])
            self.assertIsNone(result[method]['params'])
            self.assertIsNone(result[method]['value'])
            self.assertEqual(result[method]['eligible_trial_numbers'], [])

    def test_equal_values_prefer_first_eligible_trial(self):
        studies = self.studies()
        for method in ('manual', 'dfs'):
            self.add(studies, method, 'ranker', .8)
            self.add(studies, method, 'ranker', .8)
        result = matched_selection(studies)['ranker']
        self.assertEqual(result['manual']['trial_number'], 0)
        self.assertEqual(result['dfs']['trial_number'], 0)

    def test_scheduler_alternates_pairs_and_obeys_limits(self):
        studies = self.studies()
        sequence = []
        for _ in range(40):
            choice = choose_next(studies)
            self.assertIsNotNone(choice)
            sequence.append(choice)
            self.add(studies, *choice, value=.7)
        expected = [('manual', 'pointwise'), ('dfs', 'pointwise'),
                    ('manual', 'ranker'), ('dfs', 'ranker')]
        self.assertEqual(sequence[:8], expected * 2)
        self.assertIsNone(choose_next(studies))
        for objective, limit in [('ranker', 8), ('pointwise', 12)]:
            for method in ('manual', 'dfs'):
                self.assertEqual(trial_counts(studies[(method, objective)])['attempted'], limit)

    def test_scheduler_counts_fail_pruned_running_but_not_waiting(self):
        studies = self.studies()
        self.add(studies, 'manual', 'pointwise', state=TrialState.FAIL)
        self.add(studies, 'manual', 'pointwise', state=TrialState.PRUNED)
        self.add(studies, 'manual', 'pointwise', state=TrialState.RUNNING)
        self.assertEqual(choose_next(studies), ('dfs', 'pointwise'))
        fresh = self.studies()
        self.add(fresh, 'manual', 'pointwise', state=TrialState.WAITING)
        self.assertEqual(choose_next(fresh), ('manual', 'pointwise'))
        for objective in ('ranker', 'pointwise'):
            for method in ('manual', 'dfs'):
                self.add(fresh, method, objective, state=TrialState.FAIL)
        self.assertIsNone(choose_next(fresh, {'ranker': 1, 'pointwise': 1}))

    def test_sqlite_restart_enqueues_baseline_once_and_fails_stale_running(self):
        with tempfile.TemporaryDirectory(prefix='optuna-selection-') as directory:
            paths = {'results': Path(directory)}
            first = create_studies(paths)
            for (method, objective), study in first.items():
                self.assertEqual(len(study.trials), 1)
                self.assertEqual(study.trials[0].system_attrs['fixed_params'], baseline_parameters(objective))
            running = first[('manual', 'pointwise')].ask()
            suggest_parameters(running)
            second = create_studies(paths)
            self.assertEqual(second[('manual', 'pointwise')].trials[0].state, TrialState.FAIL)
            for study in second.values():
                self.assertEqual(len(study.trials), 1)

    def test_summary_saves_matched_selection_counts_and_full_history(self):
        studies = self.studies()
        for value in [.7, .99]:
            self.add(studies, 'manual', 'pointwise', value)
        self.add(studies, 'dfs', 'pointwise', .8)
        with tempfile.TemporaryDirectory(prefix='optuna-summary-') as directory:
            paths = {'results': Path(directory)}
            result = save_search_summary(paths, studies)
            for filename in ('matched_selection.json', 'search_summary.json'):
                self.assertEqual(json.loads((Path(directory) / filename).read_text()), result)
            self.assertEqual(result['pointwise']['manual']['trial_number'], 0)
            counts = json.loads((Path(directory) / 'study_counts.json').read_text())
            self.assertEqual(counts['manual_pointwise']['complete'], 2)
            for method, objective in studies:
                path = Path(directory) / f'{method}_{objective}_trials.csv'
                self.assertIn('number', path.read_text().splitlines()[0])


if __name__ == '__main__':
    unittest.main()
