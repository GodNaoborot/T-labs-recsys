"""Чередование trials и сравнение одинакового числа завершенных проб manual/DFS."""
import math

import optuna
from optuna.trial import TrialState

from optuna_config import JOBS, SEED, baseline_parameters, write_json

TRIAL_LIMITS = {'ranker': 8, 'pointwise': 12}
OBJECTIVE_ORDER = ('pointwise', 'ranker')
METHODS = ('manual', 'dfs')
ATTEMPTED_STATES = {TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED, TrialState.RUNNING}


def trial_counts(study):
    """Посчитать попытки, включая ошибки и прерывания; WAITING еще не является попыткой."""
    trials = study.get_trials(deepcopy=False)
    counts = {state.name.lower(): sum(t.state == state for t in trials)
              for state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED,
                            TrialState.RUNNING, TrialState.WAITING)}
    counts['attempted'] = sum(t.state in ATTEMPTED_STATES for t in trials)
    return counts


def create_studies(paths):
    """Продолжить SQLite studies, завершить старые RUNNING и один раз поставить baseline."""
    results = paths['results']
    results.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{results / 'optuna.sqlite3'}"
    studies = {}
    for method, objective in JOBS:
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3)
        study = optuna.create_study(storage=storage, study_name=f'{method}_{objective}',
                                   direction='maximize', sampler=sampler, load_if_exists=True)
        for trial in study.get_trials(deepcopy=False):
            if trial.state == TrialState.RUNNING:
                study.tell(trial.number, state=TrialState.FAIL)
        if not study.trials:
            study.enqueue_trial(baseline_parameters(objective))
        studies[(method, objective)] = study
    return studies


def choose_next(studies, trial_limits=None):
    """Чередовать objective парами M/D, начиная с MultiClass; время контролирует runner."""
    limits = TRIAL_LIMITS if trial_limits is None else trial_limits
    candidates = []
    for priority, objective in enumerate(OBJECTIVE_ORDER):
        counts = {method: trial_counts(studies[(method, objective)])['attempted'] for method in METHODS}
        available = [method for method in METHODS if counts[method] < limits[objective]]
        if not available:
            continue
        method = min(available, key=lambda value: (counts[value], METHODS.index(value)))
        candidates.append((min(counts.values()), priority, method, objective))
    if not candidates:
        return None
    _, _, method, objective = min(candidates)
    return method, objective


def _complete_trials(study):
    """Получить завершенные пробы по номеру, сохраняя пропуски FAIL/PRUNED."""
    trials = sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)),
                    key=lambda trial: trial.number)
    if any(trial.value is None or not math.isfinite(trial.value) for trial in trials):
        raise ValueError('COMPLETE trial должен иметь конечное значение метрики')
    return trials


def _prefix_best(complete, count):
    """Выбрать лучшую только среди первых count COMPLETE, при равенстве — самую раннюю."""
    eligible = complete[:count]
    numbers = [trial.number for trial in eligible]
    if not eligible:
        return {'trial_number': None, 'params': None, 'value': None,
                'eligible_trial_numbers': numbers}
    best = max(eligible, key=lambda trial: trial.value)
    return {'trial_number': best.number, 'params': dict(best.params), 'value': best.value,
            'eligible_trial_numbers': numbers}


def matched_selection(studies):
    """Сравнить M/D по одинаковым префиксам COMPLETE, не используя unmatched-победителя."""
    result = {}
    for objective in OBJECTIVE_ORDER:
        complete = {method: _complete_trials(studies[(method, objective)]) for method in METHODS}
        count = min(len(trials) for trials in complete.values())
        result[objective] = {'k': count,
                             **{method: _prefix_best(complete[method], count) for method in METHODS},
                             'counts': {method: trial_counts(studies[(method, objective)])
                                        for method in METHODS}}
    return result


def save_search_summary(paths, studies):
    """Сохранить matched selection, счетчики и полную историю без обращения к study.best."""
    selection = matched_selection(studies)
    results = paths['results']
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / 'matched_selection.json', selection)
    write_json(results / 'search_summary.json', selection)
    counts = {}
    for (method, objective), study in studies.items():
        counts[f'{method}_{objective}'] = trial_counts(study)
        frame = study.trials_dataframe()
        if not len(frame.columns):
            frame = frame.reindex(columns=['number', 'value', 'state'])
        frame.to_csv(results / f'{method}_{objective}_trials.csv', index=False)
    write_json(results / 'study_counts.json', counts)
    return selection
