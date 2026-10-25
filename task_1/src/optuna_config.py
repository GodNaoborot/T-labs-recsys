"""Общие настройки ограниченного поиска параметров CatBoost."""
from pathlib import Path
import hashlib
import json
import re
from importlib.metadata import version

from project_artifacts import TASK_ROOT

SEED = 42
NDCG = 'NDCG:top=10;type=Base;denominator=LogPosition'
JOBS = [(method, objective) for objective in ('ranker', 'pointwise')
        for method in ('manual', 'dfs')]
SEARCH_SPACE = {'depth': [4, 8], 'learning_rate': [0.03, 0.2],
                'l2_leaf_reg': [1e-20, 0.1, 1., 3., 10., 30.]}


def runtime_versions():
    """Сохранить версии библиотек, от которых зависит поиск и обучение."""
    return {name: version(name) for name in ('optuna', 'catboost', 'numpy', 'pyarrow', 'pandas')}


def run_paths(name):
    """Ограничить результаты отдельной папкой с безопасным именем."""
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) is None:
        raise ValueError('Имя запуска: 1–64 буквы/цифры/_/-, первый символ — буква или цифра')
    return {'data': TASK_ROOT / 'data/tuning' / name,
            'results': TASK_ROOT / 'results/tuning' / name}


def write_json(path, value):
    """Атомарно сохранить состояние, доступное во время долгого расчета."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    """Прочитать небольшую сохраненную конфигурацию."""
    return json.loads(Path(path).read_text())


def source_hashes():
    """Связать продолжение поиска с неизменным кодом и исходными моделями."""
    names = ['optuna_config.py', 'optuna_data.py', 'optuna_worker.py', 'run_optuna.py',
             'optuna_evaluation.py', 'optuna_selection.py', 'optuna_report.py',
             'train_dfs_catboost.py', 'block_bootstrap.py', 'bootstrap_intervals.py',
             'project_artifacts.py']
    paths = [TASK_ROOT / 'src' / name for name in names]
    paths += [TASK_ROOT / 'results/current_models.json']
    return {str(path.relative_to(TASK_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def fixed_parameters(objective):
    """Задать общий бюджет деревьев, сохранив прежние лоссы и bootstrap."""
    common = dict(iterations=1600, early_stopping_rounds=80, random_seed=SEED,
                  use_best_model=True, verbose=100, allow_writing_files=False,
                  nan_mode='Min', border_count=254, task_type='CPU',
                  random_strength=1., rsm=1.)
    if objective == 'ranker':
        return {**common, 'loss_function': 'YetiRank', 'eval_metric': NDCG,
                'thread_count': 4, 'bootstrap_type': 'MVS', 'subsample': 0.8}
    if objective == 'pointwise':
        return {**common, 'loss_function': 'MultiClass', 'eval_metric': 'MultiClass',
                'thread_count': 8, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.}
    raise ValueError(objective)


def baseline_parameters(objective):
    """Поставить исходную конфигурацию первой, включая почти нулевой L2 ранкера."""
    return {'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 1e-20 if objective == 'ranker' else 3.}


def suggest_parameters(trial):
    """Искать три согласованных параметра; остальные настройки зафиксированы."""
    return {'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', SEARCH_SPACE['l2_leaf_reg'])}
