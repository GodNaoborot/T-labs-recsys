"""Один шаг подготовки или обучения в отдельном процессе с ограниченной памятью."""
from pathlib import Path
import argparse
import math
import os
import resource
import signal
import threading
import time

from optuna_config import run_paths, read_json, write_json
from project_artifacts import file_hash


def arm_worker_deadline():
    """Остановить даже осиротевший worker, включая время импорта тяжелых библиотек."""
    values = [float(os.environ[name]) for name in
              ('TLAB_QUEUE_DEADLINE_UNIX', 'TLAB_SEARCH_DEADLINE_UNIX') if os.environ.get(name)]
    if not values:
        return
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Некорректный абсолютный deadline worker')
    deadline = min(values)
    remaining = deadline - time.time()
    if remaining <= 0:
        raise SystemExit(124)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    threading.Thread(target=watch_wall_clock, args=(deadline,), daemon=True).start()


def watch_wall_clock(deadline):
    """После сна компьютера проверить абсолютное время независимо от относительного таймера."""
    while time.time() < deadline:
        time.sleep(min(.25, max(0., deadline - time.time())))
    os.kill(os.getpid(), signal.SIGKILL)


def pool_info(*args):
    """Импортировать подготовку данных только после установки ограничения времени."""
    from optuna_data import pool_info as get_info
    return get_info(*args)


def load_pools(data_dir, method, scope):
    """Загрузить только два квантованных Pool и компактные метки validation."""
    import numpy as np
    from catboost import Pool
    inputs = [pool_info(data_dir, method, split, scope) for split in ('train', 'val')]
    train, val = [Pool('quantized://' + str(info['pool']), thread_count=4) for info in inputs]
    with np.load(inputs[1]['targets']) as saved:
        targets = {name: saved[name] for name in ('label', 'group_id', 'log_id', 'day')}
    np.testing.assert_array_equal(np.asarray(val.get_label(), dtype='float64'), targets['label'])
    if train.get_feature_names() != val.get_feature_names():
        raise ValueError('Порядок признаков train и val отличается')
    return train, val, targets


def search_metrics(model, objective, val, targets):
    """Выбрать trial по той же внешней NDCG, что использована в эксперименте."""
    from train_dfs_catboost import ranking_metrics
    if objective == 'ranker':
        score = model.predict(val, thread_count=4)
        return ranking_metrics(targets['label'], targets['group_id'], score)
    from sklearn.metrics import log_loss
    probabilities = model.predict_proba(val, thread_count=4)
    score = probabilities @ model.classes_.astype(float)
    metrics = ranking_metrics(targets['label'], targets['group_id'], score)
    metrics['logloss'] = float(log_loss(targets['label'], probabilities, labels=model.classes_))
    return metrics


def fit_model(args, data_dir):
    """Обучить одну конфигурацию; большие объекты исчезнут с этим процессом."""
    from catboost import CatBoostClassifier, CatBoostRanker
    scope = 'full' if args.stage == 'refit' else 'search'
    train, val, targets = load_pools(data_dir, args.method, scope)
    parameters = read_json(args.params_file)
    model_type = CatBoostRanker if args.objective == 'ranker' else CatBoostClassifier
    model = model_type(**parameters)
    started = time.perf_counter()
    model.fit(train, eval_set=val)
    fit_seconds = time.perf_counter() - started
    result = {'fit_seconds': fit_seconds, 'best_iteration': model.get_best_iteration(),
              'tree_count': model.tree_count_, 'params': model.get_all_params(),
              'train_rows': train.num_row(), 'val_rows': val.num_row()}
    result['iteration_cap_reached'] = model.get_best_iteration() >= 1550
    result['input_configs'] = {
        split: {'path': str(info['config']), 'sha256': file_hash(info['config'])}
        for split in ('train', 'val')
        for info in [pool_info(data_dir, args.method, split, scope)]}
    if args.stage == 'trial':
        result['metrics'] = search_metrics(model, args.objective, val, targets)
    else:
        from optuna_evaluation import evaluate_and_save
        result.update(evaluate_and_save(model, args.objective, val, targets, args.output))
        result['artifact_sha256'] = {name: file_hash(path)
                                     for name, path in result['artifacts'].items()}
    return result


def parser():
    """Разделить небольшую подготовку и единичное обучение для controller."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--run-name', required=True)
    command.add_argument('--stage', choices=['selection', 'prepare', 'trial', 'refit'], required=True)
    command.add_argument('--method', choices=['manual', 'dfs'])
    command.add_argument('--objective', choices=['ranker', 'pointwise'])
    command.add_argument('--split', choices=['train', 'val'])
    command.add_argument('--scope', choices=['search', 'full'], default='search')
    command.add_argument('--params-file', type=Path)
    command.add_argument('--output', type=Path, required=True)
    return command


def main():
    """Сохранить явный результат или ошибку одного изолированного шага."""
    command = parser()
    args = command.parse_args()
    if args.stage in {'trial', 'refit'} and any(value is None for value in
                                               (args.method, args.objective, args.params_file)):
        command.error('Для trial/refit нужны --method, --objective и --params-file')
    if args.stage == 'prepare' and (args.method is None or args.split is None):
        command.error('Для prepare нужны --method и --split')
    arm_worker_deadline()
    data_dir = run_paths(args.run_name)['data']
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {'completed': False, 'stage': args.stage, 'test_used': False}
    try:
        if args.stage == 'selection':
            from optuna_data import prepare_selection
            prepare_selection(data_dir, train_groups=12000, val_groups=8000, seed=42)
        elif args.stage == 'prepare':
            from optuna_data import prepare_pool
            prepare_pool(data_dir, args.method, args.split, args.scope)
        else:
            result.update(fit_model(args, data_dir))
        result['completed'] = True
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        result.update(elapsed_seconds=time.perf_counter() - started,
                      max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        write_json(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
