"""Оценка семи сохранённых моделей по замороженному плану; fit отсутствует."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

import final_test_data
from final_test_data import ROOT, PLAN_PATH, sha256, verify_plan, write_json
from replay_paths import project_path, run_name

NAME = 'final_test_20261023_v1'
RESULT = ROOT / 'results/final_test' / NAME
DATA = ROOT / 'data/final_test' / NAME
GIB = 2**30


def verify_execution():
    """Сверить план и техническое приложение перед каждым отдельным этапом."""
    path = RESULT / 'execution_appendix.json'
    execution = json.loads(path.read_text())
    if not execution['test_access_authorized'] or execution['new_training_allowed']:
        raise ValueError('Нет разрешения на сохранённую оценку')
    if sha256(PLAN_PATH) != execution['plan_sha256']:
        raise ValueError('Изменён frozen plan')
    for name, expected in execution['source_sha256'].items():
        if sha256(ROOT / 'src' / name) != expected:
            raise ValueError('После фиксации изменился исходник: ' + name)
    for name, record in execution['input_state'].items():
        stat = (ROOT / 'data' / name).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != record:
            raise ValueError('После фиксации изменились входные данные: ' + name)
    return execution


def freeze():
    """Зафиксировать только технический адаптер и прямое разрешение пользователя."""
    plan = verify_plan()
    if RESULT.exists():
        raise FileExistsError('Папка исполнения уже существует; автоматического повтора нет')
    if shutil.disk_usage('/mnt/c' if Path('/mnt/c').exists() else ROOT).free < 15 * GIB:
        raise RuntimeError('Недостаточный запас физического диска для трёх test-схем')
    source_paths = sorted((ROOT / 'src').glob('*.py'))
    sources = {path.name: sha256(path) for path in source_paths}
    inputs = {}
    for name in ['listens.parquet', 'samples.parquet', 'features.parquet']:
        stat = (ROOT / 'data' / name).stat()
        inputs[name] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    execution = {'plan_id': NAME, 'plan_sha256': sha256(PLAN_PATH),
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'test_access_authorized': True, 'new_training_allowed': False,
        'authorization': 'User requested evaluation of all seven models on test under the frozen plan, without refitting or selection.',
        'scientific_plan_unchanged': True, 'source_sha256': sources,
        'input_state': inputs, 'workers_sequential': True,
        'resource_limits': {'worker_anon_gib': 8.75, 'min_available_gib': 1.25,
                            'min_available_consecutive_samples': 3, 'disk_reserve_gib': 5},
        'synthetic_checks_required_before_test': True,
        'historical_plan_access_flag_left_unchanged': True}
    RESULT.mkdir(parents=True)
    DATA.mkdir(parents=True)
    (RESULT / 'source_snapshot').mkdir()
    for path in source_paths:
        shutil.copyfile(path, RESULT / 'source_snapshot' / path.name)
    shutil.copyfile(PLAN_PATH, RESULT / 'frozen_plan.json')
    write_json(RESULT / 'execution_appendix.json', execution)
    write_json(RESULT / 'status.json', {'completed': False, 'stage': 'frozen_not_started',
                                      'test_opened': False})
    print(json.dumps({'frozen': True, 'plan_sha256': execution['plan_sha256'],
                      'source_files': len(sources), 'test_opened': False}))


def progress(record):
    """Оставить небольшую запись хода генерации вместо печати матриц."""
    write_json(RESULT / 'progress.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def worker(args):
    verify_execution()
    plan = verify_plan(PLAN_PATH)
    if args.stage == 'targets':
        from final_test_data import prepare_targets
        return prepare_targets(DATA, RESULT, plan)
    if args.stage in ['manual', 'dfs', 'quantize']:
        from final_test_features import prepare_manual, prepare_dfs, quantize_schema
        if args.stage == 'manual':
            return prepare_manual(DATA, RESULT, plan, progress=progress)
        if args.stage == 'dfs':
            return prepare_dfs(DATA, RESULT, plan, progress=progress)
        return quantize_schema(DATA, RESULT, plan, args.schema, progress=progress)
    from final_test_evaluation import evaluate_model, summarize, validate_schema
    if args.stage == 'validate':
        record = validate_schema(plan, DATA, args.schema)
        write_json(RESULT / ('alignment_' + args.schema + '.json'), record)
        return record
    if args.stage == 'evaluate':
        for schema in plan['schemas']:
            record = json.loads((RESULT / ('alignment_' + schema + '.json')).read_text())
            if not record['completed']:
                raise ValueError('Не все схемы прошли выравнивание')
        model = next(record for record in plan['models'] if record['id'] == args.model)
        return evaluate_model(model, plan, DATA, RESULT)
    if args.stage == 'summary':
        return summarize(plan, DATA, RESULT)
    raise ValueError('Неизвестный worker-этап')


def stop_worker(process):
    """Остановить только свой worker, не затрагивая другие проекты WSL."""
    if process.poll() is not None:
        return
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    _, alive = psutil.wait_procs(children, timeout=1)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def worker_resources(process):
    """Отличить собственную anonymous RAM от вытесняемых mmap-страниц."""
    status = Path('/proc') / str(process.pid) / 'status'
    try:
        lines = status.read_text().splitlines()
        anonymous = next(int(line.split()[1]) * 1024
                         for line in lines if line.startswith('RssAnon:'))
        rss = psutil.Process(process.pid).memory_info().rss
    except (FileNotFoundError, psutil.NoSuchProcess):
        anonymous, rss = 0, 0
    return {'anonymous_gib': anonymous / GIB, 'rss_gib': rss / GIB,
            'available_gib': psutil.virtual_memory().available / GIB,
            'host_disk_free_gib': shutil.disk_usage('/mnt/c' if Path('/mnt/c').exists() else ROOT).free / GIB,
            'wsl_disk_free_gib': shutil.disk_usage(ROOT).free / GIB}


def run_stage(stage, schema=None, model=None):
    """Выполнить один этап в новом процессе с внешним контролем RAM и диска."""
    identity = '_'.join(str(value) for value in [stage, schema, model] if value)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', stage,
               '--run-name', NAME, '--plan', str(PLAN_PATH)]
    if schema:
        command += ['--schema', schema]
    if model:
        command += ['--model', model]
    write_json(RESULT / 'status.json', {'completed': False, 'stage': identity,
                                       'test_opened': True, 'unix_time': time.time()})
    (RESULT / 'logs').mkdir(exist_ok=True)
    started, low_count, last_saved = time.perf_counter(), 0, 0
    peaks = {'anonymous_gib': 0., 'rss_gib': 0.}
    print('Started: ' + identity, flush=True)
    with (RESULT / 'logs' / (identity + '.log')).open('x', buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        try:
            while process.poll() is None:
                state = worker_resources(process)
                for key in peaks:
                    peaks[key] = max(peaks[key], state[key])
                low_count = low_count + 1 if state['available_gib'] < 1.25 else 0
                reason = None
                if state['anonymous_gib'] > 8.75 or low_count >= 3:
                    reason = 'memory_reserve'
                if min(state['host_disk_free_gib'], state['wsl_disk_free_gib']) < 5:
                    reason = 'disk_reserve'
                if time.monotonic() - last_saved >= 5 or reason:
                    write_json(RESULT / 'resources.json', {'stage': identity,
                        'elapsed_seconds': time.perf_counter() - started, **state, 'peaks': peaks})
                    last_saved = time.monotonic()
                if reason:
                    raise RuntimeError(reason)
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            raise
    record = {'stage': identity, 'exit_code': process.returncode,
              'elapsed_seconds': time.perf_counter() - started, 'peaks': peaks}
    with (RESULT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    if process.returncode:
        raise RuntimeError('Этап завершился с ошибкой; см. ' + str(RESULT / 'logs' / (identity + '.log')))
    print(json.dumps(record), flush=True)


def supervisor():
    """Одна очередь генерации и predict; никаких fit или выбора по test."""
    verify_execution()
    plan = verify_plan()
    if (RESULT / 'events.jsonl').exists() or (RESULT / 'result.json').exists():
        raise FileExistsError('Повтор test требует отдельной регистрации причины')
    started = time.perf_counter()
    try:
        run_stage('targets')
        run_stage('manual')
        run_stage('dfs')
        for schema in ['manual42', 'manual49', 'dfs462']:
            run_stage('quantize', schema=schema)
        for schema in ['manual42', 'manual49', 'dfs462']:
            run_stage('validate', schema=schema)
        for model in plan['models']:
            run_stage('evaluate', model=model['id'])
        run_stage('summary')
        write_json(RESULT / 'result.json', {'completed': True, 'test_evaluated': True,
            'plan_id': NAME, 'plan_sha256': sha256(PLAN_PATH), 'models': 7, 'comparisons': 7,
            'new_training_performed': False, 'elapsed_seconds': time.perf_counter()-started})
        write_json(RESULT / 'status.json', {'completed': True, 'stage': 'finished',
            'test_opened': True, 'elapsed_seconds': time.perf_counter()-started})
    except BaseException as error:
        write_json(RESULT / 'stopped.json', {'reason': str(error), 'type': type(error).__name__,
            'elapsed_seconds': time.perf_counter()-started, 'auto_restart': False})
        write_json(RESULT / 'status.json', {'completed': False, 'stage': 'stopped',
            'error': str(error), 'elapsed_seconds': time.perf_counter()-started})
        raise


def main():
    global NAME, RESULT, DATA, PLAN_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default=NAME)
    parser.add_argument('--plan', default=str(PLAN_PATH))
    parser.add_argument('--stage', required=True, choices=['freeze','all','targets','manual',
        'dfs','quantize','validate','evaluate','summary'])
    parser.add_argument('--schema', choices=['manual42','manual49','dfs462'])
    parser.add_argument('--model')
    args = parser.parse_args()
    NAME = run_name(args.run_name)
    RESULT = ROOT / 'results/final_test' / NAME
    DATA = ROOT / 'data/final_test' / NAME
    PLAN_PATH = project_path(args.plan)
    final_test_data.PLAN_PATH = PLAN_PATH
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'all':
        supervisor()
    else:
        record = worker(args)
        print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


if __name__ == '__main__':
    main()
