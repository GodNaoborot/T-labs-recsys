"""Отдельный контроль C4: пакеты событий, новые признаки и три ранкера."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30
STAGES = ('targets', 'features', 'quantize', 'train')


def write_json(path, value):
    """Заменить небольшой отчет атомарно, не оставляя частичного JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def run_paths(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Некорректное имя запуска')
    return ROOT / 'data/c4' / name, ROOT / 'results/c4' / name


def source_records():
    """Зафиксировать исполняемые исходники; исторические файлы не изменяются."""
    return {str(path.relative_to(ROOT)): file_hash(path)
            for path in sorted((ROOT / 'src').glob('*.py'))}


def verify_sources(protocol):
    if source_records() != protocol['source_sha256']:
        raise ValueError('Исходники изменились после фиксации протокола C4')


def freeze_protocol(data_dir, result_dir, args):
    """Новый запуск получает неизменный протокол до подготовки и обучения."""
    path = result_dir / 'protocol.json'
    if path.exists():
        protocol = json.loads(path.read_text())
        if (protocol['run_name'] != args.run_name
                or protocol['max_training_seconds'] != args.max_training_seconds
                or protocol['train_selection']['whole_mixed_user_day_groups'] != getattr(args, 'train_groups', 40000)):
            raise ValueError('Параметры запуска отличаются от зафиксированных')
        verify_sources(protocol)
        return protocol
    if data_dir.exists() or result_dir.exists():
        raise FileExistsError('Для нового C4 нужна новая пустая папка запуска')
    result_dir.mkdir(parents=True)
    protocol = {
        'run_name': args.run_name, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'train_validation_only', 'test_used': False,
        'models': ['M-t', 'C2b-t', 'D462-t'],
        'max_training_seconds': args.max_training_seconds,
        'budget_scope': 'Общая последовательная фаза моделей, включая загрузку и оценку workers; подготовка отдельно',
        'user_authorization': 'Реализовать C4 с уточнениями и запустить M-t, C2b-t, D462-t; на обучение до 2 часов; test не трогать',
        'epoch_rule': 't = previous strictly smaller raw listen timestamp of this user; target = next distinct timestamp packet',
        'history_rule': 'original timestamp < t; windows [t-window,t); previous packet at t excluded',
        'tie_rule': 'timestamp, immutable physical source row id; no target outcome as tie breaker',
        'target_guard': 'exclude target if ANY attached like/dislike timestamp < t, regardless of final label',
        'label_rule': 'frozen existing labels and attachment rule; like overrides dislike; eligibility still uses that mapping',
        'split': {'train_cutoff': [225*86400, 270*86400], 'val_cutoff': [270*86400, 285*86400],
                  'target_end_strictly_before_next_boundary_minus_seconds': 1800},
        'train_selection': {'whole_mixed_user_day_groups': getattr(args, 'train_groups', 40000), 'replace': False, 'seed': 42},
        'validation': 'all eligible mixed user-day groups; reused exploratory validation',
        'quantization': 'fresh C4 train-only native CatBoost borders; validation never learns borders',
        'comparisons': [['D462-t','M-t'], ['C2b-t','M-t'], ['C2b-t','D462-t']],
        'bootstrap': {'type': 'paired circular day blocks', 'block_days': [3,4],
                      'replicates': 20000, 'seed': 42, 'confidence': .95,
                      'two_sided': True, 'weights': 'groups', 'multiplicity_correction': False},
        'data_dir': str(data_dir), 'result_dir': str(result_dir),
        'source_sha256': source_records(),
        'limitations': ['Timestamp availability is an assumption; not guaranteed before physical listen start',
                        'User-day ranking covers observed events with different cutoffs, not catalog recommendation',
                        'Existing label attachment and selected 462 formulas are inherited from earlier research',
                        'Old test is already open; no test preparation or evaluation is authorized for this queue'],
    }
    write_json(path, protocol)
    snapshot = result_dir / 'source_snapshot'
    snapshot.mkdir()
    for relative in protocol['source_sha256']:
        destination = snapshot / Path(relative).name
        shutil.copyfile(ROOT / relative, destination)
    (result_dir / 'protocol.sha256').write_text(file_hash(path) + '\n')
    return protocol


def worker(args, data_dir, result_dir):
    """Каждый большой этап живет отдельно и освобождает память при выходе."""
    protocol = json.loads((result_dir / 'protocol.json').read_text())
    verify_sources(protocol)
    progress_path = result_dir / 'progress.json'
    def progress(value):
        write_json(progress_path, {'stage': args.stage, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                                   'progress': value})
    if args.stage == 'targets':
        from c4_targets import prepare_targets
        report = prepare_targets(data_dir, result_dir, progress=progress, train_groups=protocol['train_selection']['whole_mixed_user_day_groups'], seed=42)
    elif args.stage == 'features':
        from c4_features import prepare_features
        report = prepare_features(data_dir, result_dir, progress=progress)
    elif args.stage == 'quantize':
        from c4_training import prepare_pools
        report = prepare_pools(data_dir, result_dir, progress=progress)
    elif args.stage == 'train':
        from c4_training import run_training
        report = run_training(data_dir, result_dir,
                              max_training_seconds=args.max_training_seconds, progress=progress)
        if not report.get('completed', False):
            raise RuntimeError('Фаза обучения C4 не завершила все три модели: ' + json.dumps(report, ensure_ascii=False))
    else:
        raise ValueError('Неизвестный worker stage')
    write_json(result_dir / ('stage_' + args.stage + '.json'),
               {'completed': True, 'test_used': False, 'report': report})


def anonymous_memory(pid):
    """Учитывать anonymous RAM, отделяя ее от вытесняемых страниц mmap."""
    try:
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
            if line.startswith('Anonymous:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return 0


def stop_child(process):
    """Завершить только группу дочернего процесса, созданную этой очередью."""
    if process.poll() is not None:
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def resource_sample(process):
    try:
        parent = psutil.Process(process.pid)
        family = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        family = []
    return {'anonymous_gib': sum(anonymous_memory(p.pid) for p in family) / GIB,
            'available_gib': psutil.virtual_memory().available / GIB,
            'windows_disk_free_gib': shutil.disk_usage('/mnt/c' if Path('/mnt/c').is_dir() else ROOT).free / GIB,
            'worker_count': len(family)}


def run_stage(args, stage, data_dir, result_dir, status):
    command = [sys.executable, str(Path(__file__).resolve()), '--run-name', args.run_name,
               '--stage', stage, '--worker', '--max-training-seconds', str(args.max_training_seconds),
               '--train-groups', str(getattr(args, 'train_groups', 40000))]
    log_dir = result_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    status.update(stage=stage, completed=False, worker_pid=None)
    with (log_dir / (stage + '.log')).open('a', buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1',
                                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '4'})
        status['worker_pid'] = process.pid
        write_json(result_dir / 'status.json', status)
        danger_count, last_report = 0, 0
        try:
            while process.poll() is None:
                sample = resource_sample(process)
                status['peak_anonymous_gib'] = max(status.get('peak_anonymous_gib', 0), sample['anonymous_gib'])
                dangerous = sample['anonymous_gib'] > 8.75 or sample['available_gib'] < .75 or sample['windows_disk_free_gib'] < 6
                danger_count = danger_count + 1 if dangerous else 0
                if danger_count >= 3:
                    stop_child(process)
                    raise RuntimeError('Остановка C4 по ресурсам: ' + json.dumps(sample))
                if time.monotonic() - last_report >= 5:
                    write_json(result_dir / 'external_resources.json', {'stage': stage, **sample,
                               'stage_elapsed_seconds': time.monotonic() - started})
                    last_report = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_child(process)
            raise
        if process.returncode:
            raise RuntimeError(f'Этап {stage} завершился с exit_code={process.returncode}; см. logs/{stage}.log')
    record = json.loads((result_dir / ('stage_' + stage + '.json')).read_text())
    if not record['completed']:
        raise RuntimeError('Нет успешного маркера этапа ' + stage)
    status.setdefault('stage_seconds', {})[stage] = time.monotonic() - started
    status['worker_pid'] = None
    write_json(result_dir / 'status.json', status)


def supervisor(args, data_dir, result_dir):
    protocol = freeze_protocol(data_dir, result_dir, args)
    old_status = result_dir / 'status.json'
    if old_status.exists():
        raise FileExistsError('Этот запуск уже стартовал; повтор требует разбора его состояния')
    status = {'run_name': args.run_name, 'completed': False, 'test_used': False,
              'supervisor_pid': os.getpid(), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(result_dir / 'status.json', status)
    started = time.monotonic()
    try:
        stages = STAGES if args.stage == 'all' else (args.stage,)
        for stage in stages:
            verify_sources(protocol)
            run_stage(args, stage, data_dir, result_dir, status)
        status.update(completed=args.stage == 'all', stage='completed' if args.stage == 'all' else args.stage + '_completed',
                      elapsed_seconds=time.monotonic() - started, exit_code=0)
        if args.stage == 'all':
            for name in ['comparison.csv', 'bootstrap_summary.csv']:
                if not (result_dir / name).is_file():
                    raise FileNotFoundError('Нет итогового отчета ' + name)
            for variant in ['M-t', 'C2b-t', 'D462-t']:
                if not (result_dir / 'models' / variant / 'model.cbm').is_file():
                    raise FileNotFoundError('Нет готовой модели ' + variant)
            write_json(result_dir / 'result.json', {**status, 'protocol_sha256': file_hash(result_dir / 'protocol.json')})
    except BaseException as error:
        status.update(completed=False, failed_stage=status.get('stage'), stage='stopped', error=repr(error),
                      elapsed_seconds=time.monotonic() - started, exit_code=1)
        write_json(result_dir / 'stopped.json', status)
        raise
    finally:
        write_json(result_dir / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--stage', choices=('all',) + STAGES, default='all')
    parser.add_argument('--max-training-seconds', type=int, default=7200)
    parser.add_argument('--train-groups', type=int, default=40000,
                        help='Число целых train-групп; исходный эксперимент: 40000')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_training_seconds <= 7200:
        parser.error('Разрешенный общий лимит обучения — не больше 7200 секунд')
    if args.train_groups <= 0:
        parser.error('--train-groups должен быть положительным')
    data_dir, result_dir = run_paths(args.run_name)
    if args.status:
        print((result_dir / 'status.json').read_text())
    elif args.worker:
        worker(args, data_dir, result_dir)
    else:
        supervisor(args, data_dir, result_dir)


if __name__ == '__main__':
    main()
