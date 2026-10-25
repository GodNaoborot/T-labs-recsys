"""Внешний контроль памяти отдельной очереди контрольных моделей."""
from pathlib import Path
import argparse
import json
import re
import signal
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/experiment-1/temporal_controls'
RUNNER = 'run_temporal_controls.py'
GIB = 2**30


def _option(command, name):
    """Прочитать значение отдельного аргумента, без поиска по подстроке."""
    try:
        position = command.index(name)
    except ValueError:
        return None
    return command[position + 1] if position + 1 < len(command) else None


def _matches(command, run_name, worker=False):
    script_matches = any(Path(part).name == RUNNER for part in command)
    return (script_matches and _option(command, '--run-name') == run_name
            and (not worker or _option(command, '--stage') is not None))


def _write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _parent_alive(parent, birth):
    try:
        return (parent.is_running() and parent.create_time() == birth
                and parent.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def _workers(parent, run_name):
    """Найти только worker-потомков проверенной очереди и снять их ресурсы."""
    result = []
    for child in parent.children(recursive=True):
        try:
            command = child.cmdline()
            if not _matches(command, run_name, worker=True):
                continue
            status = Path(f'/proc/{child.pid}/status').read_text().splitlines()
            anonymous = next(int(line.split()[1]) * 1024
                             for line in status if line.startswith('RssAnon:'))
            result.append({
                'pid': child.pid, 'create_time': child.create_time(),
                'command': command, 'stage': _option(command, '--stage'),
                'variant': _option(command, '--variant'),
                'split': _option(command, '--split'),
                'rss_gib': child.memory_info().rss / GIB,
                'anonymous_rss_gib': anonymous / GIB,
            })
        except (psutil.NoSuchProcess, FileNotFoundError):
            continue
    return result


def _same_worker(record, run_name):
    """PID можно использовать повторно: перед сигналом сверяем время и команду."""
    try:
        child = psutil.Process(record['pid'])
        if (child.create_time() == record['create_time']
                and child.cmdline() == record['command']
                and _matches(child.cmdline(), run_name, worker=True)
                and child.status() != psutil.STATUS_ZOMBIE):
            return child
    except psutil.NoSuchProcess:
        pass
    return None


def _stop_workers(records, run_name):
    """Завершить только известные собственные процессы; при необходимости убить."""
    selected = []
    for record in records:
        child = _same_worker(record, run_name)
        if child is not None:
            try:
                child.send_signal(signal.SIGTERM)
                selected.append(child)
            except psutil.NoSuchProcess:
                pass
    if not selected:
        return
    _, alive = psutil.wait_procs(selected, timeout=5)
    alive_pids = {child.pid for child in alive}
    for record in records:
        if record['pid'] in alive_pids:
            child = _same_worker(record, run_name)
            if child is not None:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass


def _resource_reason(state, low_count):
    """Пороги действуют на anonymous RAM и фактически доступную память WSL."""
    if state['wsl_disk_free_gib'] < 5:
        return 'disk_reserve_wsl'
    if state['host_disk_free_gib'] < 5:
        return 'disk_reserve_windows'
    if state['anonymous_rss_gib'] > 8.75 or low_count >= 3:
        return 'memory_reserve'
    return None


def watch(parent_pid, run_name, folder):
    """Проверять ресурсы раз в секунду, пока существует исходная очередь."""
    parent = psutil.Process(parent_pid)
    if not _matches(parent.cmdline(), run_name):
        raise ValueError('PID не принадлежит указанной очереди контролей')
    birth = parent.create_time()
    known, peaks = {}, {}
    low_count, peak_rss, peak_anon = 0, 0.0, 0.0
    try:
        while _parent_alive(parent, birth):
            workers = _workers(parent, run_name)
            for worker in workers:
                known[worker['pid']] = worker
                key = '/'.join(str(worker[name] or '-') for name in ['stage', 'variant', 'split'])
                peak = peaks.setdefault(key, {'rss_gib': 0.0, 'anonymous_rss_gib': 0.0})
                for name in ['rss_gib', 'anonymous_rss_gib']:
                    peak[name] = max(peak[name], worker[name])
            rss = max((worker['rss_gib'] for worker in workers), default=0.0)
            anonymous = max((worker['anonymous_rss_gib'] for worker in workers), default=0.0)
            peak_rss, peak_anon = max(peak_rss, rss), max(peak_anon, anonymous)
            available = psutil.virtual_memory().available / GIB
            low_count = low_count + 1 if available < 1.25 else 0
            state = {
                'parent_pid': parent_pid, 'parent_create_time': birth,
                'workers': workers, 'worker_pids': [worker['pid'] for worker in workers],
                'worker_rss_gib': rss, 'anonymous_rss_gib': anonymous,
                'peak_worker_rss_gib': peak_rss, 'peak_anonymous_rss_gib': peak_anon,
                'peak_by_worker': peaks, 'available_gib': available,
                'host_disk_free_gib': psutil.disk_usage('/mnt/c' if Path('/mnt/c').exists() else str(Path(__file__).resolve().parents[1])).free / GIB,
                'wsl_disk_free_gib': psutil.disk_usage(ROOT).free / GIB,
                'low_memory_samples': low_count, 'unix_time': time.time(),
            }
            _write_json(folder / 'external_resources.json', state)
            reason = _resource_reason(state, low_count)
            if reason:
                _write_json(folder / 'external_stopped.json', {'reason': reason, **state})
                _stop_workers(list(known.values()), run_name)
                return
            time.sleep(1)
        # Detached worker не должен остаться без защиты после смерти очереди.
        alive = [worker for worker in known.values() if _same_worker(worker, run_name) is not None]
        if alive:
            _write_json(folder / 'external_stopped.json', {
                'reason': 'parent_exited', 'parent_pid': parent_pid,
                'worker_pids': [worker['pid'] for worker in alive], 'unix_time': time.time()})
            _stop_workers(alive, run_name)
    except Exception as error:
        _write_json(folder / 'external_stopped.json', {
            'reason': 'guard_error', 'parent_pid': parent_pid,
            'error': str(error), 'unix_time': time.time()})
        _stop_workers(list(known.values()), run_name)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_name):
        raise ValueError('Некорректное имя запуска')
    folder = RESULTS / args.run_name
    if folder.resolve().parent != RESULTS.resolve() or not folder.is_dir():
        raise ValueError('Нужна существующая папка указанного запуска контролей')
    try:
        watch(args.parent_pid, args.run_name, folder)
    except psutil.NoSuchProcess:
        pass


if __name__ == '__main__':
    main()
