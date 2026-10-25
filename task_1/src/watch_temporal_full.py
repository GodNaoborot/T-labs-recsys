"""Внешний контроль RAM: работает и пока нативный CatBoost удерживает Python GIL."""
from pathlib import Path
import argparse
import json
import signal
import time

import psutil


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def watch(parent_pid, run_name, folder):
    """Наблюдать только явно выбранную очередь и ее собственный worker."""
    parent = psutil.Process(parent_pid)
    if 'run_temporal_full.py' not in ' '.join(parent.cmdline()) or run_name not in parent.cmdline():
        raise ValueError('PID не принадлежит указанной очереди')
    birth = parent.create_time()
    low_count, peak = 0, 0.0
    while parent.is_running() and parent.create_time() == birth and parent.status() != psutil.STATUS_ZOMBIE:
        workers = []
        for child in parent.children(recursive=True):
            try:
                command = child.cmdline()
                if run_name in command and '--stage' in command:
                    workers.append(child)
            except psutil.NoSuchProcess:
                continue
        rss_values = []
        anon_values = []
        for child in workers:
            try:
                rss_values.append(child.memory_info().rss)
                lines = Path(f'/proc/{child.pid}/status').read_text().splitlines()
                anon_values.append(next(int(line.split()[1]) * 1024 for line in lines if line.startswith('RssAnon:')))
            except (psutil.NoSuchProcess, FileNotFoundError):
                pass
        rss = max(rss_values, default=0) / 2**30
        peak = max(peak, rss)
        anonymous = max(anon_values, default=0) / 2**30
        available = psutil.virtual_memory().available / 2**30
        free = psutil.disk_usage('/mnt/c' if Path('/mnt/c').exists() else str(Path(__file__).resolve().parents[1])).free / 2**30
        low_count = low_count + 1 if available < 1.25 else 0
        state = {'parent_pid': parent_pid, 'worker_pids': [child.pid for child in workers],
                 'worker_rss_gib': rss, 'anonymous_rss_gib': anonymous, 'peak_worker_rss_gib': peak,
                 'available_gib': available, 'host_disk_free_gib': free, 'unix_time': time.time()}
        write_json(folder / 'external_resources.json', state)
        reason = 'disk_reserve' if free < 5 else 'memory_reserve' if anonymous > 8.75 or low_count >= 3 else None
        if reason:
            write_json(folder / 'external_stopped.json', {'reason': reason, **state})
            for child in workers:
                child.send_signal(signal.SIGTERM)
            return
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / 'results/experiment-1/temporal_runs' / args.run_name
    if folder.resolve().parent != (root / 'results/experiment-1/temporal_runs').resolve():
        raise ValueError('Некорректное имя запуска')
    try:
        watch(args.parent_pid, args.run_name, folder)
    except psutil.NoSuchProcess:
        pass


if __name__ == '__main__':
    main()
