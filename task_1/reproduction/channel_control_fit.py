"""Portable publication of the original manual48 audit fit.

Default: print the plan without writing files. --execute runs the fixed fit.
Set TLAB_CHANNEL_TRAINING_RUN and TLAB_CHANNEL_CONTROL_RUN to select runs.
The separate default output channel_source_repeat must not exist yet.
"""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'task_1'


def run_name(variable, default):
    value = os.environ.get(variable, default)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Invalid run name in {variable}: {value!r}')
    return value


TRAIN_RUN = run_name('TLAB_CHANNEL_TRAINING_RUN', 'c4_packet_20261024_v1')
REFERENCE_RUN = run_name('TLAB_CHANNEL_REFERENCE_TRAINING_RUN', 'c4_packet_20261024_v1')
TEST_RUN = run_name('TLAB_CHANNEL_TEST_RUN', 'c4_test_20261024_v1')
CONTROL_RUN = run_name('TLAB_CHANNEL_CONTROL_RUN', 'channel_source_repeat')
if CONTROL_RUN == 'no_current_channel':
    raise ValueError('The imported no_current_channel reports cannot be overwritten')
OUT = TASK / 'results/c4' / CONTROL_RUN
DATA = TASK / 'data/c4' / TRAIN_RUN
RESULTS = TASK / 'results/c4' / TRAIN_RUN
REFERENCE = TASK / 'results/c4' / REFERENCE_RUN
TEST_RESULTS = TASK / 'results/c4_test' / TEST_RUN
TEST_DATA = TASK / 'data/c4_test' / TEST_RUN
PY = Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and (not args.execute or os.environ.get('TLAB_CHANNEL_SOURCE_PARENT_PID') != str(os.getppid())):
        parser.error('--worker is reserved for the explicitly executed parent process')
    if not args.execute:
        print(json.dumps({'mode': 'plan', 'training_run': TRAIN_RUN,
                          'output': str(OUT.relative_to(ROOT)), 'active_features': 48,
                          'ignored_feature': 'is_organic', 'maximum_seconds': 2100,
                          'execute_required': True}, indent=2))
        return
    if not args.worker:
        OUT.mkdir(parents=True, exist_ok=False)
    if args.worker:
        sys.path.insert(0, str(TASK / 'src'))
        from c4_training import PARAMETERS
        from temporal_full_training import fit_ranker
        plan = json.loads((RESULTS / 'training_plan.json').read_text())
        names = plan['schemas']['manual49']
        parameters = {**PARAMETERS, 'ignored_features': [names.index('is_organic')]}
        pool = RESULTS / 'quantized/manual49'
        result = fit_ranker(pool / 'train.qbin', pool / 'val.qbin',
                            DATA / 'val_targets.npz', names, OUT / 'model', parameters)
        result.update(active_features=48, exclusion='current target is_organic',
                      original_sources_changed=False, new_hyperparameter_search=False,
                      note='audit control; fixed original configuration and train-only borders')
        (OUT / 'result.json').write_text(json.dumps(result, indent=2))
    else:
        ready = RESULTS / 'pools.json'
        (OUT / 'status.json').write_text(json.dumps({'stage': 'waiting_for_fresh_pools', 'completed': False}))
        while not ready.exists():
            stopped = RESULTS / 'stopped.json'
            if stopped.exists():
                raise RuntimeError('Primary reproduction stopped before pools were ready')
            time.sleep(3)
        if not json.loads(ready.read_text()).get('completed'):
            raise RuntimeError('Incomplete quantization')
        started = time.monotonic()
        (OUT / 'status.json').write_text(json.dumps({'stage': 'training', 'completed': False,
            'started_unix': time.time(), 'maximum_seconds': 2100}))
        with (OUT / 'fit.log').open('w') as stream:
            proc = subprocess.Popen(['nice', '-n', '10', str(PY), str(Path(__file__).resolve()), '--worker', '--execute'],
                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                 env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
                      'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1',
                      'TLAB_CHANNEL_SOURCE_PARENT_PID': str(os.getpid())})
            try:
                code = proc.wait(timeout=2100)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 15)
                proc.wait(timeout=15)
                raise
        (OUT / 'status.json').write_text(json.dumps({'stage': 'completed' if code == 0 else 'failed',
            'completed': code == 0, 'seconds': time.monotonic() - started, 'returncode': code}))
        raise SystemExit(code)

if __name__ == '__main__':
    main()
