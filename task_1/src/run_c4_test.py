"""Оценить три сохранённые модели на прежнем test без обучения."""
from pathlib import Path
from replay_paths import saved_path
import argparse
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import sys
import time

from run_c4 import write_json, file_hash, resource_sample, stop_child

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_RUN = 'c4_packet_20261024_v1'
STAGES = ('targets', 'features', 'quantize', 'evaluate')
VARIANTS = ('M-t', 'C2b-t', 'D462-t')


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def run_paths(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Некорректное имя оценки')
    return ROOT / 'data/c4_test' / name, ROOT / 'results/c4_test' / name


def training_path(name):
    """Найти папку обучения по имени внутри results/c4."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Некорректное имя обучения')
    parent = (ROOT / 'results/c4').resolve()
    path = (parent / name).resolve()
    if path.parent != parent:
        raise ValueError('Папка обучения выходит за пределы results/c4')
    return path


def verify_training_reference(plan, requested=None):
    """Проверить, что план и модели относятся к выбранному обучению."""
    # Старые планы уже содержат точные frozen-пути, но ещё не отдельное имя.
    recorded = saved_path(plan['c4_provenance']['protocol']['path']).parent
    name = plan.get('training_run', recorded.name)
    expected = training_path(name)
    if requested is not None and requested != name:
        raise ValueError('Запрошенное обучение отличается от frozen test-плана')
    if recorded.resolve() != expected:
        raise ValueError('Происхождение плана относится к другому обучению')
    if plan.get('training_result_dir', str(expected)) != str(expected):
        raise ValueError('План содержит другую папку обучения')
    for record in plan['models']:
        variant = record['variant']
        if variant not in VARIANTS:
            raise ValueError('Неизвестный вариант модели в плане')
        model_path = expected / 'models' / variant / 'model.cbm'
        if saved_path(record['model']['path']).resolve() != model_path.resolve():
            raise ValueError('Модель относится к другому обучению: ' + variant)
    return name


def verify_plan(folder, check_sources=True):
    """Проверить неизменность плана, весов, порогов и формул до каждого этапа."""
    folder = Path(folder)
    path = folder / 'plan.json'
    if file_hash(path) != (folder / 'plan.sha256').read_text().strip():
        raise ValueError('Изменился frozen test-план')
    plan = json.loads(path.read_text())
    verify_training_reference(plan)
    records = [record['model'] for record in plan['models']]
    records += list(plan['train_borders'].values())
    records += [plan['selection']] + list(plan['definitions'].values())
    records += list(plan['c4_provenance'].values())
    records += list(plan['sources'].values())
    for record in records:
        if file_hash(saved_path(record['path'])) != record['sha256']:
            raise ValueError('Изменился зафиксированный артефакт: ' + record['path'])
    if check_sources:
        for name, expected in plan['source_sha256'].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError('Изменился исходник test-очереди: ' + name)
    return plan


def build_plan(name, data_dir, result_dir, training_run=DEFAULT_TRAINING_RUN):
    """Проверить готовое обучение без создания папок и чтения test-запросов."""
    c4 = training_path(training_run)
    for filename in ('status.json', 'result.json'):
        if json.loads((c4 / filename).read_text()).get('completed') is not True:
            raise ValueError('Выбранное обучение не завершено: ' + training_run)
    c4_protocol = json.loads((c4 / 'protocol.json').read_text())
    if c4_protocol.get('run_name') != training_run:
        raise ValueError('Имя обучения не совпадает с его протоколом')
    for relative, expected in c4_protocol['source_sha256'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Изменён исторический исходник C4: ' + relative)
    training = json.loads((c4 / 'training_plan.json').read_text())
    generation = json.loads((c4 / 'features_preparation.json').read_text())
    targets = json.loads((c4 / 'targets.json').read_text())
    models = []
    for variant in VARIANTS:
        folder = c4 / 'models' / variant
        control = json.loads((folder / 'control.json').read_text())
        record = artifact(folder / 'model.cbm')
        if not control['completed'] or record['sha256'] != control['model_sha256']:
            raise ValueError('Не совпала готовая модель ' + variant)
        if control['training_plan_sha256'] != file_hash(c4 / 'training_plan.json'):
            raise ValueError('Модели не принадлежат одному обучению C4')
        models.append({'variant': variant, 'schema': control['schema'],
            'active_features': control['active_features'], 'ignored_indices': control['ignored_indices'],
            'tree_count': control['evaluation']['tree_count'], 'model': record})
    from temporal_full_data import DEFAULT_SELECTION, DEFAULT_DEFINITIONS, BRANCHES
    definitions = {parent: artifact(DEFAULT_DEFINITIONS / (parent + '.json')) for parent, _ in BRANCHES}
    if artifact(DEFAULT_SELECTION)['sha256'] != generation['sources_sha256']['selection']:
        raise ValueError('Изменился исходный выбор DFS-формул')
    for parent, record in definitions.items():
        if record['sha256'] != generation['sources_sha256'][parent]:
            raise ValueError('Изменились определения ветви ' + parent)
    train_borders = {}
    for schema in training['schemas']:
        folder = c4 / 'quantized' / schema
        record = artifact(folder / 'train_borders.tsv')
        original = json.loads((folder / 'train_quantization.json').read_text())
        if not original['completed'] or record['sha256'] != original['borders_sha256']:
            raise ValueError('Изменились пороги исходного C4 train: ' + schema)
        train_borders[schema] = record
    plan = {
        'plan_id': name, 'run_name': name, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_run': training_run, 'training_result_dir': str(c4),
        'test_evaluation_authorized': True, 'new_training_allowed': False,
        'test_already_open': True, 'evaluation_kind': 'repeat evaluation after protocol correction; not a new blind test',
        'authorization': 'Оценить M-t, C2b-t, D462-t на прежнем test без переобучения и подбора по нему',
        'sources': targets['sources'], 'models': models, 'schemas': training['schemas'],
        'manual_features': training['schemas']['manual49'], 'dfs_features': training['schemas']['dfs462'],
        'selection': artifact(DEFAULT_SELECTION), 'definitions': definitions,
        'train_borders': train_borders,
        'c4_provenance': {key: artifact(c4 / name) for key, name in
            [('protocol', 'protocol.json'), ('training_plan', 'training_plan.json'),
             ('features', 'features_preparation.json'), ('targets', 'targets.json')]},
        'target_protocol': {
            'cutoff': 'previous strictly smaller original listen timestamp of same user, all target-packet rows share t',
            'test_start': 285 * 86400, 'end': 'all remaining observed next packets in original journal',
            'finite_end_gap': False, 'first_packet': 'excluded',
            'history': 'original_timestamp < t; windows [t-window,t); previous packet at t excluded',
            'reaction_guard': 'exclude ANY attached like/dislike timestamp < t, regardless of final label',
            'label_mapping': 'unchanged saved start_ts attachment, like overrides dislike',
            'group': 'uid*1000+floor(t/DAY); recompute mixed-label groups; retain all eligible test groups',
            'ordering': 'group_id, immutable log_id',
        },
        'metrics': {'ndcg': 'linear NDCG@10', 'query_auc': 'QueryAUC:type=Ranking',
                    'tie_seed': 42, 'group_oracle_atol': 1e-12},
        'comparisons': [['D462-t','M-t'], ['C2b-t','M-t'], ['C2b-t','D462-t']],
        'bootstrap': {'block_days': [3,4], 'replicates': 20000, 'seed': 42,
                      'confidence': .95, 'two_sided': True, 'weights': 'groups',
                      'calendar': 'full actual cutoff-day interval', 'multiplicity_correction': False},
        'selection_on_test_allowed': False, 'new_quantization_borders_allowed': False,
        'data_dir': str(data_dir), 'result_dir': str(result_dir),
        'limitations': ['previously opened temporal segment; no new blind test',
                        'timestamp availability assumption; no guarantee before physical listen start',
                        'inherited label attachment/guard still depend on saved start_ts',
                        'user-day ranking of observed events with different cutoffs; not catalog recommendations',
                        'fixed models and previously selected formulas; marginal intervals, no selection uncertainty'],
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in sorted((ROOT / 'src').glob('*.py'))},
    }
    verify_training_reference(plan, training_run)
    # Проверка завершения обучения не читает и не создаёт test-запросы.
    for record in plan['sources'].values():
        if file_hash(saved_path(record['path'])) != record['sha256']:
            raise ValueError('Изменился исходный журнал: ' + record['path'])
    return plan


def freeze_plan(name, data_dir, result_dir, training_run=DEFAULT_TRAINING_RUN):
    """Сохранить проверенный план до подготовки новых test-запросов."""
    if data_dir.exists() or result_dir.exists():
        raise FileExistsError('Нужны новые папки повторной оценки')
    plan = build_plan(name, data_dir, result_dir, training_run)
    result_dir.mkdir(parents=True)
    write_json(result_dir / 'plan.json', plan)
    (result_dir / 'plan.sha256').write_text(file_hash(result_dir / 'plan.json') + '\n')
    snapshot = result_dir / 'source_snapshot'
    snapshot.mkdir()
    for name in plan['source_sha256']:
        shutil.copyfile(ROOT / name, snapshot / Path(name).name)
    return plan


def worker(stage, data_dir, result_dir, training_run=None):
    plan = verify_plan(result_dir)
    verify_training_reference(plan, training_run)
    def progress(value):
        write_json(result_dir / 'progress.json', {'stage': stage,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'progress': value})
    if stage == 'targets':
        from c4_test_data import prepare_test_targets
        report = prepare_test_targets(data_dir, result_dir, plan, progress=progress)
    elif stage == 'features':
        from c4_test_features import prepare_test_features
        report = prepare_test_features(data_dir, result_dir, plan, progress=progress)
    elif stage == 'quantize':
        from c4_test_evaluation import quantize_test
        report = quantize_test(data_dir, result_dir, plan, progress=progress)
    else:
        from c4_test_evaluation import evaluate_test
        report = evaluate_test(data_dir, result_dir, plan, progress=progress)
        if not report.get('completed'):
            raise ValueError('Нет завершённой оценки трёх моделей')
    verify_plan(result_dir)
    write_json(result_dir / ('stage_' + stage + '.json'), {'completed': True,
        'fit_executed': False, 'stage': stage, 'report': report})


def run_stage(args, stage, data_dir, result_dir, status):
    """Следить за памятью и запускать большие этапы в отдельных процессах."""
    logs = result_dir / 'logs'
    logs.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-name', args.run_name,
               '--stage', stage, '--worker']
    if getattr(args, 'training_run', None) is not None:
        command.extend(['--training-run', args.training_run])
    started = time.monotonic()
    with (logs / (stage + '.log')).open('x', buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '4'})
        status.update(stage=stage, worker_pid=process.pid)
        write_json(result_dir / 'status.json', status)
        danger, last_update = 0, 0
        try:
            while process.poll() is None:
                sample = resource_sample(process)
                status['peak_anonymous_gib'] = max(status.get('peak_anonymous_gib', 0), sample['anonymous_gib'])
                danger = danger + 1 if (sample['anonymous_gib'] > 8.75 or sample['available_gib'] < .75
                    or sample['windows_disk_free_gib'] < 6) else 0
                if danger >= 3:
                    raise RuntimeError('Остановка test по ресурсам: ' + json.dumps(sample))
                if time.monotonic() - args.started > 7200:
                    raise TimeoutError('Технический предел очереди test 7200 секунд')
                if time.monotonic() - last_update >= 5:
                    write_json(result_dir / 'external_resources.json', {'stage': stage, **sample,
                        'stage_elapsed_seconds': time.monotonic() - started})
                    last_update = time.monotonic()
                time.sleep(1)
        finally:
            stop_child(process)
        if process.returncode:
            raise RuntimeError(f'Этап {stage}: exit_code={process.returncode}; logs/{stage}.log')
    if not json.loads((result_dir / ('stage_' + stage + '.json')).read_text())['completed']:
        raise ValueError('Нет успешного маркера ' + stage)
    status.setdefault('stage_seconds', {})[stage] = time.monotonic() - started
    status['worker_pid'] = None
    write_json(result_dir / 'status.json', status)


def supervisor(args, data_dir, result_dir):
    args.training_run = getattr(args, 'training_run', None) or DEFAULT_TRAINING_RUN
    freeze_plan(args.run_name, data_dir, result_dir, args.training_run)
    args.started = time.monotonic()
    status = {'completed': False, 'run_name': args.run_name, 'test_used': True,
              'fit_executed': False, 'supervisor_pid': os.getpid(),
              'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(result_dir / 'status.json', status)
    try:
        for stage in STAGES:
            verify_plan(result_dir)
            run_stage(args, stage, data_dir, result_dir, status)
        for name in ('comparison.csv','bootstrap_summary.csv','test_result.json'):
            if not (result_dir / name).is_file():
                raise FileNotFoundError(name)
        evaluation = json.loads((result_dir / 'test_result.json').read_text())
        if not evaluation['completed'] or [row['variant'] for row in evaluation['models']] != list(VARIANTS):
            raise ValueError('Повторная оценка трёх моделей не завершена')
        for variant in VARIANTS:
            record = json.loads((result_dir / 'models' / variant / 'evaluation.json').read_text())
            if not record['completed'] or record['fit_executed'] is not False:
                raise ValueError('Нет корректной frozen-оценки ' + variant)
        verify_plan(result_dir)
        status.update(completed=True, stage='completed', exit_code=0,
            elapsed_seconds=time.monotonic() - args.started,
            plan_sha256=file_hash(result_dir / 'plan.json'))
        write_json(result_dir / 'result.json', status)
    except BaseException as error:
        status.update(completed=False, failed_stage=status.get('stage'), stage='stopped',
            error=repr(error), exit_code=1, elapsed_seconds=time.monotonic() - args.started)
        write_json(result_dir / 'stopped.json', status)
        raise
    finally:
        write_json(result_dir / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--training-run', help='Имя готового обучения в results/c4; по умолчанию ' + DEFAULT_TRAINING_RUN)
    parser.add_argument('--check-only', action='store_true',
                        help='Проверить артефакты обучения без подготовки или оценки test')
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    data_dir, result_dir = run_paths(args.run_name)
    if args.training_run is not None:
        training_path(args.training_run)
    if args.check_only:
        if args.worker or args.stage or args.status:
            parser.error('--check-only нельзя сочетать с --worker, --stage или --status')
        chosen = args.training_run or DEFAULT_TRAINING_RUN
        plan = build_plan(args.run_name, data_dir, result_dir, chosen)
        print(json.dumps({'completed': True, 'training_run': chosen,
            'models': [record['variant'] for record in plan['models']],
            'test_queries_prepared': False, 'test_evaluated': False, 'fit_executed': False},
            ensure_ascii=False))
    elif args.status:
        print((result_dir / 'status.json').read_text())
    elif args.worker:
        if args.stage is None:
            parser.error('--worker требует --stage')
        worker(args.stage, data_dir, result_dir, args.training_run)
    else:
        if args.stage:
            parser.error('Этапы запускаются только через внутренний worker')
        supervisor(args, data_dir, result_dir)


if __name__ == '__main__':
    main()
