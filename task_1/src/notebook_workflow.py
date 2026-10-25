"""Связать вычислительные этапы ноутбуков одним именем повторного запуска."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import time


TASK_ROOT = Path(__file__).resolve().parents[1]
TEMPORAL = Path('results/experiment-1/temporal_runs')
CONTROLS = Path('results/experiment-1/temporal_controls')


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,29}', value):
        raise ValueError('Имя запуска: 1–30 букв, цифр, _ или -')
    return value


class NotebookWorkflow:
    """report читает публикацию; recompute создаёт новые данные и модели."""
    def __init__(self, mode='report', run_name='repeat_01', download=False,
                 raw_events=None, artist_mapping=None, include_test=False):
        if mode not in ('report', 'recompute'):
            raise ValueError('MODE должен быть report или recompute')
        self.mode = mode
        self.name = safe_name(run_name)
        self.download = download
        self.raw_events = Path(raw_events).resolve() if raw_events else None
        self.artist_mapping = Path(artist_mapping).resolve() if artist_mapping else None
        self.include_test = include_test
        self.done = set()
        if bool(self.raw_events) != bool(self.artist_mapping):
            raise ValueError('Нужны оба локальных входа: события и artist mapping')
        if download and raw_events:
            raise ValueError('Выберите скачивание либо локальные входы')

    @property
    def test_available(self):
        return self.mode == 'report' or self.include_test

    def path(self, stage):
        """Все читатели получают результаты того же нового запуска."""
        old = {
            'eda': '../docs/results/dataset-analysis',
            'baseline': 'results/baselines/timing-20261020',
            'dfs130': 'results/experiment-1/dfs',
            'bootstrap': 'results/experiment-1/dfs/bootstrap_noninferiority',
            'rolling': 'results/experiment-1/rolling_windows',
            'optuna': 'results/tuning/optuna_20261023_2h_v2/comparison',
            'selection': str(TEMPORAL / 'temporal_topk_20261023_v2'),
            'used': str(TEMPORAL / 'temporal_used_20261023_v1'),
            'full': str(TEMPORAL / 'temporal_full462_20261023_v5'),
            'controls': str(CONTROLS / 'dfs462_controls_20261023_v1'),
            'old-test': 'results/final_test/final_test_20261023_v1',
            'corrected': 'results/c4/c4_packet_20261024_v1',
            'corrected-test': 'results/c4_test/c4_test_20261024_v1',
            'channel-control': 'results/c4/no_current_channel',
            'feature-analysis': '../docs/results/feature-analysis',
        }
        new = {
            'eda': f'results/reproduction/{self.name}/eda',
            'baseline': f'results/baselines/runs/{self.name}',
            'dfs130': f'results/experiment-1/runs/{self.name}/dfs',
            'bootstrap': f'results/experiment-1/runs/{self.name}/dfs/bootstrap_noninferiority',
            'rolling': f'results/experiment-1/rolling_windows_runs/{self.name}_windows',
            'optuna': f'results/tuning/{self.name}_tuning/comparison',
            'selection': str(TEMPORAL / (self.name + '_topk')),
            'used': str(TEMPORAL / (self.name + '_used')),
            'full': str(TEMPORAL / (self.name + '_full')),
            'controls': str(CONTROLS / (self.name + '_controls')),
            'old-test': f'results/final_test/{self.name}_test',
            'corrected': f'results/c4/{self.name}_packet',
            'corrected-test': f'results/c4_test/{self.name}_packet_test',
            'channel-control': f'results/c4/{self.name}_channel_control',
            'feature-analysis': f'results/reproduction/{self.name}/feature-analysis',
        }
        return (TASK_ROOT / (old if self.mode == 'report' else new)[stage]).resolve()

    def history_commands(self):
        sources = TASK_ROOT / 'data/history_preparation.json'
        if sources.is_file():
            report = json.loads(sources.read_text())
            if report.get('completed') is not True:
                raise ValueError('Подготовка истории не завершена')
            if not all((TASK_ROOT / 'data' / name).is_file() for name in ['listens.parquet', 'events.parquet']):
                raise FileNotFoundError('Описание истории есть, но файлы истории отсутствуют')
            return []
        command = ['src/prepare_history.py']
        if self.download:
            command += ['--download']
        elif self.raw_events and self.artist_mapping:
            command += ['--raw-events', str(self.raw_events), '--artist-mapping', str(self.artist_mapping)]
        else:
            raise ValueError('Для пересчёта укажите DOWNLOAD_DATA=True либо оба локальных файла')
        return [command]

    def commands(self, stage):
        """Составить команды; само составление ничего не запускает."""
        name = self.name
        baseline = f'results/tuning/{name}_tuning/queue/control_manual_ranker/attempt_0001'
        borders = f'results/baselines/runs/{name}/train_borders.tsv'
        if stage == 'history':
            return self.history_commands()
        if stage == 'channel-control':
            common = ['src/run_channel_control.py', '--training-run', name + '_packet',
                      '--run-name', name + '_channel_control']
            commands = [[*common, '--stage', 'fit']]
            if self.include_test:
                commands.append([*common, '--stage', 'evaluate', '--include-test',
                                 '--test-run', name + '_packet_test'])
            return commands
        if stage == 'eda':
            source = self.raw_events
            if source is None:
                report = TASK_ROOT / 'data/history_preparation.json'
                source = json.loads(report.read_text())['sources']['raw_events']['path'] if report.is_file() else '<prepared-raw-events>'
            return [['src/analyze_dataset.py', '--raw-events', str(source), '--output-dir', str(self.path(stage))]]
        simple = {
            'legacy': [['src/prepare_legacy.py']],
            'baseline': [['src/train_manual_baseline.py', '--run-name', name]],
            'dfs130': ([['src/generate_cumulative.py', '--split', split] for split in ['train', 'val']]
                       + [['src/train_dfs_catboost.py', '--stage', 'prepare', '--split', split] for split in ['train', 'val']]
                       + [['src/train_dfs_catboost.py', '--stage', 'train']]),
            'bootstrap': [['src/run_legacy_bootstrap.py', '--run-name', name, '--output', str(self.path('bootstrap'))]],
            'rolling': [['src/run_rolling_windows.py']],
            'optuna': [['src/run_optuna.py', '--run-name', name + '_tuning', '--budget-hours', '2', '--max-total-hours', '12']],
            'selection': [['src/run_temporal_selection.py', '--legacy-run', name, '--run-name', name + '_topk',
                           '--train-rows', '10000', '--val-rows', '5000', '--iterations', '300', '--budget-minutes', '30',
                           '--k-values', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192']],
            'used': [['src/check_used_temporal_features.py', '--source-run', name + '_topk', '--run-name', name + '_used']],
            'full': [['src/run_temporal_full.py', '--legacy-run', name, '--run-name', name + '_full',
                      '--selection-run', name + '_used', '--definitions-run', name + '_topk', '--baseline-dir', baseline]],
            'controls': [['src/run_temporal_controls.py', '--legacy-run', name, '--run-name', name + '_controls',
                          '--full-run', name + '_full', '--baseline-dir', baseline, '--base-borders', borders, '--max-hours', '6']],
            'direct-intervals': [['src/compare_temporal_controls_pairs.py', '--run-name', name + '_controls',
                                  '--output-name', 'direct_pairs']],
            'test-plan': [['src/prepare_replay_test_plan.py', '--run-name', name + '_test', '--controls-run', name + '_controls',
                           '--full-run', name + '_full', '--baseline-dir', baseline, '--base-borders', borders]],
            'old-test': [['src/run_final_test.py', '--run-name', name + '_test',
                          '--plan', f'results/test_plans/{name}_test/plan.json', '--stage', 'freeze'],
                         ['src/run_final_test.py', '--run-name', name + '_test',
                          '--plan', f'results/test_plans/{name}_test/plan.json', '--stage', 'all']],
            'corrected': [['src/run_c4.py', '--run-name', name + '_packet', '--max-training-seconds', '7200']],
            'corrected-test': [['src/run_c4_test.py', '--run-name', name + '_packet_test', '--training-run', name + '_packet']],
            'feature-analysis': [['src/summarize_selected_features.py', '--training-run', name + '_packet',
                                  '--output-dir', str(self.path('feature-analysis'))]],
            'formula-benchmark': [['src/benchmark_formulas.py', '--output', str(self.path('dfs130') / 'formula_construction_benchmark.json')]],
        }
        if stage == 'diagnostics':
            common = ['--controls-run', name + '_controls']
            output = str(TASK_ROOT / 'results/analysis' / name)
            return [
                ['src/audit_start_time_formula.py', *common, '--output', output + '_formula'],
                ['src/audit_start_time_phase.py', *common, '--legacy-run', name,
                 '--plan', f'results/test_plans/{name}_test/plan.json', '--output', output + '_phase'],
                ['src/audit_start_time_phase_permutation.py', *common, '--output', output + '_permutation'],
                ['src/audit_start_time_dfs_phase_permutation.py', *common, '--full-run', name + '_full',
                 '--output', output + '_dfs_permutation'],
            ]
        return simple[stage]

    def dependencies(self, stage):
        return {
            'history': [], 'eda': ['history'], 'legacy': ['history'], 'baseline': ['legacy'],
            'dfs130': ['baseline'], 'bootstrap': ['dfs130'], 'rolling': ['legacy'],
            'optuna': ['bootstrap'], 'selection': ['dfs130'], 'used': ['selection'],
            'full': ['used', 'optuna'], 'controls': ['full'], 'direct-intervals': ['controls'],
            'test-plan': ['controls'], 'old-test': ['test-plan'], 'diagnostics': ['test-plan'],
            'corrected': ['history'], 'corrected-test': ['corrected'], 'feature-analysis': ['corrected'],
            'channel-control': ['corrected', *(['corrected-test'] if self.include_test else [])],
            'formula-benchmark': [],
        }[stage]

    def verify_outputs(self, stage):
        """Нулевого exit code недостаточно: проверить завершение и выходные файлы."""
        filenames = {
            'eda': ['snapshot.json', 'schema.csv'], 'baseline': ['run_config.json', 'validation_metrics.csv', 'train_borders.tsv'],
            'dfs130': ['run_config.json', 'validation_comparison.csv'],
            'bootstrap': ['summary.csv', 'daily_ndcg.csv'], 'rolling': ['status.json', 'summary.csv'],
            'optuna': ['models.csv', 'summary.csv', 'bootstrap_summary.csv'],
            'selection': ['result.json', 'selection/selected_features.json', 'selection/k_curve.csv'],
            'used': ['result.json', 'protocol.json', 'comparison.csv'],
            'full': ['result.json', 'comparison.csv', 'model/model.cbm'],
            'controls': ['status.json', 'result.json', 'comparison.csv', 'bootstrap_summary.csv'],
            'old-test': ['result.json', 'comparison_sub.csv', 'comparison_full.csv'],
            'corrected': ['status.json', 'result.json', 'comparison.csv', 'bootstrap_summary.csv'],
            'corrected-test': ['status.json', 'result.json', 'comparison.csv', 'bootstrap_summary.csv'],
            'channel-control': ['result.json', 'protocol.json', 'comparison.csv', 'bootstrap_summary.csv'],
            'feature-analysis': ['selected_features.csv', 'provenance.json'],
        }
        if stage == 'legacy':
            paths = [TASK_ROOT / 'data' / name for name in ['samples.parquet', 'features.parquet']]
        elif stage == 'history':
            paths = [TASK_ROOT / 'data' / name for name in ['history_preparation.json', 'listens.parquet', 'events.parquet']]
        elif stage == 'direct-intervals':
            paths = [self.path('controls') / 'direct_pairs/bootstrap_summary.csv']
        elif stage == 'test-plan':
            paths = [TASK_ROOT / f'results/test_plans/{self.name}_test/plan.json']
        elif stage == 'formula-benchmark':
            paths = [self.path('dfs130') / 'formula_construction_benchmark.json']
        elif stage == 'diagnostics':
            prefix = TASK_ROOT / 'results/analysis' / self.name
            paths = [Path(str(prefix) + suffix) / filename for suffix, filename in
                     [('_formula', 'result.json'), ('_phase', 'summary.json'),
                      ('_permutation', 'result.json'), ('_dfs_permutation', 'result.json')]]
        else:
            paths = [self.path(stage) / name for name in filenames[stage]]
        if stage == 'optuna':
            status = json.loads((self.path(stage).parent / 'status.json').read_text())
            if not status.get('completed') or status.get('outcome') != 'completed':
                raise RuntimeError('Optuna завершилась не полностью: ' + str(status.get('outcome')))
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError('Этап не создал обязательный файл: ' + str(path))
            if path.name in ('status.json', 'result.json'):
                if json.loads(path.read_text()).get('completed') is not True:
                    raise RuntimeError('Этап не завершён: ' + str(path))
        from prepare_history import file_hash
        return {str(path): file_hash(path) for path in paths if path.suffix in ('.json', '.csv')}

    def ensure(self, stage):
        """Выполнить зависимости один раз; ошибки не заменять сохранёнными метриками."""
        if self.mode == 'report':
            return
        if stage in self.done:
            return
        if stage == 'history' and (TASK_ROOT / 'data/history_preparation.json').is_file():
            self.history_commands()
            report = json.loads((TASK_ROOT / 'data/history_preparation.json').read_text())
            from prepare_history import file_hash
            for name, expected in report['outputs'].items():
                if file_hash(TASK_ROOT / 'data' / name) != expected['sha256']:
                    raise ValueError('История изменилась после подготовки: ' + name)
            self.done.add(stage)
            return
        if stage in ('old-test', 'corrected-test') and not self.include_test:
            raise ValueError('Для новой оценки test нужен INCLUDE_TEST=True')
        for dependency in self.dependencies(stage):
            self.ensure(dependency)
        commands = self.commands(stage)
        source_hash = hashlib.sha256()
        for source in sorted((TASK_ROOT / 'src').glob('*.py')):
            source_hash.update(source.name.encode())
            source_hash.update(source.read_bytes())
        settings = {'commands': commands, 'source_sha256': source_hash.hexdigest(), 'run_name': self.name}
        folder = TASK_ROOT / 'results/reproduction' / self.name
        record = folder / (stage + '.json')
        if record.exists():
            previous = json.loads(record.read_text())
            if previous.get('completed') and previous.get('settings') == settings:
                if self.verify_outputs(stage) != previous.get('outputs'):
                    raise ValueError('Выходные отчёты изменились: ' + stage)
                self.done.add(stage)
                return
            raise ValueError('Этап уже запускался с другими входами или не завершился: ' + stage)
        folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        value = {'completed': False, 'settings': settings}
        record.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        env = {**os.environ, 'TLAB_EXPERIMENT_RUN': self.name,
               'TLAB_ROLLING_RUN': self.name + '_windows', 'PYTHONUNBUFFERED': '1'}
        try:
            for command in commands:
                print(' '.join([sys.executable, *command]), flush=True)
                subprocess.run([sys.executable, *command], cwd=TASK_ROOT, env=env, check=True)
            outputs = self.verify_outputs(stage)
            value.update(completed=True, seconds=time.monotonic() - started, outputs=outputs)
            self.done.add(stage)
        except BaseException as error:
            value.update(error=str(error), seconds=time.monotonic() - started)
            raise
        finally:
            record.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def from_environment(mode='report', run_name='repeat_01', download=False, include_test=False):
    return NotebookWorkflow(mode, run_name, download,
                            os.getenv('TLAB_RAW_EVENTS'), os.getenv('TLAB_ARTIST_MAPPING'), include_test)
