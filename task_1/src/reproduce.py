"""Подготовить историю и повторить основной эксперимент отдельным запуском.

По умолчанию печатается только план. Дорогие этапы требуют явного --stage.
Исторические эксперименты и повторный поиск признаков в этот маршрут не входят.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys


TASK_ROOT = Path(__file__).resolve().parents[1]
STAGES = ('plan', 'reports', 'prepare', 'train', 'evaluate', 'all')
FORMULA_INPUTS = (
    'results/experiment-1/temporal_controls/dfs462_controls_20261023_v1/protocol.json',
    'results/experiment-1/temporal_runs/temporal_used_20261023_v1/protocol.json',
) + tuple('data/experiment-1/temporal_runs/temporal_topk_20261023_v2/definitions/'
          + entity + '.json' for entity in ('Users', 'Items', 'UserItem', 'UserArtist'))


def run_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,58}', value):
        raise argparse.ArgumentTypeError('Имя: до 59 букв, цифр, _ и -, без путей')
    return value


def build_commands(args):
    """Собрать команды подготовки, обучения и оценки."""
    python = sys.executable
    prepare = [python, 'src/prepare_history.py', '--output-dir', 'data']
    if args.download:
        prepare += ['--download']
        if args.revision:
            prepare += ['--revision', args.revision]
    elif args.raw_events is not None and args.artist_mapping is not None:
        prepare += ['--raw-events', str(args.raw_events.resolve()),
                    '--artist-mapping', str(args.artist_mapping.resolve())]
    return {
        'reports': [python, 'src/run_report_notebooks.py'],
        'prepare': prepare,
        'train': [python, 'src/run_c4.py', '--run-name', args.run_name,
                  '--max-training-seconds', str(args.max_training_seconds)],
        'evaluate': [python, 'src/run_c4_test.py', '--run-name', args.run_name + '_test',
                     '--training-run', args.run_name],
    }


def requested_stages(args):
    return ('prepare', 'train', 'evaluate') if args.stage == 'all' else (args.stage,)


def check_inputs(args, stages):
    """Проверить входы и свободное имя до запуска расчёта."""
    if 'train' in stages:
        for name in FORMULA_INPUTS:
            if not (TASK_ROOT / name).is_file():
                raise FileNotFoundError('Отсутствует опубликованное описание признаков: ' + name)
    if 'prepare' in stages:
        if not args.download and not (args.raw_events and args.artist_mapping):
            raise ValueError('Нужны --download или оба локальных входа: --raw-events и --artist-mapping')
        for path in [args.raw_events, args.artist_mapping]:
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        for name in ['listens.parquet', 'events.parquet', 'history_preparation.json']:
            if (TASK_ROOT / 'data' / name).exists():
                raise FileExistsError('История уже есть; используйте отдельную чистую копию или --stage train')
    elif 'train' in stages:
        for name in ['listens.parquet', 'events.parquet']:
            if not (TASK_ROOT / 'data' / name).is_file():
                raise FileNotFoundError('Сначала --stage prepare: нет data/' + name)
    if 'train' in stages:
        for folder in ['data/c4', 'results/c4']:
            if (TASK_ROOT / folder / args.run_name).exists():
                raise FileExistsError('Имя обучения уже использовано; задайте новый --run-name')
    if 'evaluate' in stages:
        for folder in ['data/c4_test', 'results/c4_test']:
            if (TASK_ROOT / folder / (args.run_name + '_test')).exists():
                raise FileExistsError('Имя test-оценки уже использовано')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', type=run_name, default='reproduce_01')
    parser.add_argument('--stage', choices=STAGES, default='plan')
    parser.add_argument('--download', action='store_true', help='Скачать входы Yambda при подготовке')
    parser.add_argument('--revision', help='Зафиксированная ревизия Hugging Face для скачивания')
    parser.add_argument('--raw-events', type=Path)
    parser.add_argument('--artist-mapping', type=Path)
    parser.add_argument('--max-training-seconds', type=int, default=7200)
    args = parser.parse_args(argv)
    if not 1 <= args.max_training_seconds <= 7200:
        parser.error('Лимит фазы обучения должен быть от 1 до 7200 секунд')
    if bool(args.raw_events) != bool(args.artist_mapping):
        parser.error('Локальные входы задаются вместе')
    if args.download and args.raw_events:
        parser.error('Выберите --download или локальные входы')
    if args.revision and not args.download:
        parser.error('--revision применяется вместе с --download')
    commands = build_commands(args)
    if args.stage == 'plan':
        print(json.dumps({'executed': False, 'cwd': str(TASK_ROOT),
                          'commands': commands,
                          'note': 'all запускает подготовку, три обучения и повторную оценку открытого test; reports только читает опубликованные результаты'},
                         ensure_ascii=False, indent=2))
        return
    stages = requested_stages(args)
    check_inputs(args, stages)
    for stage in stages:
        print('Этап: ' + stage, flush=True)
        subprocess.run(commands[stage], cwd=TASK_ROOT, check=True)
    print(json.dumps({'completed': True, 'stages': stages, 'run_name': args.run_name,
                      'training_results': 'results/c4/' + args.run_name,
                      'test_results': 'results/c4_test/' + args.run_name + '_test',
                      'original_reports_replaced': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
