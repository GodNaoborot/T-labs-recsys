"""Выполнить ноутбуки в режиме отчётов либо полного пересчёта."""
from pathlib import Path
import argparse
import json

from run_report_notebooks import NOTEBOOKS, TASK_ROOT, execute_notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['report', 'recompute'], default='report')
    parser.add_argument('--run-name', default='repeat_01')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--raw-events', type=Path)
    parser.add_argument('--artist-mapping', type=Path)
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--notebooks', nargs='+', choices=NOTEBOOKS, default=list(NOTEBOOKS))
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--check', action='store_true', help='Выполнить без записи notebook-файлов')
    output.add_argument('--output-dir', type=Path, help='Записать выполненные ноутбуки отдельно')
    output.add_argument('--in-place', action='store_true',
                        help='Обновить исходные ноутбуки; только для report')
    args = parser.parse_args(argv)
    if args.in_place and args.mode != 'report':
        parser.error('--in-place разрешён только в режиме report')
    from notebook_workflow import safe_name
    safe_name(args.run_name)
    if bool(args.raw_events) != bool(args.artist_mapping):
        parser.error('Локальные входы передаются вместе')
    if args.download and args.raw_events:
        parser.error('Выберите скачивание либо локальные входы')
    environment = {'TLAB_NOTEBOOK_MODE': args.mode, 'TLAB_NOTEBOOK_RUN': args.run_name,
                   'TLAB_NOTEBOOK_DOWNLOAD': '1' if args.download else '0',
                   'TLAB_INCLUDE_TEST': '1' if args.include_test else '0',
                   'TLAB_RAW_EVENTS': str(args.raw_events.resolve()) if args.raw_events else '',
                   'TLAB_ARTIST_MAPPING': str(args.artist_mapping.resolve()) if args.artist_mapping else ''}
    records = []
    for name in args.notebooks:
        path = TASK_ROOT / 'notebooks' / name
        # Новые выводы хранятся отдельно от исходного исследования.
        output = TASK_ROOT / 'results/reproduction' / args.run_name / 'notebooks' / name
        if args.check:
            destination = None
        elif args.output_dir is not None:
            destination = args.output_dir / name
        elif args.mode == 'recompute':
            destination = output
        else:
            destination = path if args.in_place else None
        records.append(execute_notebook(path, 172800 if args.mode == 'recompute' else 180,
                                        destination, environment=environment))
    print(json.dumps({'mode': args.mode, 'test_requested': args.include_test,
                      'notebooks': records}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
