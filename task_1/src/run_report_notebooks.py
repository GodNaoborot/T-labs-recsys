"""Выполнить отчётные ноутбуки в отдельных ядрах без данных и моделей."""
from pathlib import Path
import argparse
import json
import re
import os
from unittest.mock import patch
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


TASK_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('dataset-analysis.ipynb', 'model-baseline.ipynb',
             'experiment-1.ipynb', 'experiment-2.ipynb', 'final-comparison.ipynb')


def make_kernel(directory):
    """Запустить ядро в текущей среде Python."""
    folder = directory / 'recsys-report'
    folder.mkdir()
    specification = {
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'RecSys report', 'language': 'python',
    }
    (folder / 'kernel.json').write_text(json.dumps(specification))
    manager = KernelSpecManager(kernel_dirs=[str(directory)], ensure_native_kernel=False)
    return KernelManager(kernel_name='recsys-report', kernel_spec_manager=manager)


def relocate_links(notebook, source, destination):
    """Сохранить рабочие ссылки, если выполненный notebook записан в другую папку."""
    if source.parent.resolve() == destination.parent.resolve():
        return
    def replace(match):
        target = match.group(2)
        if target.startswith(('/', '#')) or '://' in target:
            return match.group(0)
        path, separator, anchor = target.partition('#')
        resolved = (source.parent / path).resolve()
        if not resolved.exists():
            return match.group(0)
        relative = Path(os.path.relpath(resolved, destination.parent)).as_posix()
        return match.group(1) + relative + (separator + anchor if separator else '') + ')'
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            cell.source = re.sub(r'(\]\()([^\s)]+)\)', replace, cell.source)


def execute_notebook(path, timeout, destination=None, environment=None):
    """Каждый файл получает новое ядро; ошибка прерывает проверку до записи файла."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        cell.metadata.pop('execution', None)
    if notebook.metadata.get('tlab_report') is not True:
        raise ValueError(f'Не объявлен отчётный режим: {path.name}')
    first = next(cell for cell in notebook.cells if cell.cell_type == 'code')
    original = first.source
    first.source = ('import sys as _report_kernel\n'
                    f'assert _report_kernel.prefix == {sys.prefix!r}, _report_kernel.prefix\n'
                    'del _report_kernel\n' + original)
    started = time.monotonic()
    execution_env = {'TLAB_NOTEBOOK_MODE': 'report', **(environment or {})}
    with patch.dict(os.environ, execution_env), tempfile.TemporaryDirectory(prefix='recsys-kernel-') as temporary:
        kernel = make_kernel(Path(temporary))
        try:
            NotebookClient(notebook, km=kernel, timeout=timeout, record_timing=False).execute(
                cwd=str(path.parent))
        finally:
            if kernel.has_kernel:
                kernel.shutdown_kernel(now=True)
    first.source = original
    counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == 'code']
    if counts != list(range(1, len(counts) + 1)):
        raise AssertionError(f'Непоследовательные номера в {path.name}: {counts}')
    notebook.metadata['kernelspec'] = {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        relocate_links(notebook, path, destination)
        nbformat.write(notebook, destination)
    return {'notebook': path.name, 'code_cells': len(counts),
            'sequential_from_one': True, 'kernel_matches_current_environment': True,
            'seconds': time.monotonic() - started, 'saved': destination is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--check', action='store_true',
                        help='Выполнить без записи; поведение по умолчанию')
    output.add_argument('--output-dir', type=Path,
                        help='Сохранить выполненные файлы в отдельной папке')
    output.add_argument('--in-place', action='store_true',
                        help='Явно обновить выводы исходных ноутбуков')
    parser.add_argument('--timeout', type=int, default=180,
                        help='Лимит одной отчётной ячейки, секунд')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout должен быть положительным')
    records = []
    for name in NOTEBOOKS:
        path = TASK_ROOT / 'notebooks' / name
        destination = (args.output_dir / name) if args.output_dir else (path if args.in_place else None)
        print(f'Выполнение {name}', flush=True)
        record = execute_notebook(path, args.timeout, destination)
        records.append(record)
        print(f"Готово: {record['code_cells']} ячеек, {record['seconds']:.1f} с", flush=True)
    print(json.dumps({'mode': 'saved_reports', 'new_research_training': False,
                      'test_evaluation': False, 'notebooks': records},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
