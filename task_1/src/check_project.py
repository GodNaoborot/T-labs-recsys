"""Быстрая проверка проекта без генерации миллионов строк и обучения моделей."""
from pathlib import Path
import argparse
import ast
import json
import subprocess
import sys

from project_artifacts import TASK_ROOT, PROJECT_ROOT, verify_current_artifacts


def check_sources():
    """Проверить синтаксис скриптов и ячеек, не выполняя их."""
    scripts = list((TASK_ROOT / 'src').glob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    notebooks = list((TASK_ROOT / 'notebooks').glob('*.ipynb'))
    code_cells = 0
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for number, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'{path.name}:cell{number}')
                code_cells += 1
    return {'scripts': len(scripts), 'notebooks': len(notebooks), 'code_cells': code_cells}


def main(argv=None):
    """Сверить текущие артефакты; по запросу выполнить небольшие проверки расчётов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Проверить расчёты на маленьких примерах')
    parser.add_argument('--skip-artifacts', action='store_true',
                        help='Проверить публичную копию без локальных файлов моделей')
    args = parser.parse_args(argv)
    artifacts = ({'status': 'skipped', 'reason': 'Явно задан --skip-artifacts'}
                 if args.skip_artifacts else verify_current_artifacts())
    result = {'artifacts': artifacts, 'syntax': check_sources()}
    if args.smoke:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                        str(TASK_ROOT / 'tests'), '-p', 'test_*.py'],
                       cwd=PROJECT_ROOT, check=True)
        result['smoke'] = 'passed'
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
