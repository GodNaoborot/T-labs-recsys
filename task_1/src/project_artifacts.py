"""Пути к текущему сравнению моделей и проверка сохранённых результатов."""
from pathlib import Path
import hashlib
import json

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parent
CURRENT_MANIFEST = TASK_ROOT / 'results/current_models.json'


def file_hash(path):
    """Посчитать SHA-256 порциями, не загружая файл целиком в память."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact_path(relative_path):
    """Разрешить путь относительно task_1 и исключить выход за папку проекта."""
    path = (TASK_ROOT / relative_path).resolve()
    if not path.is_relative_to(TASK_ROOT):
        raise ValueError(f'Путь выходит за task_1: {relative_path}')
    return path


def load_current_artifacts():
    """Получить одну согласованную версию manual/DFS для ноутбуков и отчёта."""
    manifest = json.loads(CURRENT_MANIFEST.read_text())
    artifacts = {}
    for method in ('manual', 'dfs'):
        entry = manifest[method]
        artifacts[method] = {
            'config': artifact_path(entry['config']['path']),
            'metrics': artifact_path(entry['metrics']['path']),
            'models': {name: artifact_path(model['path'])
                       for name, model in entry['models'].items()},
        }
    return artifacts


def verify_current_artifacts():
    """Проверить, что модели и таблицы всё ещё принадлежат записанному сравнению."""
    manifest = json.loads(CURRENT_MANIFEST.read_text())
    checked = []
    for method in ('manual', 'dfs'):
        entry = manifest[method]
        files = [entry['config'], entry['metrics'], *entry['models'].values()]
        for item in files:
            path = artifact_path(item['path'])
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_hash(path) != item['sha256']:
                raise ValueError(f'Изменился файл текущего сравнения: {path}')
            checked.append(item['path'])
    return {'comparison_id': manifest['comparison_id'], 'checked_files': checked,
            'evaluation_split': manifest['evaluation_split'], 'test_used': manifest['test_used']}
