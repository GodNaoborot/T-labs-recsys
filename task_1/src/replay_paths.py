"""Переносимые входы новых запусков без изменения исторических протоколов."""
from pathlib import Path
import os
import re

from project_artifacts import TASK_ROOT


def run_name(value):
    """Проверить имя отдельного запуска без компонентов пути."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError("Имя запуска: буквы, цифры, _ и -, без пути")
    return value


def legacy_featuretools_dir(root=None):
    """Найти входы DFS130 текущего повторного запуска."""
    root = TASK_ROOT if root is None else Path(root)
    name = os.environ.get("TLAB_EXPERIMENT_RUN")
    if name:
        return root / "data/experiment-1/runs" / run_name(name) / "featuretools"
    return root / "data/experiment-1/featuretools"


def project_path(value):
    """Разрешить CLI-путь относительно task_1."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else TASK_ROOT / path


def saved_path(value):
    """Перенести сохранённый путь task_1 в текущую копию проекта."""
    path = Path(value)
    if not path.is_absolute():
        return TASK_ROOT / path
    parts = path.parts
    if "task_1" in parts:
        suffix = parts[parts.index("task_1") + 1:]
        return TASK_ROOT.joinpath(*suffix)
    return path


def configure_legacy(name):
    """Передать выбранный набор targets/metadata дочерним процессам."""
    if name:
        os.environ["TLAB_EXPERIMENT_RUN"] = run_name(name)
