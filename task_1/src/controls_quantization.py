"""Квантование 42 ручных и семи сессионных признаков для контролей.

Старые пороги сохраняются. Пороги новых признаков обучаются только на выбранном
train: частичный input_borders иначе задает новым колонкам 32 порога в CatBoost.
"""
from pathlib import Path
from replay_paths import saved_path
import gc
import json
import time

import catboost
import numpy as np
from catboost import Pool

from temporal_full_training import (
    SUPPORTED_CATBOOST, _check_names, _check_pool, _prepare_request,
    _sha256, _write_json, load_targets,
)

BASE_FEATURE_COUNT = 42
SESSION_FEATURE_COUNT = 7
FEATURE_COUNT = BASE_FEATURE_COUNT + SESSION_FEATURE_COUNT
THREADS = 4


def _border_indices(text, feature_count):
    """Проверить номера колонок и вернуть признаки с сохраненными порогами."""
    indices = set()
    for line in text.splitlines():
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) not in (2, 3):
            raise ValueError('Неверная строка файла порогов')
        index = int(fields[0])
        if not 0 <= index < feature_count:
            raise ValueError('Номер признака вне файла порогов')
        indices.add(index)
    return indices


def _constant_columns(matrix):
    """Найти константы; смесь NaN и числа считается двумя разными значениями."""
    result = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        missing = np.isnan(column)
        if missing.all() or (not missing.any() and np.all(column == column[0])):
            result.append(index)
    return result


def _quantize(pool, borders=None, ignored=()):
    """Ограничить нативную квантизацию четырьмя потоками и подсказкой RAM 2 GB."""
    parameters = {
        'border_count': 254, 'random_seed': 42, 'nan_mode': 'Min',
        'thread_count': THREADS, 'used_ram_limit': '2GB',
    }
    if borders is not None:
        parameters['input_borders'] = str(borders)
    if ignored:
        parameters['ignored_features'] = list(ignored)
    # У публичного Pool.quantize нет thread_count; API проверен для этой версии.
    pool._quantize(parameters)


def _pool_values(matrix, ignored):
    """Обнулить только train-константы в отдельной F-копии, не меняя исходник.

    ignored_features сам по себе не запрещает reader строить пороги на val.
    Копия нужна только при наличии констант; максимум около 0.6 GB для full val.
    """
    values = np.array(matrix, dtype='float32', order='F', copy=True) if ignored else matrix
    if ignored:
        values[:, ignored] = 0
    values.flags.writeable = False
    return values


def _learn_session_borders(matrix, names, output_dir):
    """Обучить пороги семи новых колонок без изменения базовых 42 колонок."""
    values = matrix[:, BASE_FEATURE_COUNT:]
    ignored = _constant_columns(values)
    path = output_dir / 'session_train_borders.tsv'
    if len(ignored) == SESSION_FEATURE_COUNT:
        path.write_text('')
        return ''
    values = _pool_values(values, ignored)
    pool = Pool(values, feature_names=names[BASE_FEATURE_COUNT:], thread_count=THREADS)
    _quantize(pool, ignored=ignored)
    pool.save_quantization_borders(str(path))
    del pool
    gc.collect()
    text = path.read_text()
    _border_indices(text, SESSION_FEATURE_COUNT)
    return text


def _combined_borders(base_path, session_text):
    """Перенумеровать новые колонки 0..6 в 42..48 и сохранить исходные пороги."""
    base_text = base_path.read_text()
    _border_indices(base_text, BASE_FEATURE_COUNT)
    if base_text and not base_text.endswith('\n'):
        base_text += '\n'
    added = []
    for line in session_text.splitlines():
        if line:
            index, remaining = line.split('\t', 1)
            added.append(str(int(index) + BASE_FEATURE_COUNT) + '\t' + remaining + '\n')
    return base_text + ''.join(added)


def _open_matrix(path, rows, names):
    """Открыть F-order float32 без копии и отклонить бесконечные значения."""
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    if (matrix.dtype != np.dtype('float32') or matrix.shape != (rows, len(names))
            or not matrix.flags.f_contiguous):
        raise ValueError('Нужна F-order float32 матрица с теми же строками и 49 признаками')
    for index in range(len(names)):
        if np.isinf(matrix[:, index]).any():
            raise ValueError('Бесконечное значение в признаке: ' + names[index])
    return matrix


def _training_report(output_dir, names):
    """Проверить train-пороги перед подготовкой validation."""
    path = output_dir / 'train_report.json'
    if not path.exists():
        raise FileNotFoundError('Сначала подготовьте train в той же папке')
    report = json.loads(path.read_text())
    borders = output_dir / 'train_borders.tsv'
    if (not report.get('completed') or report['features'] != names
            or _sha256(borders) != report['borders_sha256']):
        raise ValueError('Train-пороги или порядок признаков изменились')
    return report


def _recover(report_path, request_path, request):
    """Переиспользовать только завершенные собственные артефакты с теми же входами."""
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Входы augmented-квантизации изменились')
    else:
        if report_path.exists() or report_path.with_name(request['split'] + '.qbin').exists():
            raise FileExistsError('Есть артефакты без собственного request')
        _write_json(request_path, request)
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    if not report.get('completed'):
        raise ValueError('Незавершенный отчет квантизации')
    for kind in ('pool', 'borders', 'targets'):
        if _sha256(saved_path(report[kind + '_path'])) != report[kind + '_sha256']:
            raise ValueError('Изменился завершенный артефакт: ' + kind)
    return report


def prepare_augmented_pool(matrix_path, targets_path, feature_names, output_dir, split,
                           base_borders=None):
    """Сохранить Pool 42+7: базовые пороги прежние, новые обучаются на selected train.

    Train получает base_borders исходных 42 признаков. Validation берет полный
    train_borders.tsv и train_report.json из output_dir, собственных порогов не учит.
    Отсутствующие train-пороги означают константу: колонка игнорируется в обоих split.
    """
    if catboost.__version__ != SUPPORTED_CATBOOST:
        raise RuntimeError('Для другой версии CatBoost повторите тесты native-квантизации')
    if split not in ('train', 'val'):
        raise ValueError('Ожидается split=train или val')
    if (split == 'train') != (base_borders is not None):
        raise ValueError('Только train получает base_borders; val использует готовые train-пороги')
    names = _check_names(feature_names)
    if len(names) != FEATURE_COUNT:
        raise ValueError('Ожидаются ровно 42 ручных и 7 сессионных признаков')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    borders_path = output_dir / 'train_borders.tsv'
    train_report = _training_report(output_dir, names) if split == 'val' else None
    input_path = Path(base_borders).resolve() if split == 'train' else borders_path
    request = _prepare_request(matrix_path, targets_path, names, input_path)
    request.update(split=split, method='controls_augmented_42_plus_7_v1',
                   base_origin='full_train', new_origin='selected_train')
    if train_report is not None:
        request['train_report_sha256'] = _sha256(output_dir / 'train_report.json')
    report_path = output_dir / (split + '_report.json')
    request_path = output_dir / (split + '_request.json')
    recovered = _recover(report_path, request_path, request)
    if recovered is not None:
        return recovered
    started = time.perf_counter()
    targets = load_targets(targets_path)
    matrix = _open_matrix(matrix_path, len(targets['label']), names)
    if split == 'train':
        session_text = _learn_session_borders(matrix, names, output_dir)
        borders_path.write_text(_combined_borders(input_path, session_text))
    text = borders_path.read_text()
    present = _border_indices(text, len(names))
    ignored = sorted(set(range(len(names))) - present)
    if len(ignored) == len(names):
        raise ValueError('Все признаки константны: CatBoost Pool не содержит полезных колонок')
    if train_report is not None and ignored != train_report['ignored_features']:
        raise ValueError('Список констант отличается от train')
    values = _pool_values(matrix, ignored)
    pool = Pool(values, label=targets['label'], group_id=targets['group_id'],
                feature_names=names, thread_count=THREADS)
    quantize_started = time.perf_counter()
    _quantize(pool, borders_path, ignored)
    quantize_seconds = time.perf_counter() - quantize_started
    _check_pool(pool, targets, names)
    applied = output_dir / (split + '_applied_borders.tsv')
    pool.save_quantization_borders(str(applied))
    if applied.read_bytes() != borders_path.read_bytes():
        raise AssertionError('Нативный Pool изменил зафиксированные train-пороги')
    pool_path = output_dir / (split + '.qbin')
    partial = output_dir / (split + '.qbin.partial')
    pool.save(str(partial))
    del pool, values, matrix
    gc.collect()
    partial.replace(pool_path)
    saved_targets = output_dir / (split + '_targets.npz')
    np.savez_compressed(saved_targets, **targets)
    report = {
        'completed': True, 'split': split, 'features': names, 'feature_count': len(names),
        'rows': len(targets['label']), 'groups': int(len(np.unique(targets['group_id']))),
        'pool_path': str(pool_path), 'borders_path': str(borders_path),
        'targets_path': str(saved_targets), 'matrix_path': str(Path(matrix_path).resolve()),
        'source_targets_path': str(Path(targets_path).resolve()),
        'source_targets_sha256': request['source_targets_sha256'],
        'pool_sha256': _sha256(pool_path), 'borders_sha256': _sha256(borders_path),
        'targets_sha256': _sha256(saved_targets), 'ignored_features': ignored,
        'ignored_feature_names': [names[index] for index in ignored],
        'constant_columns_zeroed_before_quantization': True,
        'base_origin': 'full_train', 'new_origin': 'selected_train',
        'base_feature_count': BASE_FEATURE_COUNT, 'new_feature_count': SESSION_FEATURE_COUNT,
        'quantize_seconds': quantize_seconds, 'elapsed_seconds': time.perf_counter() - started,
        'applied_borders_verified': True, 'test_used': False,
        'quantization': {'border_count': 254, 'nan_mode': 'Min', 'random_seed': 42,
                         'thread_count': THREADS, 'used_ram_limit': '2GB',
                         'catboost': catboost.__version__,
                         'method': 'train-only added borders; complete fixed border file'},
    }
    _write_json(report_path, report)
    return report
