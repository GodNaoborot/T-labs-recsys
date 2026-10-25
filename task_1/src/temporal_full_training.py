"""Квантование полных временных признаков с диска и обучение ранкера.

Float32-матрица читается небольшими порциями. CatBoost получает числовой
текстовый файл сразу в режиме квантования, без полного raw Pool в памяти.
"""
from pathlib import Path
import gc
import hashlib
import inspect
import json
import mmap
import time

import catboost
from catboost import CatBoostRanker, Pool
import numpy as np
import pyarrow as pa
import pyarrow.csv as csv

from optuna_evaluation import ARTIFACT_NAMES, evaluate_and_save

BATCH_ROWS = 8192
THREADS = 4
SUPPORTED_CATBOOST = '1.2.10'


def _sha256(path):
    """Посчитать хеш файла без загрузки его содержимого целиком."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _emit(progress, stage, split, rows, total_rows):
    event = {'stage': stage, 'split': split, 'rows': int(rows), 'total_rows': int(total_rows)}
    if progress is not None:
        progress(event)


def load_targets(path):
    """Прочитать канонические цели, сохранив порядок строк и целых групп."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in ('label', 'group_id', 'log_id')}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or not len(arrays['label']):
        raise ValueError('Targets должны иметь одинаковую ненулевую длину')
    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError('Targets должны быть одномерными')
    if not np.isin(arrays['label'], [0, 1, 2]).all():
        raise ValueError('Нужны метки 0/1/2')
    groups = arrays['group_id']
    if groups.dtype.kind not in 'iu' or np.any(groups[1:] < groups[:-1]):
        raise ValueError('Targets должны идти по целочисленному group_id')
    if arrays['log_id'].dtype.kind not in 'iu':
        raise ValueError('log_id должен быть целочисленным')
    arrays['day'] = groups % 1000
    return arrays


def _check_names(features):
    names = list(features)
    if not names or len(set(names)) != len(names):
        raise ValueError('Имена признаков должны быть непустыми и уникальными')
    if any(not isinstance(name, str) or not name or any(c in name for c in '\t\r\n') for name in names):
        raise ValueError('Имена признаков должны быть строками без tab/newline')
    return names


def write_numeric_dataset(matrix_path, targets, features, output_path, split, progress=None,
                          batch_rows=BATCH_ROWS):
    """Потоково сохранить float32 без потери точности и без широкой копии в RAM."""
    names = _check_names(features)
    matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
    rows = len(targets['label'])
    if matrix.dtype != np.dtype('float32') or matrix.shape != (rows, len(names)):
        raise ValueError('Нужна float32-матрица той же формы, что targets и features')
    if batch_rows < 1:
        raise ValueError('batch_rows должен быть положительным')
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    schema = pa.schema([('__target', pa.int8()), ('__group', pa.int64())] +
                       [(name, pa.float32()) for name in names])
    options = csv.WriteOptions(include_header=False, delimiter='\t', quoting_style='none')
    with csv.CSVWriter(output_path, schema, write_options=options) as writer:
        for left in range(0, rows, batch_rows):
            right = min(rows, left + batch_rows)
            columns = [pa.array(targets['label'][left:right], type=pa.int8()),
                       pa.array(targets['group_id'][left:right], type=pa.int64())]
            for index in range(len(names)):
                values = matrix[left:right, index]
                if np.isinf(values).any():
                    raise ValueError(f'Бесконечный признак: {names[index]}')
                columns.append(pa.array(values, type=pa.float32()))
            writer.write_table(pa.Table.from_arrays(columns, schema=schema))
            del columns, values
            # Прочитанные страницы остаются на диске и не накапливаются в RSS.
            matrix._mmap.madvise(mmap.MADV_DONTNEED)
            _emit(progress, 'export_dsv', split, right, rows)
    del matrix
    return output_path


def _quantize_file(data_path, columns_path, input_borders=None, threads=THREADS):
    """Вызвать нативный reader ровно один раз в режиме квантования.

    В CatBoost 1.2.10 utils.quantize сначала создает полный raw Pool(file),
    затем повторно вызывает _read с quantization_params. Здесь начальный Pool
    содержит одну ячейку: единственный большой read сразу квантует данные.
    Это приватный API, поэтому версия и сигнатура проверяются явно.
    """
    if catboost.__version__ != SUPPORTED_CATBOOST:
        raise RuntimeError('Для другой версии CatBoost повторите тест одношагового reader')
    if 'quantization_params' not in inspect.signature(Pool._read).parameters:
        raise RuntimeError('Сигнатура CatBoost Pool._read изменилась')
    params = {'border_count': 254, 'random_seed': 42, 'nan_mode': 'Min',
              'used_ram_limit': '3GB', 'task_type': 'CPU', 'dev_block_size': BATCH_ROWS}
    if input_borders is not None:
        params['input_borders'] = str(input_borders)
    pool = Pool(np.zeros((1, 1), dtype='float32'), thread_count=1)
    pool._read(str(data_path), str(columns_path), None, None, None,
               '\t', False, False, threads, params)
    if not pool.is_quantized():
        raise AssertionError('Нативный reader вернул неквантованный Pool')
    return pool


def _check_pool(pool, targets, features):
    """Проверить имена, метки и те же границы ranking-групп после чтения файла."""
    if pool.num_row() != len(targets['label']) or pool.get_feature_names() != list(features):
        raise AssertionError('Pool изменил число строк или порядок признаков')
    np.testing.assert_array_equal(np.asarray(pool.get_label(), dtype='float64'), targets['label'])
    hashes = pool.get_group_id_hash()
    groups = targets['group_id']
    np.testing.assert_array_equal(hashes[1:] != hashes[:-1], groups[1:] != groups[:-1])
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    if len(np.unique(hashes[starts])) != len(starts):
        raise AssertionError('Разные группы получили одинаковый идентификатор')


def _prepare_request(matrix_path, targets_path, names, input_borders):
    """Описать входы, чтобы возобновление не смешивало разные расчеты."""
    matrix_path, targets_path = Path(matrix_path).resolve(), Path(targets_path).resolve()
    state = matrix_path.stat()
    return {'matrix_path': str(matrix_path), 'matrix_bytes': state.st_size,
            'matrix_mtime_ns': state.st_mtime_ns, 'source_targets_path': str(targets_path),
            'source_targets_sha256': _sha256(targets_path), 'features': names,
            'catboost': catboost.__version__, 'input_borders':
            None if input_borders is None else str(Path(input_borders).resolve()),
            'input_borders_sha256': None if input_borders is None else _sha256(input_borders)}


def _recover_quantization(paths, request, input_borders):
    """Переиспользовать проверенный результат или очистить свой незавершенный этап."""
    request_path = paths['request']
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Возобновление получило другие входы квантования')
        if paths['report'].exists():
            report = json.loads(paths['report'].read_text())
            for key, artifact in [('pool', paths['pool']), ('targets', paths['targets']),
                                  ('borders', paths['borders'])]:
                if not artifact.is_file() or _sha256(artifact) != report[f'{key}_sha256']:
                    raise ValueError(f'Поврежден завершенный артефакт: {artifact}')
            paths['dsv'].unlink(missing_ok=True)
            return report
        # Этот request был записан до создания файлов. Чужие пути не удаляются.
        own = [paths[name] for name in ('pool', 'targets', 'dsv', 'columns', 'applied_borders')]
        own += [paths['pool'].with_suffix('.qbin.partial'), Path(str(paths['report']) + '.tmp')]
        if input_borders is None:
            own.append(paths['borders'])
        root = request_path.parent.resolve()
        for artifact in own:
            if artifact.parent.resolve() != root:
                raise ValueError('Временный артефакт оказался вне каталога этапа')
            artifact.unlink(missing_ok=True)
    else:
        own = [paths[name] for name in ('pool', 'targets', 'dsv', 'columns', 'report', 'applied_borders')]
        if input_borders is None:
            own.append(paths['borders'])
        if any(artifact.exists() for artifact in own):
            raise FileExistsError('Есть артефакты без собственного request: автоматическое удаление запрещено')
        _write_json(request_path, request)
    return None


def prepare_quantized_split(matrix_path, targets_path, features, output_dir, split,
                            input_borders=None, progress=None):
    """Подготовить один qbin; validation применяет только готовые train-пороги.

    Временный DSV удаляется только после сохранения проверенного qbin и отчета.
    Train и validation нужно вызывать последовательно в разных процессах.
    """
    if split not in {'train', 'val'}:
        raise ValueError('Ожидается train или val')
    if (split == 'val') != (input_borders is not None):
        raise ValueError('Только validation должна получать input_borders из train')
    if input_borders is not None and not Path(input_borders).is_file():
        raise FileNotFoundError(input_borders)
    started = time.perf_counter()
    names = _check_names(features)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / filename for name, filename in {
        'pool': f'{split}.qbin', 'targets': f'{split}_targets.npz', 'dsv': f'{split}.tsv',
        'columns': f'{split}.columns.tsv', 'report': f'{split}_quantization.json',
        'request': f'{split}_quantization_request.json',
        'applied_borders': f'{split}_applied_borders.tsv'}.items()}
    paths['borders'] = output_dir / 'train_borders.tsv' if split == 'train' else Path(input_borders)
    request = _prepare_request(matrix_path, targets_path, names, input_borders)
    previous = _recover_quantization(paths, request, input_borders)
    if previous is not None:
        return previous
    targets = load_targets(targets_path)
    paths['columns'].write_text('0\tLabel\n1\tGroupId\n' +
                               ''.join(f'{index + 2}\tNum\t{name}\n' for index, name in enumerate(names)))
    write_numeric_dataset(matrix_path, targets, names, paths['dsv'], split, progress)
    dsv_bytes = paths['dsv'].stat().st_size
    _emit(progress, 'quantize_start', split, 0, len(targets['label']))
    quantize_started = time.perf_counter()
    pool = _quantize_file(paths['dsv'], paths['columns'], input_borders)
    quantize_seconds = time.perf_counter() - quantize_started
    _check_pool(pool, targets, names)
    if split == 'train':
        pool.save_quantization_borders(str(paths['borders']))
    else:
        pool.save_quantization_borders(str(paths['applied_borders']))
        if _sha256(paths['applied_borders']) != _sha256(paths['borders']):
            raise AssertionError('Validation Pool изменил train-пороги')
    partial = paths['pool'].with_suffix('.qbin.partial')
    pool.save(str(partial))
    partial.replace(paths['pool'])
    del pool
    gc.collect()
    np.savez_compressed(paths['targets'], **targets)
    report = {'completed': True, 'split': split, 'pool_path': str(paths['pool']),
              'borders_path': str(paths['borders']), 'targets_path': str(paths['targets']),
              'applied_borders_verified': True,
              'source_targets_path': str(Path(targets_path)),
              'source_targets_sha256': request['source_targets_sha256'],
              'matrix_path': str(Path(matrix_path)), 'features': names,
              'rows': len(targets['label']), 'groups': int(len(np.unique(targets['group_id']))),
              'pool_bytes': paths['pool'].stat().st_size, 'temporary_dsv_bytes': dsv_bytes,
              'pool_sha256': _sha256(paths['pool']), 'borders_sha256': _sha256(paths['borders']),
              'targets_sha256': _sha256(paths['targets']), 'quantize_seconds': quantize_seconds,
              'elapsed_seconds': time.perf_counter() - started, 'test_used': False,
              'quantization': {'border_count': 254, 'nan_mode': 'Min', 'random_seed': 42,
                               'borders_origin': 'full_train', 'catboost': catboost.__version__,
                               'reader': 'single Pool._read with quantization_params',
                               'reader_block_rows': BATCH_ROWS, 'input_borders':
                               None if input_borders is None else str(input_borders)}}
    _write_json(paths['report'], report)
    paths['dsv'].unlink()
    _emit(progress, 'quantize_done', split, report['rows'], report['rows'])
    return report


def _pool_state(path):
    path = Path(path).resolve()
    state = path.stat()
    return {'path': str(path), 'bytes': state.st_size, 'mtime_ns': state.st_mtime_ns}


def _recover_fit(output_dir, request):
    """Повторить только собственное незавершенное обучение с теми же входами."""
    request_path = output_dir / 'fit_request.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Возобновление обучения получило другие параметры или данные')
        fit_path = output_dir / 'fit.json'
        if fit_path.exists():
            result = json.loads(fit_path.read_text())
            if _sha256(output_dir / 'model.cbm') != result['model_sha256']:
                raise ValueError('Хеш завершенного ранкера изменился')
            return result
        own_names = set(ARTIFACT_NAMES.values()) | {'requested_params.json'}
        for name in own_names:
            path = output_dir / name
            if path.parent.resolve() != output_dir.resolve():
                raise ValueError('Артефакт обучения вне собственного каталога')
            path.unlink(missing_ok=True)
            Path(str(path) + '.tmp').unlink(missing_ok=True)
    else:
        own_names = set(ARTIFACT_NAMES.values()) | {'requested_params.json', 'fit.json'}
        if any((output_dir / name).exists() for name in own_names):
            raise FileExistsError('Есть артефакты обучения без собственного request')
        _write_json(request_path, request)
    return None


def fit_ranker(train_qbin, val_qbin, val_targets_path, feature_names, output_dir, parameters):
    """Обучить один ранкер с фиксированными параметрами и сохранить оценку validation."""
    names = _check_names(feature_names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = dict(parameters)
    request = {'parameters': parameters, 'features': names,
               'train_pool': _pool_state(train_qbin), 'val_pool': _pool_state(val_qbin),
               'val_targets_path': str(Path(val_targets_path).resolve()),
               'val_targets_sha256': _sha256(val_targets_path)}
    previous = _recover_fit(output_dir, request)
    if previous is not None:
        return previous
    threads = parameters.get('thread_count', THREADS)
    train = Pool('quantized://' + str(train_qbin), thread_count=threads)
    val = Pool('quantized://' + str(val_qbin), thread_count=threads)
    if train.get_feature_names() != names or val.get_feature_names() != names:
        raise AssertionError('Признаки Pool отличаются от зафиксированного списка')
    targets = load_targets(val_targets_path)
    _check_pool(val, targets, names)
    _write_json(output_dir / 'requested_params.json', parameters)
    model = CatBoostRanker(**parameters)
    started = time.perf_counter()
    model.fit(train, eval_set=val)
    fit_seconds = time.perf_counter() - started
    del train
    gc.collect()
    evaluation = evaluate_and_save(model, 'ranker', val, targets, output_dir)
    result = {'completed': True, 'fit_seconds': fit_seconds, 'evaluation': evaluation,
              'train_pool_path': str(train_qbin), 'val_pool_path': str(val_qbin),
              'val_targets_sha256': request['val_targets_sha256'],
              'model_sha256': _sha256(output_dir / 'model.cbm'), 'test_used': False}
    _write_json(output_dir / 'fit.json', result)
    return result
