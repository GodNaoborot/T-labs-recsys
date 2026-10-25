"""Батчевая native-квантизация по столбцам и объединение CatBoost qbin v1.

Каждый столбец квантует сам CatBoost. Родные FlatBuffer chunks копируются без
изменений; Python объединяет только protobuf-метаданные и индекс смещений.
Формат сверяется с catboost/private/libs/quantized_pool/serialization.cpp.
"""
from pathlib import Path
from replay_paths import saved_path
import struct
import gc
import ctypes
import json
import mmap
import time

import catboost
import numpy as np
from catboost import Pool
from temporal_full_training import (load_targets, _check_names, _check_pool,
                                    _prepare_request, _write_json, _sha256)

SUPPORTED_CATBOOST = '1.2.10'
MAGIC_END = b'CatboostQuantizedPoolEnd\0'


def encode_varint(value):
    """Записать неотрицательное целое в компактном формате protobuf."""
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def decode_varint(data, position):
    """Прочитать одно целое protobuf и вернуть следующую позицию."""
    result, shift = 0, 0
    while True:
        value = data[position]
        position += 1
        result |= (value & 127) << shift
        if value < 128:
            return result, position
        shift += 7


def protobuf_fields(data):
    """Разобрать числовые и строковые поля protobuf без стороннего генератора.

    Эти четыре wire-типа покрывают metadata и quantization schema qbin v1.
    """
    position, fields = 0, []
    while position < len(data):
        tag, position = decode_varint(data, position)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, position = decode_varint(data, position)
        elif wire == 2:
            size, position = decode_varint(data, position)
            value = data[position:position + size]
            position += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[position:position + size]
            position += size
        else:
            raise ValueError(f'Неподдержанный protobuf wire type: {wire}')
        fields.append((number, wire, value))
    return fields


def protobuf_field(number, wire, value):
    """Собрать одно поле protobuf, сохранив его исходный тип."""
    head = encode_varint(number * 8 + wire)
    if wire == 0:
        return head + encode_varint(value)
    if wire == 2:
        return head + encode_varint(len(value)) + value
    return head + value


def read_index(path):
    """Прочитать небольшие metadata/schema и смещения native chunks.

    Сами значения признаков остаются в файле; проверяются версия и границы секций.
    """
    with open(path, 'rb') as stream:
        stream.seek(-len(MAGIC_END) - 32, 2)
        offsets = struct.unpack('<QQQQ', stream.read(32))
        if stream.read(len(MAGIC_END)) != MAGIC_END:
            raise ValueError('Поврежден epilog qbin')
        stream.seek(0)
        header = stream.read(offsets[0])
        magic = b'CatboostQuantizedPool\0'
        if not header.startswith(magic) or len(header) != 48:
            raise ValueError('Неизвестный заголовок qbin')
        if struct.unpack_from('<II', header, len(magic)) != (1, 0x62baf5a0):
            raise ValueError('Неизвестная версия бинарного qbin')
        size = Path(path).stat().st_size
        if not (48 <= offsets[1] < offsets[2] < offsets[3] < size - 57):
            raise ValueError('Неверные смещения секций qbin')
        messages = []
        for position in offsets[1:3]:
            stream.seek(position)
            message_size = struct.unpack('<I', stream.read(4))[0]
            if message_size > size - position - 4:
                raise ValueError('Неверная длина protobuf qbin')
            messages.append(stream.read(message_size))
        stream.seek(offsets[3])
        columns = {}
        count = struct.unpack('<I', stream.read(4))[0]
        for _ in range(count):
            column, chunks = struct.unpack('<II', stream.read(8))
            if column in columns or chunks > 1_000_000:
                raise ValueError('Неверный индекс chunks')
            columns[column] = [struct.unpack('<IQII', stream.read(20)) for _ in range(chunks)]
            for chunk_size, offset, document_offset, documents in columns[column]:
                if offset < 48 or offset + chunk_size > offsets[1]:
                    raise ValueError('Chunk вне своей секции qbin')
                if document_offset + documents > 2**32 - 1:
                    raise ValueError('Слишком много строк в qbin')
    return header, messages[0], messages[1], columns


def schema_column(schema, old, new):
    """Взять native-схему одной колонки и заменить ее индекс в общей таблице.

    Пороги и режим обработки NaN копируются побайтно, без нового подбора.
    """
    for number, wire, value in protobuf_fields(schema):
        if number != 1 or wire != 2:
            raise ValueError('Неподдержанная схема qbin')
        fields = protobuf_fields(value)
        key = next(value for number, wire, value in fields if number == 1)
        if key == old:
            remapped = b''.join(protobuf_field(number, wire, new if number == 1 else value)
                                for number, wire, value in fields)
            return protobuf_field(1, 2, remapped)
    raise ValueError('В native-схеме не найден столбец')


def build_pool_metadata(rows, names):
    """Описать float-признаки, GroupId и Target в protobuf pool metadata.

    Номера полей и enum-типы соответствуют проверенному CatBoost qbin v1.
    """
    count = len(names) + 2
    result = bytearray()
    column_types = [(index, 1) for index in range(len(names))]
    column_types += [(len(names), 8), (len(names) + 1, 2)]
    for column, kind in column_types:
        entry = protobuf_field(1, 0, column) + protobuf_field(2, 0, kind)
        result.extend(protobuf_field(1, 2, entry))
    result.extend(protobuf_field(2, 0, rows))
    # Native metadata резервирует три индекса для строковых идентификаторов.
    for field, index in [(4, count), (5, count + 1), (6, count + 2)]:
        result.extend(protobuf_field(field, 0, index))
    for column, name in enumerate(list(names) + ['GroupId', 'Target']):
        entry = protobuf_field(1, 0, column) + protobuf_field(2, 2, name.encode())
        result.extend(protobuf_field(7, 2, entry))
    return bytes(result)


def _release(matrix=None):
    gc.collect()
    if matrix is not None and hasattr(matrix._mmap, 'madvise'):
        matrix._mmap.madvise(mmap.MADV_DONTNEED)
    allocator = ctypes.CDLL(None)
    if hasattr(allocator, 'malloc_trim'):
        allocator.malloc_trim(0)


def _border_columns(path, count):
    """Разложить готовые train-пороги по исходным индексам признаков."""
    columns = {index: [] for index in range(count)}
    if path is None:
        return columns
    for line in Path(path).read_text().splitlines():
        parts = line.split('\t')
        if len(parts) not in (2, 3) or int(parts[0]) not in columns:
            raise ValueError('Некорректный файл train-порогов')
        index = int(parts[0])
        columns[index].append('0\t' + '\t'.join(parts[1:]) + '\n')
    return columns


def _selected_border_text(path):
    return ''.join(line + '\n' for line in Path(path).read_text().splitlines()
                   if line.split('\t')[0] == '0')


def _column_paths(folder, index):
    prefix = folder / f'{index:05d}'
    return {name: Path(str(prefix) + ending) for name, ending in
            [('pool', '.qbin'), ('borders', '.borders.tsv'), ('report', '.json')]}


def _quantize_column(matrix, index, name, targets, paths, input_lines):
    """Одновременно в native Pool находится только один столбец и его bins."""
    rows = len(matrix)
    # Без train-порогов признак был константой и игнорируется моделью. На val
    # сохраняем эту структуру, даже если появились новые значения признака.
    train_constant = input_lines is not None and not input_lines
    values = np.zeros((rows, 1), dtype='float32') if train_constant else matrix[:, index:index+1]
    if np.isinf(values).any():
        raise ValueError(f'Бесконечные значения: {name}')
    arguments = {'feature_names': [name], 'thread_count': 1}
    if index == 0:
        arguments.update(label=targets['label'], group_id=targets['group_id'])
    parameters = {'border_count': 254, 'random_seed': 42, 'nan_mode': 'Min',
                  'used_ram_limit': '512MB', 'thread_count': 1, 'dev_efb_max_buckets': 0}
    local_input = paths['borders'].with_suffix('.input.tsv')
    if input_lines:
        local_input.write_text(''.join(input_lines))
        parameters['input_borders'] = str(local_input)
    pool = Pool(values, **arguments)
    started = time.perf_counter()
    try:
        pool._quantize(parameters)
    except catboost.CatBoostError as error:
        if 'All features are either constant or ignored' not in str(error):
            raise
        # Native quantization требует хотя бы один непостоянный столбец.
        # Вспомогательный столбец в итоговый qbin и train-пороги не попадает.
        del pool
        augmented = np.empty((rows, 2), dtype='float32', order='F')
        augmented[:, 0] = values[:, 0]
        augmented[:, 1] = np.arange(rows, dtype='int64') % 2
        arguments['feature_names'] = [name, '__quantization_dummy']
        pool = Pool(augmented, **arguments)
        pool._quantize(parameters)
    partial = Path(str(paths['pool']) + '.partial')
    pool.save(str(partial))
    exported = paths['borders'].with_suffix('.export.tsv')
    pool.save_quantization_borders(str(exported))
    text = _selected_border_text(exported)
    if input_lines is not None and text != ''.join(input_lines):
        raise AssertionError('Validation изменила train-пороги одного столбца')
    paths['borders'].write_text(text)
    partial.replace(paths['pool'])
    del pool, values
    exported.unlink(missing_ok=True)
    local_input.unlink(missing_ok=True)
    _release(matrix)
    record = {'completed': True, 'feature_index': index, 'feature': name,
              'rows': rows, 'pool_sha256': _sha256(paths['pool']),
              'borders_sha256': _sha256(paths['borders']),
              'seconds': time.perf_counter() - started,
              'training_constant_ignored': train_constant}
    _write_json(paths['report'], record)
    return record


def _column_record(matrix, index, name, targets, folder, input_lines):
    """Переиспользовать только полностью сохраненный и проверенный столбец."""
    paths = _column_paths(folder, index)
    if paths['report'].exists():
        record = json.loads(paths['report'].read_text())
        if record['feature'] != name or record['rows'] != len(matrix):
            raise ValueError('Checkpoint столбца относится к другим данным')
        for kind in ['pool', 'borders']:
            if not paths[kind].is_file() or _sha256(paths[kind]) != record[kind + '_sha256']:
                raise ValueError('Поврежден checkpoint столбца')
    else:
        record = _quantize_column(matrix, index, name, targets, paths, input_lines)
    return paths, record


def _copy_chunks(source_path, destination, mapping, source_columns):
    result = {}
    with open(source_path, 'rb') as source:
        for old, new in mapping.items():
            result[new] = []
            for size, offset, document_offset, documents in source_columns[old]:
                destination.write(b'\0' * ((-destination.tell()) % 16))
                new_offset = destination.tell()
                source.seek(offset)
                remaining = size
                while remaining:
                    block = source.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError('Оборванный native chunk')
                    destination.write(block)
                    remaining -= len(block)
                result[new].append((size, new_offset, document_offset, documents))
    return result


def _column_types(message):
    result = {}
    for number, wire, value in protobuf_fields(message):
        if number == 1:
            fields = {n: v for n, w, v in protobuf_fields(value)}
            result[fields[1]] = fields[2]
    return result


def _merge_native_columns(paths, names, rows, output):
    """Потоково скопировать native chunks, меняя только индекс столбцов."""
    schema, columns, header = [], {}, None
    with open(output, 'wb') as destination:
        for index, path in enumerate(paths):
            current_header, metainfo, current_schema, source_columns = read_index(path)
            if header is None:
                header = current_header
                destination.write(header)
            if header != current_header:
                raise ValueError('Столбцы используют разные версии qbin')
            schema.append(schema_column(current_schema, 0, index))
            mapping = {0: index}
            if index == 0:
                kinds = _column_types(metainfo)
                for kind, new in [(8, len(names)), (2, len(names) + 1)]:
                    candidates = [old for old, value in kinds.items() if value == kind]
                    if len(candidates) != 1:
                        raise ValueError('В первом столбце нет targets/groups')
                    mapping[candidates[0]] = new
            columns.update(_copy_chunks(path, destination, mapping, source_columns))
        metainfo_offset = destination.tell()
        message = build_pool_metadata(rows, names)
        destination.write(struct.pack('<I', len(message)))
        destination.write(message)
        schema_offset = destination.tell()
        message = b''.join(schema)
        destination.write(struct.pack('<I', len(message)))
        destination.write(message)
        table_offset = destination.tell()
        destination.write(struct.pack('<I', len(columns)))
        for column, chunks in sorted(columns.items()):
            destination.write(struct.pack('<II', column, len(chunks)))
            for chunk in chunks:
                destination.write(struct.pack('<IQII', *chunk))
        destination.write(struct.pack('<QQQQ', len(header), metainfo_offset,
                                      schema_offset, table_offset))
        destination.write(MAGIC_END)
    read_index(output)


def prepare_quantized_split(matrix_path, targets_path, features, output_dir, split,
                            input_borders=None, progress=None, matrix_sha256=None):
    """Квантовать train/val по одному столбцу; продолжать с последнего checkpoint.

    Применяется только CatBoost 1.2.10 и бинарный quantized-pool format v1.
    Validation использует train-пороги без нового подбора границ.
    """
    if catboost.__version__ != SUPPORTED_CATBOOST:
        raise RuntimeError('Повторите binary-format тесты для другой версии CatBoost')
    if split not in {'train', 'val'} or (split == 'val') != (input_borders is not None):
        raise ValueError('Train строит пороги, val получает input_borders из train')
    names = _check_names(features)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folder = output_dir / f'{split}_columns'
    folder.mkdir(exist_ok=True)
    request = _prepare_request(matrix_path, targets_path, names, input_borders)
    request.update(method='native_column_quantization_v1', matrix_sha256=matrix_sha256)
    request_path = output_dir / f'{split}_column_quantization_request.json'
    report_path = output_dir / f'{split}_quantization.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Входы column-quantization изменились после checkpoint')
    else:
        if report_path.exists() or any(folder.iterdir()):
            raise FileExistsError('Артефакты без собственного request')
        _write_json(request_path, request)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for kind in ['pool', 'targets', 'borders']:
            if _sha256(saved_path(report[kind + '_path'])) != report[kind + '_sha256']:
                raise ValueError('Изменился завершенный quantized-артефакт')
        return report
    started = time.perf_counter()
    targets = load_targets(targets_path)
    matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
    if (matrix.dtype != np.dtype('float32') or matrix.shape != (len(targets['label']), len(names))
            or not matrix.flags.f_contiguous or len(matrix) > 2**32 - 1):
        raise ValueError('Ожидается float32 F-order матрица с canonical строками')
    border_columns = _border_columns(input_borders, len(names))
    column_paths, records = [], []
    for index, name in enumerate(names):
        inputs = None if input_borders is None else border_columns[index]
        paths, record = _column_record(matrix, index, name, targets, folder, inputs)
        column_paths.append(paths['pool'])
        records.append(record)
        if progress:
            progress({'stage': 'quantize_column', 'split': split, 'rows': index + 1,
                      'total_rows': len(names), 'feature_index': index, 'feature': name})
    del matrix
    _release()
    borders_path = output_dir / 'train_borders.tsv' if split == 'train' else Path(input_borders)
    if split == 'train':
        borders_path.write_text(''.join(
            line.replace('0\t', str(index) + '\t', 1)
            for index in range(len(names))
            for line in _column_paths(folder, index)['borders'].read_text().splitlines(keepends=True)))
    pool_path = output_dir / f'{split}.qbin'
    partial = Path(str(pool_path) + '.partial')
    _merge_native_columns(column_paths, names, len(targets['label']), partial)
    pool_path_temporary = Pool('quantized://' + str(partial), thread_count=1)
    _check_pool(pool_path_temporary, targets, names)
    applied = output_dir / f'{split}_applied_borders.tsv'
    pool_path_temporary.save_quantization_borders(str(applied))
    if _sha256(applied) != _sha256(borders_path):
        raise AssertionError('Объединенный qbin изменил train-пороги')
    del pool_path_temporary
    _release()
    partial.replace(pool_path)
    saved_targets = output_dir / f'{split}_targets.npz'
    np.savez_compressed(saved_targets, **targets)
    report = {'completed': True, 'split': split, 'pool_path': str(pool_path),
              'targets_path': str(saved_targets), 'borders_path': str(borders_path),
              'matrix_path': str(Path(matrix_path)), 'matrix_sha256': matrix_sha256,
              'features': names, 'rows': len(targets['label']),
              'groups': int(np.sum(np.r_[True, np.diff(targets['group_id']) != 0])),
              'pool_bytes': pool_path.stat().st_size, 'pool_sha256': _sha256(pool_path),
              'targets_sha256': _sha256(saved_targets), 'borders_sha256': _sha256(borders_path),
              'quantize_seconds': sum(record['seconds'] for record in records),
              'elapsed_seconds': time.perf_counter() - started, 'temporary_dsv_bytes': 0,
              'applied_borders_verified': True, 'test_used': False,
              'column_checkpoints': str(folder),
              'quantization': {'border_count': 254, 'nan_mode': 'Min', 'random_seed': 42,
                               'borders_origin': 'full_train', 'catboost': catboost.__version__,
                               'method': 'native Pool.quantize per column; stream native qbin chunks',
                               'binary_format_version': 1, 'thread_count': 1,
                               'dev_efb_max_buckets': 0, 'native_api': 'Pool._quantize'}}
    _write_json(report_path, report)
    # После готового и проверенного общего qbin столбцы не нужны для resume.
    for path in column_paths:
        path.unlink()
    return report
