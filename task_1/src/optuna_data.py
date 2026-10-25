"""Небольшие общие выборки и CatBoost Pool для поиска manual/DFS.

Поиск использует целые user-day группы из канонического sorted/mixed Pool.
Признаки уже рассчитаны по полной прошлой истории. Test здесь не поддержан.
Квантование DFS и, при наличии сохраненных порогов, manual использует полный train.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid

from catboost import Pool
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400
CHUNK = 100_000
ROW_CAPS = {'train': 600_000, 'val': 500_000}
TARGET_FIELDS = ('label', 'group_id', 'log_id', 'day')
ALIGN_COLUMNS = ('uid', 'item_id', 'start_ts', 'label', 'group_id',
                 'is_organic', 'track_length_seconds')


def _sha256(path):
    """Получить SHA256 без загрузки файла целиком в память."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    """Прочитать небольшой конфиг."""
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    """Атомарно сохранить служебный JSON в пространстве нового запуска."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_state(path):
    """Связать уже проверенный checksum с текущей версией файла."""
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}


def _artifact(path, sha256=None):
    """Записать checksum один раз; последующие проверки сначала сверяют версию файла."""
    return {'state': _file_state(path), 'sha256': sha256 or _sha256(path)}


def _check_artifact(record, checksum=False):
    """Не перечитывать большой Pool при неизменных size/mtime/ctime/inode."""
    path = Path(record['state']['path'])
    if _file_state(path) != record['state']:
        raise ValueError(f'Артефакт изменился: {path}')
    if checksum and _sha256(path) != record['sha256']:
        raise ValueError(f'Checksum не совпал: {path}')
    return path


@contextmanager
def _preparation_lock(run_dir):
    """Не допускать двух одновременных подготовок в одном новом запуске."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.data_prepare.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield run_dir


def _source_paths(split):
    """Пути канонических train/val: они используются только для чтения."""
    if split not in ('train', 'val'):
        raise ValueError('Допустимы только train и val')
    name = os.environ.get('TLAB_EXPERIMENT_RUN')
    if name is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Некорректный TLAB_EXPERIMENT_RUN')
    data = (ROOT / 'data/experiment-1/featuretools' if name is None else
            ROOT / 'data/experiment-1/runs' / name / 'featuretools')
    cache = data / 'catboost_pools'
    folder = data / f'full_{split}_cumulative'
    return {'pool': cache / f'{split}.qbin', 'targets': cache / f'{split}_targets.npz',
            'config': cache / f'{split}_config.json',
            'metadata': folder / f'{split}_all_cumulative_metadata.parquet'}


def _canonical_source(split):
    """Проверить схему канонического конфига и размеры без чтения большого Pool."""
    paths = _source_paths(split)
    config = _read_json(paths['config'])
    if not config.get('targets_match_baseline') or len(config['features']) != 130:
        raise ValueError('Нужен проверенный канонический DFS Pool из 130 признаков')
    if paths['pool'].stat().st_size != config['pool_bytes']:
        raise ValueError(f'Изменился размер исходного Pool: {split}')
    return {'config_sha256': _sha256(paths['config']),
            'pool': _file_state(paths['pool']), 'targets': _file_state(paths['targets']),
            'pool_sha256': config['pool_sha256'], 'rows': config['rows_used'],
            'groups': config['groups_used'], 'features': config['features']}


def _verify_source_pool(run_dir, split, source):
    """Полностью проверить исходный qbin один раз на его неизменную версию."""
    path = run_dir / 'verified_source_pools.json'
    verified = _read_json(path) if path.exists() else {}
    if verified.get(split) == source:
        return
    if _sha256(_source_paths(split)['pool']) != source['pool_sha256']:
        raise ValueError(f'Checksum исходного DFS Pool не совпал: {split}')
    verified[split] = source
    _write_json(path, verified)


def _load_canonical_targets(split):
    """Прочитать узкие целевые массивы в точном порядке готового Pool."""
    with np.load(_source_paths(split)['targets']) as stored:
        arrays = {name: stored[name] for name in ('label', 'group_id', 'log_id')}
    if any(value.dtype.kind not in 'iu' for value in arrays.values()):
        raise ValueError('Канонические targets должны быть целочисленными')
    if not np.isin(arrays['label'], [0, 1, 2]).all():
        raise ValueError('Некорректные исходные label')
    arrays = {'label': arrays['label'].astype('int8', copy=False),
              'group_id': arrays['group_id'].astype('int64', copy=False),
              'log_id': arrays['log_id'].astype('int64', copy=False)}
    arrays['day'] = arrays['group_id'] % 1000
    _validate_targets(arrays)
    config = _read_json(_source_paths(split)['config'])
    if len(arrays['label']) != config['rows_used']:
        raise ValueError('Число целевых строк не совпало с каноническим Pool')
    return arrays


def _validate_targets(arrays):
    """Проверить длины, порядок целых групп, уникальные события и диапазон меток."""
    size = len(arrays['label'])
    if not size or any(len(arrays[name]) != size for name in TARGET_FIELDS):
        raise ValueError('Нужны непустые целевые массивы одинаковой длины')
    groups, labels = arrays['group_id'], arrays['label']
    if (groups[1:] < groups[:-1]).any() or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('Нарушен порядок групп или диапазон меток')
    if len(np.unique(arrays['log_id'])) != size:
        raise ValueError('log_id должны быть уникальными')
    np.testing.assert_array_equal(arrays['day'], groups % 1000)


def _group_positions(groups, requested, row_cap, seed):
    """Выбрать целые группы без замены и вернуть позиции в прежнем порядке Pool."""
    if requested <= 0:
        raise ValueError('Число групп должно быть положительным')
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(groups)]
    candidates = np.random.default_rng(seed).permutation(len(starts))[:requested]
    chosen, rows = [], 0
    for group in candidates:
        size = int(ends[group] - starts[group])
        if rows + size <= row_cap:
            chosen.append(group)
            rows += size
    if not chosen:
        raise ValueError('В лимит строк не поместилась ни одна целая группа')
    return np.concatenate([np.arange(starts[g], ends[g], dtype='int64') for g in sorted(chosen)])


def _save_targets(path, arrays, positions=None):
    """Сохранить общий формат targets; positions есть только в selection."""
    values = {name: arrays[name] for name in TARGET_FIELDS}
    if positions is not None:
        values['positions'] = positions
    with Path(path).open('wb') as stream:
        np.savez_compressed(stream, **values)


def _target_order_sha256(arrays):
    """Получить общий для manual/DFS fingerprint точных целевых строк и порядка."""
    digest = hashlib.sha256()
    for name in TARGET_FIELDS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.view('uint8'))
    return digest.hexdigest()


def prepare_selection(run_dir: Path, train_groups=12000, val_groups=8000, seed=42) -> None:
    """Зафиксировать одну подвыборку целых групп для всех методов и objective."""
    with _preparation_lock(run_dir) as run_dir:
        folder = run_dir / 'selection'
        folder.mkdir(exist_ok=True)
        request = {'train_groups': train_groups, 'val_groups': val_groups, 'seed': seed,
                   'row_caps': ROW_CAPS, 'sources': {s: _canonical_source(s) for s in ('train', 'val')}}
        path = folder / 'config.json'
        if path.exists():
            saved = _read_json(path)
            if saved['request'] != request:
                raise ValueError('Selection уже зафиксирован с другими параметрами или источниками')
            for record in saved['splits'].values():
                _check_artifact(record['artifact'], checksum=True)
            return
        if any((folder / f'{s}.npz').exists() for s in ('train', 'val')):
            raise FileExistsError('Неполный selection: нужен новый run_dir')
        _build_selection(run_dir, folder, request)


def _build_selection(run_dir, folder, request):
    """Последовательно сохранить train/val selection, не удерживая обе полные части."""
    records = {}
    for offset, split in enumerate(('train', 'val')):
        _verify_source_pool(run_dir, split, request['sources'][split])
        arrays = _load_canonical_targets(split)
        positions = _group_positions(arrays['group_id'], request[f'{split}_groups'],
                                     ROW_CAPS[split], request['seed'] + offset)
        selected = {name: value[positions] for name, value in arrays.items()}
        path = folder / f'{split}.npz'
        _save_targets(path, selected, positions)
        records[split] = {'artifact': _artifact(path), 'rows': len(positions),
                         'groups': len(np.unique(selected['group_id'])),
                         'row_order_sha256': _target_order_sha256(selected)}
    _write_json(folder / 'config.json', {'request': request, 'splits': records})


def _selection(run_dir, split, scope):
    """Вернуть общие targets и позиции search либо все targets для полного fit."""
    config = _read_json(run_dir / 'selection/config.json')
    if config['request']['sources'][split] != _canonical_source(split):
        raise ValueError('Канонический источник изменился после выбора групп')
    if scope == 'full':
        return _load_canonical_targets(split), None
    record = config['splits'][split]
    path = _check_artifact(record['artifact'], checksum=True)
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in TARGET_FIELDS}
        positions = stored['positions']
    _validate_targets(arrays)
    if _target_order_sha256(arrays) != record['row_order_sha256']:
        raise ValueError('Порядок selection изменился')
    return arrays, positions


def _cache_paths(run_dir, method, split, scope):
    """Задать новые cache-пути, не пересекающиеся с историческими артефактами."""
    if method not in ('manual', 'dfs') or scope not in ('search', 'full'):
        raise ValueError('Нужны method=manual/dfs и scope=search/full')
    _source_paths(split)
    folder = run_dir / 'cache' / scope / method
    return {name: folder / f'{split}{suffix}' for name, suffix in
            [('pool', '.qbin'), ('targets', '_targets.npz'), ('config', '_config.json')]}


def _feature_names(method):
    """Сохранить канонические имена и порядок 42 manual либо 130 DFS признаков."""
    suffix = 'baselines/timing-20261020' if method == 'manual' else 'experiment-1/dfs'
    names = _read_json(ROOT / f'results/{suffix}/run_config.json')['features']
    if len(names) != (42 if method == 'manual' else 130) or len(set(names)) != len(names):
        raise ValueError('Неожиданный набор признаков')
    return names


def _manual_border_source(run_dir, split, scope, names):
    """Выбрать train-пороги; val никогда не обучает собственное квантование."""
    local = run_dir / 'cache' / scope / 'manual/train_borders.tsv'
    if split == 'val':
        if not local.exists():
            raise FileNotFoundError('Сначала подготовьте manual train того же scope')
        config = _check_cache(local.with_name('train_config.json'))
        _check_artifact(config['artifacts']['borders'], checksum=True)
        origin = 'canonical_full_train' if config['quantization_uses_full_train'] else 'selected_train'
        return local, origin
    name = os.environ.get('TLAB_EXPERIMENT_RUN')
    legacy = (ROOT / 'results/baselines/timing-20261020/train_borders.tsv' if name is None
              else ROOT / 'results/baselines/runs' / name / 'train_borders.tsv')
    if legacy.exists():
        for line in legacy.read_text().splitlines():
            if line and not 0 <= int(line.split('\t')[0]) < len(names):
                raise ValueError('Сохраненные manual borders не соответствуют 42 признакам')
        return legacy, 'canonical_full_train'
    return None, 'selected_train' if scope == 'search' else 'full_train'


def _request(run_dir, method, split, scope, names):
    """Связать cache с выборкой, порядком признаков и неизменными источниками."""
    source = _canonical_source(split)
    request = {'method': method, 'split': split, 'scope': scope, 'features': names,
               'selection_sha256': _sha256(run_dir / 'selection/config.json'), 'source': source}
    if method == 'manual':
        borders, origin = _manual_border_source(run_dir, split, scope, names)
        request.update(manual_features=_file_state(ROOT / 'data/features.parquet'),
                       manual_metadata=_file_state(_source_paths(split)['metadata']),
                       border_source=_artifact(borders) if borders else None, borders_origin=origin)
    return request


def _check_cache(config_path, request=None):
    """Проверить fingerprints и завершенность cache без повторного хеширования qbin."""
    config = _read_json(config_path)
    if not config.get('completed') or (request is not None and config['request'] != request):
        raise ValueError('Cache относится к другой выборке/источнику либо не завершен')
    for name, record in config['artifacts'].items():
        _check_artifact(record, checksum=name != 'pool')
    return config


def pool_info(run_dir, method, split, scope):
    """Вернуть проверенные пути {'pool': Path, 'targets': Path, 'config': Path}."""
    run_dir = Path(run_dir).resolve()
    paths = _cache_paths(run_dir, method, split, scope)
    config = _check_cache(paths['config'], _request(run_dir, method, split, scope, _feature_names(method)))
    return {'pool': Path(config['artifacts']['pool']['state']['path']),
            'targets': paths['targets'], 'config': paths['config']}


def _manual_metadata(split, arrays):
    """Отобразить канонические log_id в исходные строки узкой DFS metadata."""
    columns = ['log_id'] + ['time' if c == 'start_ts' else c for c in ALIGN_COLUMNS]
    meta = pq.read_table(_source_paths(split)['metadata'], columns=columns)
    source_ids = meta['log_id'].to_numpy()
    if (source_ids[1:] <= source_ids[:-1]).any():
        raise ValueError('Исходная metadata должна иметь строго возрастающие log_id')
    source_positions = np.searchsorted(source_ids, arrays['log_id'])
    if (source_positions >= len(source_ids)).any():
        raise ValueError('Целевой log_id отсутствует в metadata')
    np.testing.assert_array_equal(source_ids[source_positions], arrays['log_id'])
    for name in ('label', 'group_id'):
        np.testing.assert_array_equal(meta[name].to_numpy()[source_positions], arrays[name], err_msg=name)
    np.testing.assert_array_equal(meta['time'].to_numpy()[source_positions] // DAY, arrays['day'])
    destination = np.full(len(meta), -1, dtype='int64')
    destination[source_positions] = np.arange(len(source_positions))
    return meta, destination


def _align_manual_batch(table, metadata, offset):
    """Доказать совпадение физических строк по всем полям старого read_pool_metadata."""
    expected = metadata.slice(offset, len(table))
    if len(expected) != len(table):
        raise ValueError('В manual больше строк, чем в DFS metadata')
    for name in ALIGN_COLUMNS:
        other = 'time' if name == 'start_ts' else name
        np.testing.assert_array_equal(table[name].to_numpy(), expected[other].to_numpy(), err_msg=name)


def _fill_manual_matrix(matrix, names, split, metadata, destination):
    """Потоково выровнять manual и перенести только нужные строки в float32 memmap."""
    columns = list(dict.fromkeys(names + list(ALIGN_COLUMNS) + ['split']))
    offset = 0
    for batch in pq.ParquetFile(ROOT / 'data/features.parquet').iter_batches(batch_size=CHUNK, columns=columns):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.equal(table['split'], split))
        _align_manual_batch(table, metadata, offset)
        mapping = destination[offset:offset + len(table)]
        keep = mapping >= 0
        for column, name in enumerate(names):
            values = table[name].to_numpy().astype('float32')
            if np.isinf(values).any():
                raise ValueError(f'Бесконечные manual признаки: {name}')
            matrix[mapping[keep], column] = values[keep]
        offset += len(table)
    if offset != len(metadata):
        raise ValueError('В manual и DFS metadata разное число строк')
    matrix.flush()


def _build_manual(run_dir, paths, split, scope, names, arrays):
    """Подготовить один manual Pool в отдельном процессе без pandas-копии матрицы."""
    if scope == 'full' and psutil.virtual_memory().available < 4 * 2**30:
        raise MemoryError('Для full manual preparation нужно минимум 4 GiB свободной RAM')
    metadata, destination = _manual_metadata(split, arrays)
    matrix_path = paths['pool'].with_suffix('.float32.npy')
    try:
        matrix = np.lib.format.open_memmap(matrix_path, mode='w+', dtype='float32',
                                          shape=(len(arrays['label']), len(names)), fortran_order=True)
        _fill_manual_matrix(matrix, names, split, metadata, destination)
        del metadata, destination
        _quantize_manual(run_dir, paths['pool'], split, scope, names, arrays, matrix)
        del matrix
    finally:
        matrix_path.unlink(missing_ok=True)


def _quantize_manual(run_dir, output, split, scope, names, arrays, matrix):
    """Применить fulltrain borders либо выучить их только на train; сохранить qbin."""
    borders, _ = _manual_border_source(run_dir, split, scope, names)
    matrix.flags.writeable = False
    pool = Pool(matrix, label=arrays['label'], group_id=arrays['group_id'], feature_names=names, thread_count=4)
    pool.quantize(border_count=254, random_seed=42, nan_mode='Min', used_ram_limit='3GB',
                  input_borders=str(borders) if borders else None)
    if split == 'train':
        local = output.parent / 'train_borders.tsv'
        shutil.copyfile(borders, local) if borders else pool.save_quantization_borders(str(local))
    pool.save(str(output))


def _build_dfs(paths, split, scope, arrays, positions):
    """Для full переиспользовать canonical qbin; для search срезать только целые группы."""
    source = _source_paths(split)['pool']
    if scope == 'full':
        return source
    pool = Pool('quantized://' + str(source), thread_count=4)
    if pool.get_feature_names() != _feature_names('dfs'):
        raise ValueError('Порядок признаков исходного DFS Pool отличается от конфига')
    selected = pool.slice(positions)
    np.testing.assert_array_equal(np.asarray(selected.get_label(), dtype='float64'), arrays['label'])
    selected.save(str(paths['pool']))
    return paths['pool']


def prepare_pool(run_dir, method, split, scope):
    """Подготовить один cache после prepare_selection; готовый cache не перезаписывается."""
    with _preparation_lock(run_dir) as run_dir:
        names = _feature_names(method)
        paths = _cache_paths(run_dir, method, split, scope)
        request = _request(run_dir, method, split, scope, names)
        if paths['config'].exists():
            config = _check_cache(paths['config'], request)
            return Path(config['artifacts']['pool']['state']['path'])
        paths['pool'].parent.mkdir(parents=True, exist_ok=True)
        if any(paths[name].exists() for name in ('pool', 'targets')):
            raise FileExistsError('Неполный cache: нужен новый run_dir')
        arrays, positions = _selection(run_dir, split, scope)
        _verify_source_pool(run_dir, split, request['source'])
        return _build_pool(run_dir, paths, method, split, scope, names, arrays, positions, request)


def _build_pool(run_dir, paths, method, split, scope, names, arrays, positions, request):
    """Сохранить цели и checksum только после успешного построения Pool."""
    if method == 'manual':
        _build_manual(run_dir, paths, split, scope, names, arrays)
        output = paths['pool']
    else:
        output = _build_dfs(paths, split, scope, arrays, positions)
    _save_targets(paths['targets'], arrays)
    digest = request['source']['pool_sha256'] if method == 'dfs' and scope == 'full' else None
    config = {'completed': True, 'request': request, 'rows': len(arrays['label']),
              'groups': len(np.unique(arrays['group_id'])), 'features': names,
              'row_order_sha256': _target_order_sha256(arrays), 'test_used': False,
              'quantization_uses_full_train': method == 'dfs' or scope == 'full' or request.get('borders_origin') == 'canonical_full_train',
              'checksum_policy': 'sha256 at preparation; qbin reuse checks size/mtime/ctime/inode',
              'artifacts': {'pool': _artifact(output, digest), 'targets': _artifact(paths['targets'])}}
    if method == 'manual':
        config['artifacts']['borders'] = _artifact(output.parent / 'train_borders.tsv')
    _write_json(paths['config'], config)
    return output
