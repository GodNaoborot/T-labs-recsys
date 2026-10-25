"""Разобрать сохранённые DFS-формулы и важности без данных и моделей.

Скрипт читает JSON Featuretools как описание дерева операций. Матрицы признаков,
модели и test-данные не открываются; нового отбора признаков здесь нет.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN = Path('task_1/results/c4/c4_packet_20261024_v1')
DEFINITIONS = Path('task_1/data/experiment-1/temporal_runs/temporal_topk_20261023_v2/definitions')
OUTPUT = Path('docs/results/feature-analysis')
ENTITY_LABELS = {
    'Users': 'Пользователь', 'Items': 'Трек',
    'UserItem': 'Пользователь–трек', 'UserArtist': 'Пользователь–исполнитель',
}
KIND_LABELS = {
    'aggregate': 'Один агрегат',
    'event_product_aggregate': 'Агрегат произведения полей события',
    'aggregate_product': 'Произведение двух агрегатов',
    'freshness': 'Давность первого/последнего события',
}
COLUMN_LABELS = {
    'label': 'историческая метка', 'track_length_seconds': 'длина трека',
    'is_organic': 'is_organic', 'end_ts': 'timestamp', 'log_id': 'строка',
}
FEATURE_COLUMNS = [
    'feature_index', 'feature', 'entity', 'entity_label', 'window_days',
    'window_label', 'feature_kind', 'operation_label', 'aggregation_operations',
    'source_columns', 'importance', 'importance_rank', 'explanation',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_importance(path):
    """Сохранить точные числа из уже посчитанного PredictionValuesChange."""
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = [row['Feature Id'] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError('Повторные имена в importance CSV')
    values = {row['Feature Id']: float(row['Importances']) for row in rows}
    if any(not 0 <= value <= 100 for value in values.values()):
        raise ValueError('Ожидались неотрицательные нормированные важности')
    if abs(sum(values.values()) - 100) > 1e-6:
        raise ValueError('Сумма PredictionValuesChange должна быть равна 100')
    return values


def describe_tree(definition, key):
    """Вернуть состав формулы, проходя по её сериализованным зависимостям."""
    node = definition['feature_definitions'][key]
    args = node['arguments']
    if node['type'] == 'IdentityFeature':
        column = args['column_name']
        return {'columns': {column}, 'operations': set(), 'windows': set(),
                'event_product': False, 'text': COLUMN_LABELS[column]}
    children = [describe_tree(definition, child) for child in args['base_features']]
    primitive = definition['primitive_definitions'][args['primitive']]['type']
    columns = set().union(*(child['columns'] for child in children))
    operations = set().union(*(child['operations'] for child in children))
    windows = set().union(*(child['windows'] for child in children))
    event_product = any(child['event_product'] for child in children)
    if primitive == 'MultiplyNumeric':
        event_product |= key.startswith('Interactions: ')
        text = '(' + ' × '.join(child['text'] for child in children) + ')'
    else:
        operations.add(primitive)
        window = args['use_previous']
        if window is not None:
            if window['unit'] != 'd' or window['value'] not in range(1, 31):
                raise ValueError(f'Неожиданное окно: {window}')
            windows.add(int(window['value']))
        text = aggregation_text(primitive, children[0]['text'])
    return {'columns': columns, 'operations': operations, 'windows': windows,
            'event_product': event_product, 'text': text}


def aggregation_text(primitive, value):
    texts = {
        'Count': 'число взаимодействий', 'Mean': f'среднее({value})',
        'Std': f'стандартное отклонение({value})', 'Min': f'минимум({value})',
        'Max': f'максимум({value})',
        'TimeSinceFirst': 'дней с первого доступного взаимодействия',
        'TimeSinceLast': 'дней с последнего доступного взаимодействия',
    }
    if primitive not in texts:
        raise ValueError(f'Операция не поддержана в отчёте: {primitive}')
    return texts[primitive]


def feature_record(index, name, definitions, importance):
    """У одной формулы одно окно, в том числе у произведения агрегатов."""
    entity, formula = name.split('__', 1)
    definition = definitions[entity]
    key = f'{entity}: {formula}'
    if key not in definition['feature_list']:
        raise ValueError(f'Выбранный признак отсутствует в definitions: {name}')
    tree = describe_tree(definition, key)
    if len(tree['windows']) > 1:
        raise ValueError(f'В одной формуле несколько разных окон: {name}')
    window = next(iter(tree['windows']), None)
    operations = tree['operations']
    root_type = definition['feature_definitions'][key]['type']
    if operations & {'TimeSinceFirst', 'TimeSinceLast'}:
        kind = 'freshness'
    elif root_type == 'TransformFeature':
        kind = 'aggregate_product'
    elif tree['event_product']:
        kind = 'event_product_aggregate'
    else:
        kind = 'aggregate'
    window_label = f'{window} дн.' if window else 'Вся доступная история'
    explanation = f"{ENTITY_LABELS[entity]}: {tree['text']}; {window_label.lower().rstrip('.')}" + '.'
    return dict(zip(FEATURE_COLUMNS, [
        index, name, entity, ENTITY_LABELS[entity], window, window_label, kind,
        KIND_LABELS[kind], '|'.join(sorted(operations)),
        '|'.join(sorted(tree['columns'])), importance[name], None, explanation,
    ]))


def write_csv(path, rows, columns):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, field, label_field):
    rows = {}
    for feature in records:
        category = feature[field]
        row = rows.setdefault(category, {field: category, label_field: feature[label_field],
                                        'feature_count': 0, 'importance_sum': 0.0})
        row['feature_count'] += 1
        row['importance_sum'] += feature['importance']
    return list(rows.values())


def build_report(root, importance_path, output, run=RUN):
    source_paths = {
        'generation': root / run / 'generation_dfs_train.json',
        'features_preparation': root / run / 'features_preparation.json',
        'evaluation': root / run / 'models/D462-t/evaluation.json',
        'control': root / run / 'models/D462-t/control.json',
        'importance': importance_path,
        'selection': root / 'task_1/results/experiment-1/temporal_runs/temporal_used_20261023_v1/protocol.json',
    }
    for entity in ENTITY_LABELS:
        source_paths[f'{entity}_definitions'] = root / DEFINITIONS / f'{entity}.json'
    generation = read_json(source_paths['generation'])
    evaluation = read_json(source_paths['evaluation'])
    control = read_json(source_paths['control'])
    names = generation['features']
    if len(names) != 462 or len(set(names)) != 462 or generation['feature_count'] != 462:
        raise ValueError('Ожидался точный уникальный список 462 входов')
    if evaluation['importance_type'] != 'PredictionValuesChange':
        raise ValueError('Изменился тип сохранённой важности')
    if control['variant'] != 'D462-t' or control['active_features'] != 462:
        raise ValueError('Нужны результаты модели D462-t с 462 входами')
    importance = read_importance(importance_path)
    if set(names) != set(importance):
        raise ValueError('Имена признаков и важностей должны совпадать полностью')
    definitions = {entity: read_json(source_paths[f'{entity}_definitions'])
                   for entity in ENTITY_LABELS}
    expected_hashes = read_json(source_paths['features_preparation'])['sources_sha256']
    if (sha256(source_paths['selection']) != expected_hashes['selection']
            or read_json(source_paths['selection'])['features'] != names):
        raise ValueError('Список выбранных формул изменился относительно C4')
    for entity in ENTITY_LABELS:
        if sha256(source_paths[f'{entity}_definitions']) != expected_hashes[entity]:
            raise ValueError(f'Хеш definitions изменился: {entity}')
    records = [feature_record(i, name, definitions, importance) for i, name in enumerate(names)]
    ranked = sorted(records, key=lambda row: (-row['importance'], row['feature_index']))
    for rank, record in enumerate(ranked, 1):
        record['importance_rank'] = rank
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'selected_features.csv', records, FEATURE_COLUMNS)
    write_csv(output / 'top_features.csv', ranked[:20], FEATURE_COLUMNS)
    dimensions = [('entity', 'entity_label'), ('window_days', 'window_label'),
                  ('feature_kind', 'operation_label')]
    for field, label in dimensions:
        rows = summarize(records, field, label)
        if field == 'window_days':
            rows.sort(key=lambda row: row[field] or 0)
        filename = {'entity': 'entity', 'window_days': 'window',
                    'feature_kind': 'operation'}[field] + '_summary.csv'
        write_csv(output / filename, rows, [field, label, 'feature_count', 'importance_sum'])
    operation_counts = Counter(operation for record in records
                               for operation in record['aggregation_operations'].split('|'))
    write_csv(output / 'primitive_counts.csv',
              [{'primitive': name, 'feature_count': count} for name, count in sorted(operation_counts.items())],
              ['primitive', 'feature_count'])
    provenance = {
        'model_name': 'Автоматические DFS (462)', 'internal_variant': 'D462-t',
        'protocol_name': 'Прогноз следующего временного пакета',
        'internal_run': run.name, 'model_sha256': control['model_sha256'],
        'feature_count': len(records), 'importance_type': evaluation['importance_type'],
        'importance_sum': sum(importance.values()),
        'new_training': False, 'new_predictions': False, 'test_data_read': False,
        'selection_repeated': False, 'current_event_context_features': [],
        'history_rule': 'original_timestamp < t; window=[t-window,t)',
        'selection_note': 'Список перенесён из прежнего протокола; нового отбора на C4 не было.',
        'importance_note': 'Сохранённый PredictionValuesChange; не доля NDCG и не причинный вклад.',
        'primitive_counts_note': 'Одна формула может содержать несколько разных агрегатов; суммы не обязаны равняться 462.',
        'source_files': {name: {'path': str(path.relative_to(root)), 'sha256': sha256(path)}
                         for name, path in source_paths.items()},
        'historical_importance_path': str(run / 'models/D462-t/feature_importance.csv'),
        'outputs_sha256': {path.name: sha256(path) for path in sorted(output.glob('*.csv'))},
    }
    (output / 'provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'features': len(records), 'importance_sum': sum(importance.values()),
                      'output': str(output)}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--training-run', help='Новый источник results/c4/<имя>')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / OUTPUT
    run = RUN
    importance = root / OUTPUT / 'prediction_values_change.csv'
    if args.training_run:
        from reproduce import run_name
        run = Path('task_1/results/c4') / run_name(args.training_run)
        if args.output_dir is None:
            parser.error('Для нового обучения нужен отдельный --output-dir')
        importance = root / run / 'models/D462-t/feature_importance.csv'
    build_report(root, importance, output, run=run)


if __name__ == '__main__':
    main()
