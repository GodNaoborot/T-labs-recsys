"""Измерить составление 128 исходных DFS-формул без расчёта значений."""
from pathlib import Path
import argparse
import json
import statistics
import time

import featuretools as ft
from cumulative_features import make_definitions


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = [('Users', 'uid'), ('Items', 'item_id'),
            ('UserItem', 'user_item_id'), ('UserArtist', 'user_artist_id')]


def construct():
    output = {}
    for entity, key in BRANCHES:
        features = make_definitions(entity, key,
            agg_primitives=['count', 'mean', 'std', 'min', 'max'],
            trans_primitives=['multiply_numeric'], max_depth=2, max_features=32)
        reference = json.loads((ROOT / f'data/experiment-1/featuretools/train_sample_200_cumulative_{entity}_definitions.json').read_text())
        names = [feature.get_name() for feature in features]
        if [entity + ': ' + name for name in names] != reference['feature_list']:
            raise ValueError('Изменился порядок исходных DFS-формул: ' + entity)
        output[entity] = names
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    timings = []
    for _ in range(6):
        started = time.perf_counter()
        features = construct()
        timings.append(time.perf_counter() - started)
    report = {'generated_formula_count': sum(map(len, features.values())),
              'first_total_seconds': timings[0], 'median_total_seconds': statistics.median(timings[1:]),
              'repeat_seconds': timings[1:], 'features': features, 'values_calculated': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
