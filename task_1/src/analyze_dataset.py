"""Пересчитать сводки и графики Yambda порциями целых пользователей."""
from collections import Counter
from pathlib import Path
import argparse
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prepare_history import RAW_COLUMNS, file_hash


def user_blocks(path, batch_size=250_000):
    """Не разрывать uid: дубликаты и первые события пары считаются точно."""
    carry, previous_uid = None, -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=RAW_COLUMNS):
        frame = batch.to_pandas()
        users = frame.uid.to_numpy()
        if len(users) and (users[0] < previous_uid or np.any(users[1:] < users[:-1])):
            raise ValueError('Для порционного EDA нужен исходный порядок по uid')
        previous_uid = int(users[-1])
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        boundary = np.searchsorted(frame.uid.to_numpy(), frame.uid.iloc[-1], side='left')
        carry = frame.iloc[boundary:].copy()
        if boundary:
            yield frame.iloc[:boundary]
    if carry is not None and len(carry):
        yield carry


def liked_timeline(frame):
    """Соединить все listen с первым like той же пары, включая like без listen."""
    keys = ['uid', 'item_id']
    likes = frame.loc[frame.event_type.eq('like'), keys + ['timestamp']]
    likes = likes.groupby(keys, as_index=False).timestamp.min().rename(columns={'timestamp': 'like_ts'})
    listens = frame.loc[frame.event_type.eq('listen'), keys + ['timestamp']].merge(likes, on=keys)
    listens = listens.rename(columns={'timestamp': 'ts'}).assign(event='listen')
    timeline = pd.concat([listens, likes.assign(ts=likes.like_ts, event='like')], ignore_index=True)
    first = listens.groupby(keys).ts.min().rename('first_ts')
    return timeline.merge(first, on=keys, how='left'), len(likes)


def collect_statistics(path):
    """Один проход: схема, точные дубликаты, частоты и временные гистограммы."""
    unique = {name: set() for name in RAW_COLUMNS}
    missing = Counter()
    ratio_counts = Counter()
    edges = [np.arange(-600, 605, 5), np.arange(-10800, 11100, 300),
             np.arange(-86400, 61 * 86400, 86400)]
    histograms = [{event: np.zeros(len(bins) - 1) for event in ['listen', 'like']} for bins in edges]
    counts = Counter()
    dtypes = None
    first_users = set()
    for frame in user_blocks(path):
        dtypes = dtypes or {name: str(frame[name].dtype) for name in RAW_COLUMNS}
        counts['rows'] += len(frame)
        counts['listen_rows'] += int(frame.event_type.eq('listen').sum())
        counts['duplicated_rows'] += int(frame.duplicated().sum())
        for name in RAW_COLUMNS:
            missing[name] += int(frame[name].isna().sum())
            unique[name].update(frame[name].dropna().unique())
        ratios = frame.loc[frame.event_type.eq('listen'), ['played_ratio_pct', 'is_organic']]
        ratio_counts.update(ratios.value_counts(sort=False).to_dict())
        timeline, pairs = liked_timeline(frame)
        counts['liked_pairs'] += pairs
        for index, bins in enumerate(edges):
            selected = timeline if index < 2 else timeline.loc[timeline.first_ts.notna()]
            anchor = selected.like_ts if index < 2 else selected.first_ts
            offsets = selected.ts.to_numpy('int64') - anchor.to_numpy('int64')
            if index < 2:
                weights = np.ones(len(selected))
            else:
                user_pairs = selected[['uid', 'item_id']].drop_duplicates().groupby('uid').size()
                first_users.update(user_pairs.index)
                weights = 1 / selected.uid.map(user_pairs).to_numpy()
            in_window = (offsets >= bins[0]) & (offsets < bins[-1])
            for event in ['listen', 'like']:
                mask = in_window & selected.event.eq(event).to_numpy()
                histograms[index][event] += np.histogram(offsets[mask], bins=bins, weights=weights[mask])[0]
    if not counts['rows']:
        raise ValueError('Пустой исходный журнал')
    schema = pd.DataFrame([{'column': name, 'dtype': dtypes[name], 'missing': missing[name],
                           'missing_pct': 100 * missing[name] / counts['rows'], 'unique': len(unique[name])}
                          for name in RAW_COLUMNS])
    counts.update(users=len(unique['uid']), items=len(unique['item_id']))
    counts['reaction_rows'] = counts['rows'] - counts['listen_rows']
    ratios = pd.DataFrame([{'played_ratio_pct': key[0], 'is_organic': key[1], 'count': value}
                           for key, value in sorted(ratio_counts.items())])
    denominators = [counts['liked_pairs'], counts['liked_pairs'], len(first_users)]
    for histogram, denominator in zip(histograms, denominators):
        for event in histogram:
            histogram[event] /= max(1, denominator)
    return dict(counts), schema, ratios, edges, histograms


def save_ratio_plots(ratios, output):
    """Повторить четыре графика распределения доли прослушивания."""
    records = []
    settings = [('ratio-all.png', False, False, 1, None, 'Доля прослушивания'),
                ('ratio-organic.png', True, True, 1, None, 'Доля прослушивания по is_organic'),
                ('ratio-recommended-5.png', False, False, 5, 0, 'Рекомендованные прослушивания, шаг 5%'),
                ('ratio-recommended.png', False, False, 1, 0, 'Рекомендованные прослушивания')]
    for index, (filename, hue, logarithm, step, organic, title) in enumerate(settings):
        selected = ratios if organic is None else ratios.loc[ratios.is_organic.eq(organic)]
        fig, ax = plt.subplots(figsize=(10, 4))
        if len(selected):
            bins = np.arange(np.floor(selected.played_ratio_pct.min()) - .5,
                             np.ceil(selected.played_ratio_pct.max()) + step + .5, step)
            groups = selected.groupby('is_organic') if hue else [(None, selected)]
            for group, values in groups:
                weights = values['count'].to_numpy() / values['count'].sum()
                heights, _ = np.histogram(values.played_ratio_pct, bins=bins, weights=weights)
                ax.stairs(heights, bins, fill=not hue, alpha=.7, label=f'is_organic={group}' if hue else None)
        ax.set(xlabel='Доля прослушивания, %', ylabel='Доля событий', title=title)
        if index == 2:
            ax.set_xlim(0, 110)
        if logarithm:
            ax.set_yscale('log')
        if hue and len(selected):
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=120)
        plt.close(fig)
        records.append({'file': filename, 'caption': title, 'source_cell_index': 8 if index == 3 else 7})
    return records


def save_timeline_plots(edges, histograms, output):
    records = []
    settings = [('like-seconds.png', 1, 'с', 'События относительно первого лайка', True),
                ('like-hours.png', 3600, 'ч', 'Прослушивания относительно первого лайка', False),
                ('first-listen-days.png', 86400, 'дни', 'События относительно первого прослушивания', False)]
    for index, (filename, unit, label, title, logarithm) in enumerate(settings):
        fig, ax = plt.subplots(figsize=(10, 4))
        for event in ['listen', 'like']:
            if index < 2 and event == 'like':
                continue
            ax.stairs(histograms[index][event], edges[index] / unit, label=event, fill=event == 'listen', alpha=.7)
        ax.axvline(0, color='red', linestyle='--')
        ax.set(xlabel=f'Смещение, {label}', ylabel='Нормированное число событий', title=title)
        if logarithm and any(histograms[index]['listen'] > 0):
            ax.set_yscale('log')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=120)
        plt.close(fig)
        records.append({'file': filename, 'caption': title, 'source_cell_index': 17})
    return records



def timeline_table(edges, histograms):
    """Сохранить границы бинов и уже нормированные частоты без сырых событий."""
    rows = []
    names = ('like_seconds', 'like_hours', 'first_listen_days')
    for name, bins, histogram in zip(names, edges, histograms):
        for event, values in histogram.items():
            for left, right, value in zip(bins[:-1], bins[1:], values):
                rows.append({'histogram': name, 'event': event, 'bin_left': left,
                             'bin_right': right, 'normalized_value': value})
    return pd.DataFrame(rows)


def analyze(raw_events, output_dir):
    raw_events, output_dir = Path(raw_events).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Для нового EDA нужна отдельная папка: ' + str(output_dir))
    started = time.monotonic()
    counts, schema, ratios, edges, histograms = collect_statistics(raw_events)
    output_dir.mkdir(parents=True)
    schema.to_csv(output_dir / 'schema.csv', index=False)
    ratios.to_csv(output_dir / 'ratio_counts.csv', index=False)
    timeline_table(edges, histograms).to_csv(output_dir / 'timeline_histograms.csv', index=False)
    images = save_ratio_plots(ratios, output_dir) + save_timeline_plots(edges, histograms, output_dir)
    for image in images:
        image['sha256'] = file_hash(output_dir / image['file'])
    report = {**counts, 'completed': True, 'description': 'Сводки и графики пересчитаны из исходного журнала.',
              'source': {'path': str(raw_events), 'sha256': file_hash(raw_events)}, 'images': images,
              'elapsed_seconds': time.monotonic() - started}
    (output_dir / 'snapshot.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-events', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.raw_events, args.output_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
