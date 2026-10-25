"""Маленькие синтетические проверки подготовки Optuna без обучения моделей."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import optuna_data as data


class OptunaDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='optuna-data-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'
        for name, value in [('ROOT', self.root), ('CHUNK', 3), ('ROW_CAPS', {'train': 8, 'val': 6})]:
            change = patch.object(data, name, value)
            change.start()
            self.addCleanup(change.stop)
        self.manual_names = ['is_organic', 'track_length_seconds'] + [f'manual_{i}' for i in range(40)]
        self.dfs_names = ['is_organic', 'track_length_seconds'] + [f'dfs_{i}' for i in range(128)]
        self._make_sources()

    def _make_sources(self):
        definitions = {
            'train': ([1002, 1002, 1001, 1001, 2001, 2001, 3001, 3001, 3001, 4001, 4001],
                      [0, 1, 0, 2, 1, 1, 0, 0, 2, 0, 1]),
            'val': ([1005, 1005, 3005, 3005, 3005, 5005, 5005], [0, 2, 1, 1, 1, 0, 1]),
        }
        self.frames, self.canonical = {}, {}
        for offset, (split, (groups, labels)) in enumerate(definitions.items()):
            frame = pd.DataFrame({'group_id': np.array(groups, dtype='int64'),
                                  'label': np.array(labels, dtype='int8')})
            frame['uid'] = (frame.group_id // 1000).astype('uint32')
            frame['item_id'] = np.uint32(10)
            frame['start_ts'] = ((frame.group_id % 1000) * data.DAY).astype('int32')
            frame['is_organic'] = np.uint8(1)
            frame['track_length_seconds'] = np.uint16(100)
            frame['log_id'] = np.arange(len(frame), dtype='int64') * 2 + offset
            frame['split'] = split
            for column, name in enumerate(self.manual_names[2:]):
                frame[name] = (np.arange(len(frame), dtype='float32') + .125) * (column + 1)
            self.frames[split] = frame
            self.canonical[split] = self._mixed_sorted(frame)
        folder = self.root / 'results/baselines/timing-20261020'
        folder.mkdir(parents=True)
        (folder / 'run_config.json').write_text(json.dumps({'features': self.manual_names}))
        dfs_config = self.root / 'results/experiment-1/dfs/run_config.json'
        dfs_config.parent.mkdir(parents=True)
        dfs_config.write_text(json.dumps({'features': self.dfs_names}))
        features = pd.concat(self.frames.values()).sort_values('log_id', kind='stable')
        (self.root / 'data').mkdir()
        features.drop(columns='log_id').to_parquet(self.root / 'data/features.parquet', index=False)
        train = self.canonical['train']
        manual = Pool(train[self.manual_names].to_numpy('float32'), label=train.label,
                      group_id=train.group_id, feature_names=self.manual_names)
        manual.quantize(border_count=254, random_seed=42, nan_mode='Min')
        manual.save_quantization_borders(str(folder / 'train_borders.tsv'))
        for split, frame in self.frames.items():
            self._make_dfs_pool(split, frame)

    def _mixed_sorted(self, frame):
        mixed = frame.groupby('group_id').label.nunique().gt(1)
        return frame.loc[frame.group_id.isin(mixed.index[mixed])].sort_values('group_id', kind='stable')

    def _make_dfs_pool(self, split, frame):
        paths = data._source_paths(split)
        paths['pool'].parent.mkdir(parents=True, exist_ok=True)
        paths['metadata'].parent.mkdir(parents=True)
        columns = ['log_id'] + list(data.ALIGN_COLUMNS)
        frame[columns].rename(columns={'start_ts': 'time'}).to_parquet(paths['metadata'], index=False)
        selected = self.canonical[split]
        values = np.tile(np.arange(130, dtype='float32'), (len(selected), 1))
        values += selected.log_id.to_numpy('float32')[:, None]
        pool = Pool(values, label=selected.label, group_id=selected.group_id, feature_names=self.dfs_names)
        borders = paths['pool'].parent / 'train_borders.tsv'
        pool.quantize(border_count=254, random_seed=42, input_borders=str(borders) if split == 'val' else None)
        if split == 'train':
            pool.save_quantization_borders(str(borders))
        pool.save(str(paths['pool']))
        np.savez_compressed(paths['targets'], label=selected.label.to_numpy(),
                            group_id=selected.group_id.to_numpy(), log_id=selected.log_id.to_numpy())
        config = {'features': self.dfs_names, 'targets_match_baseline': True,
                  'rows_used': len(selected), 'groups_used': selected.group_id.nunique(),
                  'pool_bytes': paths['pool'].stat().st_size, 'pool_sha256': data._sha256(paths['pool'])}
        paths['config'].write_text(json.dumps(config))

    def _selection(self):
        data.prepare_selection(self.run, train_groups=4, val_groups=2, seed=42)

    def test_selection_preserves_whole_groups_and_is_immutable(self):
        self._selection()
        before = (self.run / 'selection/config.json').read_bytes()
        data.prepare_selection(self.run, train_groups=4, val_groups=2, seed=42)
        self.assertEqual(before, (self.run / 'selection/config.json').read_bytes())
        for split in ('train', 'val'):
            with np.load(self.run / f'selection/{split}.npz') as selected:
                self.assertLessEqual(len(selected['positions']), data.ROW_CAPS[split])
                self.assertTrue((np.diff(selected['positions']) > 0).all())
                reference = self.canonical[split].groupby('group_id').size()
                for group, size in zip(*np.unique(selected['group_id'], return_counts=True)):
                    self.assertEqual(size, reference.loc[group])
        with self.assertRaises(ValueError):
            data.prepare_selection(self.run, train_groups=4, val_groups=2, seed=43)
        self.assertEqual(before, (self.run / 'selection/config.json').read_bytes())

    def test_search_pools_share_targets_and_order(self):
        self._selection()
        for split in ('train', 'val'):
            for method in ('manual', 'dfs'):
                data.prepare_pool(self.run, method, split, 'search')
            left = data.pool_info(self.run, 'manual', split, 'search')
            right = data.pool_info(self.run, 'dfs', split, 'search')
            with np.load(left['targets']) as a, np.load(right['targets']) as b:
                for name in data.TARGET_FIELDS:
                    np.testing.assert_array_equal(a[name], b[name])
            a, b = data._read_json(left['config']), data._read_json(right['config'])
            self.assertEqual(a['row_order_sha256'], b['row_order_sha256'])
            self.assertTrue(a['quantization_uses_full_train'])
            self.assertEqual(Pool('quantized://' + str(left['pool'])).get_feature_names(), self.manual_names)

    def test_streaming_mapping_preserves_duplicates_and_feature_values(self):
        self._selection()
        arrays, _ = data._selection(self.run, 'train', 'search')
        metadata, destination = data._manual_metadata('train', arrays)
        matrix = np.lib.format.open_memmap(self.root / 'matrix.npy', mode='w+', dtype='float32',
                                          shape=(len(arrays['label']), 42), fortran_order=True)
        matrix[:] = np.nan
        data._fill_manual_matrix(matrix, self.manual_names, 'train', metadata, destination)
        expected = self.frames['train'].set_index('log_id').loc[arrays['log_id'], self.manual_names].to_numpy('float32')
        np.testing.assert_array_equal(matrix, expected)
        self.assertTrue(self.frames['train'][list(data.ALIGN_COLUMNS)].duplicated().any())

    def test_full_dfs_reuses_qbin_and_full_manual_has_same_order(self):
        self._selection()
        with patch.object(data.psutil, 'virtual_memory', return_value=types.SimpleNamespace(available=8 * 2**30)):
            for split in ('train', 'val'):
                data.prepare_pool(self.run, 'manual', split, 'full')
                actual = data.prepare_pool(self.run, 'dfs', split, 'full')
                self.assertEqual(actual, data._source_paths(split)['pool'])
                manual = data.pool_info(self.run, 'manual', split, 'full')
                dfs = data.pool_info(self.run, 'dfs', split, 'full')
                self.assertEqual(data._read_json(manual['config'])['row_order_sha256'],
                                 data._read_json(dfs['config'])['row_order_sha256'])

    def test_cache_change_is_rejected_without_overwrite(self):
        self._selection()
        data.prepare_pool(self.run, 'dfs', 'train', 'search')
        info = data.pool_info(self.run, 'dfs', 'train', 'search')
        with info['targets'].open('ab') as stream:
            stream.write(b'corruption')
        damaged = info['targets'].read_bytes()
        with self.assertRaises(ValueError):
            data.prepare_pool(self.run, 'dfs', 'train', 'search')
        self.assertEqual(damaged, info['targets'].read_bytes())

    def test_alignment_error_and_memory_guard(self):
        self._selection()
        with patch.object(data.psutil, 'virtual_memory', return_value=types.SimpleNamespace(available=0)):
            with self.assertRaises(MemoryError):
                data.prepare_pool(self.run, 'manual', 'train', 'full')
        path = data._source_paths('train')['metadata']
        table = pq.read_table(path).to_pandas()
        table.loc[0, 'uid'] += 1
        table.to_parquet(path, index=False)
        with self.assertRaises(AssertionError):
            data.prepare_pool(self.run, 'manual', 'train', 'search')

    def test_missing_legacy_borders_use_selected_train_only(self):
        (self.root / 'results/baselines/timing-20261020/train_borders.tsv').unlink()
        self._selection()
        for split in ('train', 'val'):
            data.prepare_pool(self.run, 'manual', split, 'search')
            config = data._read_json(data.pool_info(self.run, 'manual', split, 'search')['config'])
            self.assertFalse(config['quantization_uses_full_train'])
        with self.assertRaises(ValueError):
            data.prepare_pool(self.run, 'manual', 'test', 'search')

    def test_source_qbins_are_hashed_once_per_run(self):
        original = data._sha256
        with patch.object(data, '_sha256', wraps=original) as digest:
            self._selection()
            data.prepare_pool(self.run, 'dfs', 'train', 'search')
            data.prepare_pool(self.run, 'dfs', 'train', 'full')
            for split in ('train', 'val'):
                count = sum(Path(call.args[0]) == data._source_paths(split)['pool'] for call in digest.call_args_list)
                self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
