"""Границы legacy split, подготовка 42 входов и повтор дневного bootstrap."""
from pathlib import Path
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import prepare_legacy as preparation
import bootstrap_validation as validation
import train_manual_baseline as baseline
from types import SimpleNamespace
import optuna_data
import optuna_report
from run_legacy_bootstrap import bootstrap_daily


class LegacyReplayTests(unittest.TestCase):
    def test_split_and_gap_boundaries(self):
        day, gap = preparation.DAY, preparation.GAP
        start = np.array([225*day-1, 225*day, 270*day-2000, 270*day-2000,
                          270*day, 285*day-2000, 285*day], dtype='int64')
        end = start + np.array([1, 1, 199, 200, 1, 199, 1])
        rows = pd.DataFrame({name: np.ones(len(start), dtype='int64')
                             for name in preparation.SAMPLE_COLUMNS})
        rows['start_ts'], rows['timestamp'] = start, end
        result = preparation.select_samples(rows)
        self.assertEqual(result['start_ts'].tolist(), start[[1,2,4,5,6]].tolist())
        self.assertEqual(result['split'].tolist(), ['train','train','val','val','test'])

    def test_prepare_manual_uses_only_completed_history(self):
        day = preparation.DAY
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'data'
            data.mkdir()
            starts = np.array([224*day+100, 225*day+100, 270*day+100, 285*day+100], dtype='int32')
            frame = pd.DataFrame({
                'uid': np.ones(4, dtype='uint32'), 'item_id': np.ones(4, dtype='uint32'),
                'artist_id': np.ones(4, dtype='int32'), 'timestamp': starts + 20,
                'start_ts': starts, 'group_id': np.int64(1000) + starts // day,
                'label': np.array([2,0,1,2],dtype='int8'),
                'is_organic': np.ones(4,dtype='uint8'),
                'track_length_seconds': np.full(4,60,dtype='uint16'),
                'played_ratio_pct': np.array([100,0,50,100],dtype='uint8')})
            frame.to_parquet(data / 'listens.parquet',index=False)
            pd.DataFrame({'uid':pd.Series([],dtype='uint32'),
                          'item_id':pd.Series([],dtype='uint32'),
                          'timestamp':pd.Series([],dtype='int32'),
                          'event_type':pd.Series([],dtype='category')}).to_parquet(data / 'events.parquet',index=False)
            preparation.prepare_samples(data)
            report = preparation.prepare_features(data)
            values = pd.read_parquet(data / 'features.parquet')
            self.assertEqual(report['feature_count'],42)
            self.assertEqual(len(values.columns),48)
            self.assertEqual(values.loc[0,'ui_lst_amt_7'],1)
            self.assertEqual(values.loc[0,'ui_skip_cnt_7'],0)
            self.assertEqual(values.loc[0,'ui_time_since_last_lst'],day-20)
            self.assertEqual(values.loc[0,'ui_liked_now'],0)
            with self.assertRaises(FileExistsError):
                preparation.prepare_samples(data)
            with self.assertRaises(FileExistsError):
                preparation.prepare_features(data)

    def test_one_run_wires_manual_dfs_and_optuna_sources(self):
        project = Path('/temporary/project')
        with patch.dict(os.environ, {'TLAB_EXPERIMENT_RUN':'replay_demo'}), patch.object(optuna_data,'ROOT',project):
            paths = validation.analysis_paths(project)
            self.assertEqual(paths['base_config'],project/'results/baselines/runs/replay_demo/run_config.json')
            self.assertEqual(paths['dfs_config'],project/'results/experiment-1/runs/replay_demo/dfs/run_config.json')
            self.assertEqual(optuna_data._source_paths('val')['targets'],paths['targets'])
        with patch.dict(os.environ, {'TLAB_EXPERIMENT_RUN':'../escape'}):
            with self.assertRaises(ValueError):
                validation.analysis_paths(project)


    def test_manual_baseline_cli_fits_and_records_both_models(self):
        random = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'features.parquet'
            names = json.loads((preparation.ROOT/'results/baselines/timing-20261020/run_config.json').read_text())['features']
            rows = 48
            frame = pd.DataFrame({
                'uid': np.repeat(np.arange(16),3),
                'item_id': np.tile(np.arange(3),16),
                'start_ts': np.repeat(np.arange(16)*preparation.DAY,3),
                'group_id': np.repeat(np.arange(16),3),
                'split': np.repeat(['train']*12+['val']*4,3),
                'label': np.tile(np.arange(3),16)})
            for name in names:
                frame[name] = random.normal(size=rows).astype('float32')
            frame.to_parquet(source,index=False)
            args = SimpleNamespace(run_name='tiny',features=source)
            with patch.object(baseline,'PROJECT',root), contextlib.redirect_stdout(io.StringIO()):
                baseline.train(args)
            output = root/'results/baselines/runs/tiny'
            config = json.loads((output/'run_config.json').read_text())
            self.assertEqual(config['features'],names)
            self.assertFalse(config['test_used'])
            self.assertEqual(set(config['models']),{'YetiRank','MultiClass'})
            for item in config['models'].values():
                self.assertTrue(Path(item['path']).is_file())
                self.assertEqual(item['mode'],'trained_with_validation')
            metrics = pd.read_csv(output/'validation_metrics.csv')
            self.assertTrue({'catboost YetiRank','catboost MultiClass, E[label]'}.issubset(set(metrics['model'])))


    def test_optuna_fixed_models_are_from_replayed_baselines(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for method in ('manual','dfs'):
                base = (root/'results/baselines/runs/replay' if method == 'manual'
                        else root/'results/experiment-1/runs/replay/dfs')
                base.mkdir(parents=True)
                keys = ['YetiRank','MultiClass'] if method == 'manual' else ['ranker','pointwise']
                config = {'models':{key:{'params':{},'origin':'new_replay'} for key in keys}}
                (base/'run_config.json').write_text(json.dumps(config))
                pd.DataFrame({'model':['YetiRank','MultiClass'],'NDCG@10':[.6,.7],
                              'QueryAUC':[.5,.5]}).to_csv(base/'validation_metrics.csv',index=False)
            with (
                patch.object(optuna_report,'__file__',str(root/'src/optuna_report.py')),
                patch.object(optuna_report,'load_current_artifacts',return_value={}),
                patch.dict(os.environ,{'TLAB_EXPERIMENT_RUN':'replay'}),
            ):
                records = optuna_report._recorded_models()
            self.assertEqual(len(records),4)
            self.assertTrue(all(record['result']['origin']=='new_replay' for record in records.values()))

    def test_published_daily_bootstrap_matches_saved_bounds(self):
        root = Path(__file__).resolve().parents[1]
        source = root/'results/experiment-1/dfs/bootstrap_noninferiority'
        daily = pd.read_csv(source/'daily_ndcg.csv')
        expected = pd.read_csv(source/'summary.csv')
        actual, draws = bootstrap_daily(daily)
        self.assertEqual(set(draws),{'ranker_L3','ranker_L4','pointwise_L3','pointwise_L4'})
        for name in ('base_ndcg','dfs_ndcg','relative_loss','upper_bound','bootstrap_std'):
            np.testing.assert_allclose(actual[name],expected[name],rtol=0,atol=1e-14)
        self.assertEqual(actual['reject_h0'].tolist(),expected['reject_h0'].tolist())


if __name__ == '__main__':
    unittest.main()
