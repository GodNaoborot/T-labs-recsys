"""Выбор нового обучения в test-плане без моделей, данных и прогнозирования."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_c4_test as runner
import temporal_full_data


class TestTrainingRunSelection(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'src').mkdir()
        (self.root / 'src/source.py').write_text('value = 1\n')
        self.source = self.root / 'src/source.py'
        self.definitions = self.root / 'definitions'
        self.definitions.mkdir()
        self.selection = self.root / 'selection.json'
        runner.write_json(self.selection, {'features': ['tiny fixture']})
        for entity, _ in temporal_full_data.BRANCHES:
            runner.write_json(self.definitions / (entity + '.json'), {'entity': entity})
        self.patches = [
            patch.object(runner, 'ROOT', self.root),
            patch.object(temporal_full_data, 'DEFAULT_SELECTION', self.selection),
            patch.object(temporal_full_data, 'DEFAULT_DEFINITIONS', self.definitions),
        ]
        for item in self.patches:
            item.start()
        self.training_name = 'new_training_01'
        self.training = runner.training_path(self.training_name)
        self.make_training()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def make_training(self):
        """Собрать только маленькие файлы контракта, без настоящего CatBoost."""
        self.training.mkdir(parents=True)
        for filename in ['status.json', 'result.json']:
            runner.write_json(self.training / filename, {'completed': True})
        runner.write_json(self.training / 'protocol.json', {
            'run_name': self.training_name,
            'source_sha256': {'src/source.py': runner.file_hash(self.source)},
        })
        schemas = {'manual49': ['manual_' + str(i) for i in range(9)],
                   'dfs462': ['dfs_' + str(i) for i in range(3)]}
        runner.write_json(self.training / 'training_plan.json', {'schemas': schemas})
        sources_sha = {'selection': runner.file_hash(self.selection)}
        sources_sha.update({entity: runner.file_hash(self.definitions / (entity + '.json'))
                            for entity, _ in temporal_full_data.BRANCHES})
        runner.write_json(self.training / 'features_preparation.json', {'sources_sha256': sources_sha})
        raw_sources = {}
        for name in ['listens', 'events']:
            path = self.root / (name + '.parquet')
            path.write_bytes(b'Only hash verification; not a real parquet file.')
            raw_sources[name] = runner.artifact(path)
        runner.write_json(self.training / 'targets.json', {'sources': raw_sources})
        for variant in runner.VARIANTS:
            folder = self.training / 'models' / variant
            folder.mkdir(parents=True)
            model = folder / 'model.cbm'
            model.write_bytes(('Not a model: ' + variant).encode())
            schema = 'dfs462' if variant == 'D462-t' else 'manual49'
            ignored = list(range(2, 9)) if variant == 'M-t' else []
            runner.write_json(folder / 'control.json', {
                'completed': True, 'model_sha256': runner.file_hash(model),
                'training_plan_sha256': runner.file_hash(self.training / 'training_plan.json'),
                'schema': schema, 'active_features': len(schemas[schema]) - len(ignored),
                'ignored_indices': ignored, 'evaluation': {'tree_count': 4},
            })
        for schema in schemas:
            folder = self.training / 'quantized' / schema
            folder.mkdir(parents=True)
            border = folder / 'train_borders.tsv'
            border.write_text('0\t0.5\n')
            runner.write_json(folder / 'train_quantization.json', {
                'completed': True, 'borders_sha256': runner.file_hash(border),
            })

    def test_new_training_is_frozen_without_canonical_models(self):
        self.assertFalse(runner.training_path(runner.DEFAULT_TRAINING_RUN).exists())
        data, result = runner.run_paths('evaluation_new')
        plan = runner.freeze_plan('evaluation_new', data, result, self.training_name)
        self.assertEqual(plan['training_run'], self.training_name)
        self.assertEqual(plan['training_result_dir'], str(self.training))
        self.assertFalse(data.exists())
        self.assertTrue(plan['test_already_open'])
        self.assertFalse(plan['new_training_allowed'])
        self.assertEqual(len(plan['comparisons']), 3)
        for record in plan['models']:
            self.assertEqual(Path(record['model']['path']).parents[2], self.training)
        self.assertEqual(runner.verify_plan(result), plan)

    def test_preflight_does_not_create_output_or_prepare_test(self):
        data, result = runner.run_paths('check_new')
        before = {str(path.relative_to(self.training)): runner.file_hash(path)
                  for path in self.training.rglob('*') if path.is_file()}
        plan = runner.build_plan('check_new', data, result, self.training_name)
        self.assertFalse(data.exists())
        self.assertFalse(result.exists())
        self.assertEqual(plan['training_run'], self.training_name)
        after = {str(path.relative_to(self.training)): runner.file_hash(path)
                 for path in self.training.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_invalid_missing_or_incomplete_training_fails_before_output(self):
        for name in ['../other', '/tmp/job', '', 'bad/name', 'a' * 65]:
            with self.assertRaises(ValueError):
                runner.training_path(name)
        data, result = runner.run_paths('rejected')
        with self.assertRaises(FileNotFoundError):
            runner.freeze_plan('rejected', data, result, 'missing_training')
        runner.write_json(self.training / 'status.json', {'completed': False})
        with self.assertRaisesRegex(ValueError, 'не завершено'):
            runner.freeze_plan('rejected', data, result, self.training_name)
        self.assertFalse(data.exists())
        self.assertFalse(result.exists())

    def test_training_symlink_cannot_escape_results_folder(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (self.root / 'results/c4/escape').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'за пределы'):
            runner.training_path('escape')

    def test_missing_or_modified_model_rejected_without_test_output(self):
        data, result = runner.run_paths('missing_model')
        model = self.training / 'models/D462-t/model.cbm'
        model.unlink()
        with self.assertRaises(FileNotFoundError):
            runner.freeze_plan('missing_model', data, result, self.training_name)
        model.write_bytes(b'Changed model')
        with self.assertRaisesRegex(ValueError, 'Не совпала готовая модель'):
            runner.freeze_plan('missing_model', data, result, self.training_name)
        self.assertFalse(data.exists())
        self.assertFalse(result.exists())

    def test_frozen_model_path_cannot_switch_to_another_training(self):
        data, result = runner.run_paths('frozen')
        plan = runner.freeze_plan('frozen', data, result, self.training_name)
        changed = copy.deepcopy(plan)
        changed['models'][0]['model']['path'] = str(self.root / 'other/model.cbm')
        runner.write_json(result / 'plan.json', changed)
        (result / 'plan.sha256').write_text(runner.file_hash(result / 'plan.json') + '\n')
        with self.assertRaisesRegex(ValueError, 'другому обучению'):
            runner.verify_plan(result)

    def test_worker_uses_plan_and_rejects_different_requested_training(self):
        data, result = runner.run_paths('worker_new')
        plan = runner.freeze_plan('worker_new', data, result, self.training_name)
        with patch('c4_test_data.prepare_test_targets', return_value={'completed': True}) as targets:
            with self.assertRaisesRegex(ValueError, 'отличается'):
                runner.worker('targets', data, result, 'other_training')
            targets.assert_not_called()
            runner.worker('targets', data, result, self.training_name)
            targets.assert_called_once()
            self.assertEqual(targets.call_args.args[2]['training_run'], self.training_name)
        marker = json.loads((result / 'stage_targets.json').read_text())
        self.assertFalse(marker['fit_executed'])
        self.assertFalse(data.exists())

    def test_changed_source_or_training_reference_is_rejected(self):
        data, result = runner.run_paths('source_check')
        plan = runner.freeze_plan('source_check', data, result, self.training_name)
        changed = copy.deepcopy(plan)
        changed['training_run'] = 'other_training'
        with self.assertRaisesRegex(ValueError, 'другому обучению'):
            runner.verify_training_reference(changed)
        self.source.write_text('value = 2\n')
        with self.assertRaisesRegex(ValueError, 'исходник'):
            runner.verify_plan(result)

    def test_cli_check_only_selects_training_without_supervisor(self):
        args = ['run_c4_test.py', '--run-name', 'preflight', '--training-run',
                self.training_name, '--check-only']
        with patch.object(sys, 'argv', args), patch.object(runner, 'supervisor') as run:
            runner.main()
        run.assert_not_called()
        data, result = runner.run_paths('preflight')
        self.assertFalse(data.exists())
        self.assertFalse(result.exists())


if __name__ == '__main__':
    unittest.main()
