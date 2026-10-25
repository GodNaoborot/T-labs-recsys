"""Проверки сопоставления внутренних индексов CatBoost исходным столбцам."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from check_used_temporal_features import used_feature_indices


class UsedFeatureIndicesTest(unittest.TestCase):
    def example(self):
        return {'features_info': {'float_features': [
            {'feature_index': 10, 'flat_feature_index': 2, 'feature_id': 'right'},
            {'feature_index': 3, 'flat_feature_index': 0, 'feature_id': 'left'}]},
            'oblivious_trees': [{'splits': [
                {'split_type': 'FloatFeature', 'float_feature_index': 10},
                {'split_type': 'FloatFeature', 'float_feature_index': 3},
                {'split_type': 'FloatFeature', 'float_feature_index': 10}]}]}

    def test_mapping_preserves_original_column_order(self):
        self.assertEqual(used_feature_indices(self.example(), ['left', 'unused', 'right']), [0, 2])

    def test_name_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            used_feature_indices(self.example(), ['left', 'unused', 'wrong'])

    def test_unsupported_split_is_rejected(self):
        value = self.example()
        value['oblivious_trees'][0]['splits'][0]['split_type'] = 'OneHotFeature'
        with self.assertRaises(NotImplementedError):
            used_feature_indices(value, ['left', 'unused', 'right'])
