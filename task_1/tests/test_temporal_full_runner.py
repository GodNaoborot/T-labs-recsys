"""Проверки защиты ресурсов и неизменного протокола полной очереди."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_temporal_full as full


class FullRunnerTests(unittest.TestCase):
    def test_resource_guard_leaves_reserve(self):
        state = {'host_disk_free_gib': 20, 'rss_gib': 4}
        self.assertIsNone(full.resource_reason(state, low_samples=2))
        self.assertEqual(full.resource_reason(state, low_samples=3), 'memory_reserve')
        self.assertEqual(full.resource_reason({**state, 'rss_gib': 9}), 'memory_reserve')
        self.assertIsNone(full.resource_reason({**state, 'rss_gib': 10, 'anonymous_rss_gib': 4}))
        self.assertEqual(full.resource_reason({**state, 'anonymous_rss_gib': 9}), 'memory_reserve')
        self.assertEqual(full.resource_reason({**state, 'host_disk_free_gib': 4}), 'disk_reserve')

    def test_worker_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'source.py'
            source.write_text('first')
            protocol = {'sources': {'source': full.file_record(source)},
                        'inputs': {}, 'baseline': {}, 'versions': {}}
            full.write_json(root / 'protocol.json', protocol)
            full.verify_frozen(root)
            source.write_text('second')
            with self.assertRaisesRegex(ValueError, 'изменился'):
                full.verify_frozen(root)


if __name__ == '__main__':
    unittest.main()
