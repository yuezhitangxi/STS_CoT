import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'buffer'))

from norm_monitor import NormMonitor


class NormMonitorTest(unittest.TestCase):
    def test_scalar_preserves_sign(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.jsonl'
            monitor = NormMonitor(str(path), max_records=10)
            monitor.record_scalar('bank_cosine_mean', -0.25)
            monitor.close()
            with path.with_name('stats_summary.csv').open(newline='') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row['avg_mean_norm']), -0.25)
            self.assertEqual(float(row['avg_mean_abs']), 0.25)


if __name__ == '__main__':
    unittest.main()
