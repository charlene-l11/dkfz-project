import tempfile
import unittest
from pathlib import Path

import pandas as pd

from median_pipeline.evaluation import write_evaluation


class EvaluationTests(unittest.TestCase):
    def test_all_matrix_and_metric_outputs_are_written(self):
        predictions = pd.DataFrame({
            "label_index": [0, 0, 1, 1],
            "prediction_index": [0, 1, 1, 1],
        })
        cfg = {"evaluation": {"formats": ["csv", "png", "pdf"], "dpi": 72}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_evaluation(predictions, ["a", "b"], root, cfg)
            expected = [
                root / "matrices/csv/confusion_matrix_raw.csv",
                root / "matrices/csv/confusion_matrix_normalized.csv",
                root / "matrices/png/confusion_matrix_raw.png",
                root / "matrices/png/confusion_matrix_normalized.png",
                root / "matrices/pdf/confusion_matrix_raw.pdf",
                root / "matrices/pdf/confusion_matrix_normalized.pdf",
                root / "metrics/per_class_metrics.csv",
                root / "metrics/summary.json",
            ]
            self.assertTrue(all(path.exists() for path in expected))
            self.assertEqual(summary["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
