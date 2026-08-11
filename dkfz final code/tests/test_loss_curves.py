import tempfile
import unittest
from pathlib import Path

import pandas as pd

from median_pipeline.loss_curves import write_loss_curves


class LossCurveTests(unittest.TestCase):
    def test_writes_separate_training_and_validation_curves(self):
        metrics = pd.DataFrame({
            "epoch": [0, 0, 1, 1, 2, 2],
            "step": [2, 2, 5, 5, 8, 8],
            "train/ce_loss_epoch": [0.9, None, 0.7, None, 0.5, None],
            "validation/ce_loss": [None, 1.0, None, 0.8, None, 0.6],
        })
        cfg = {"evaluation": {"formats": ["csv", "png", "pdf"], "dpi": 72}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "logs" / "version_0" / "metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics.to_csv(metrics_path, index=False)

            result = write_loss_curves(metrics_path, root, cfg)

            self.assertEqual(result, root / "loss_curves")
            expected = [
                result / "loss_history.csv",
                result / "training_loss_curve.png",
                result / "validation_loss_curve.png",
            ]
            self.assertTrue(all(path.is_file() for path in expected))
            self.assertFalse((result / "training_loss_curve.pdf").exists())
            self.assertFalse((result / "validation_loss_curve.pdf").exists())
            history = pd.read_csv(result / "loss_history.csv")
            self.assertEqual(history["epoch"].tolist(), [1, 2, 3])
            self.assertEqual(history["training_loss"].tolist(), [0.9, 0.7, 0.5])
            self.assertEqual(history["validation_loss"].tolist(), [1.0, 0.8, 0.6])


if __name__ == "__main__":
    unittest.main()
