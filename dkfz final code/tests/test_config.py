import tempfile
import unittest
from pathlib import Path

import yaml

from median_pipeline.config import ConfigurationError, load_config


def base_config():
    return {
        "experiment": {"name": "test"},
        "paths": {"htc_root": ".", "data_root": ".", "runs_root": "."},
        "data": {"sources": {"a": "a", "b": "b"}},
        "splitting": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
        "training": {"seed": 1, "max_epochs": 1, "batch_size": 2, "learning_rate": 0.001},
        "evaluation": {},
    }


class ConfigTests(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            return load_config(path)

    def test_balanced_oversampling_is_valid(self):
        value = base_config()
        value["training"]["imbalance"] = {"strategy": "balanced_oversampling"}
        self.assertEqual(self.load(value)["training"]["imbalance"]["strategy"], "balanced_oversampling")

    def test_weighting_and_oversampling_cannot_be_combined(self):
        value = base_config()
        value["training"]["imbalance"] = {"strategy": "balanced_oversampling", "method": "inverse"}
        with self.assertRaises(ConfigurationError):
            self.load(value)

    def test_wavelength_selection_outside_range_is_invalid(self):
        value = base_config()
        value["wavelength"] = {"min": 500, "max": 995, "selective": "480; 700-720"}
        with self.assertRaises(ConfigurationError):
            self.load(value)

    def test_invalid_wavelength_syntax_is_configuration_error(self):
        value = base_config()
        value["wavelength"] = {"min": 500, "max": 995, "selective": "500; bad"}
        with self.assertRaises(ConfigurationError):
            self.load(value)
    def test_ratios_must_sum_to_one(self):
        value = base_config()
        value["splitting"]["test_ratio"] = 0.3
        with self.assertRaises(ConfigurationError):
            self.load(value)


if __name__ == "__main__":
    unittest.main()


