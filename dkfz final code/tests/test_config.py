import tempfile
import unittest
from pathlib import Path

import yaml

from median_pipeline.cli import scenario_layout
from median_pipeline.config import ConfigurationError, load_config


def base_config():
    return {
        "paths": {"htc_root": ".", "data_root": "/data-root"},
        "experiment_folders": {"a": "folder-a", "b": "folder-b"},
        "labelling_file": {"a": "_labelling_001.txt", "b": "_labelling_001.txt"},
        "hyperguis": {"a": "_hypergui_1", "b": "_hypergui_1"},
        "data": {},
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

    def test_scenario_layout_comes_from_config_location(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "scenario_1"
            scenario.mkdir(parents=True)
            for folder in ("output", "run_config", "splits"):
                (scenario / folder).mkdir()
            config_path = scenario / "manifold_settings.yaml"
            config_path.write_text("test", encoding="utf-8")
            layout = scenario_layout(config_path)
            scenario = scenario.resolve()
            self.assertEqual(layout["scenario_root"], scenario)
            self.assertEqual(layout["output"], scenario / "output")
            self.assertEqual(layout["run_config"], scenario / "run_config")
            self.assertEqual(layout["splits"], scenario / "splits")

    def test_input_mapping_names_must_match(self):
        value = base_config()
        value["hyperguis"] = {"a": "_hypergui_1", "different": "_hypergui_1"}
        with self.assertRaises(ConfigurationError):
            self.load(value)

    def test_relative_experiment_folders_use_data_root(self):
        cfg = self.load(base_config())
        self.assertEqual(cfg["experiment_folders"]["a"], "/data-root/folder-a")


if __name__ == "__main__":
    unittest.main()


