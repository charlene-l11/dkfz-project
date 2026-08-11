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

    def test_reverse_stepwise_can_run_without_forward_stepwise(self):
        value = base_config()
        value["stepwise_analysis"] = False
        value["reverse_stepwise"] = True
        value["wavelength"] = {"min": 500, "max": 995, "selective": None}
        cfg = self.load(value)
        self.assertFalse(cfg["stepwise_analysis"])
        self.assertTrue(cfg["reverse_stepwise"])
        self.assertEqual(cfg["reverse_stop_wavelength"], 500)

    def test_stepwise_rejects_manual_selective_ranges(self):
        value = base_config()
        value["stepwise_analysis"] = True
        value["wavelength"] = {"min": 500, "max": 995, "selective": "500-600"}
        with self.assertRaisesRegex(ConfigurationError, "selective must be null"):
            self.load(value)

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

    def test_l1pixel_source_resolves_to_nested_l1pixel_csv(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/training/a", "b": "/training/b"}
        value["data"]["use_L1pixel_normalized_values"] = True

        cfg = self.load(value)

        self.assertEqual(
            cfg["data"]["patterns"],
            ["_L1pixel/spectrum_fromCSV1_masked_data_L1pixel_0_derivative.csv"],
        )

    def test_l1pixel_switch_must_be_boolean(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/training/a", "b": "/training/b"}
        value["data"]["use_l1pixel"] = "yes"

        with self.assertRaisesRegex(ConfigurationError, "true or false"):
            self.load(value)

    def test_selective_mapping_defines_multiple_test_cases(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/data/a", "b": "/data/b"}
        value["wavelength"] = {
            "min": 500,
            "max": 995,
            "selective": {1: "630, 810", 2: "575; 650"},
        }
        cfg = self.load(value)

        self.assertEqual(cfg["wavelength"]["selective"], {"1": "630, 810", "2": "575; 650"})

    def test_selective_mapping_cannot_be_combined_with_stepwise(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/data/a", "b": "/data/b"}
        value["stepwise_analysis"] = True
        value["wavelength"] = {"min": 500, "max": 995, "selective": {1: "630; 810"}}

        with self.assertRaisesRegex(ConfigurationError, "cannot be combined"):
            self.load(value)

    def test_selective_mapping_rejects_nested_folder_names(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/data/a", "b": "/data/b"}
        value["wavelength"] = {"min": 500, "max": 995, "selective": {"bad/name": "630; 810"}}

        with self.assertRaisesRegex(ConfigurationError, "folder-safe"):
            self.load(value)

    def test_combination_analysis_is_optional_and_normalized(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/data/a", "b": "/data/b"}
        value["wavelength"] = {"min": 500, "max": 995, "selective": None}
        value["combination_analysis"] = {
            "enabled": True,
            "min": 500,
            "max": 700,
            "n_wavelengths": 3,
        }

        cfg = self.load(value)

        self.assertTrue(cfg["combination_analysis"]["enabled"])
        self.assertEqual(cfg["combination_analysis"]["n_wavelengths"], 3)
        self.assertEqual(cfg["combination_analysis"]["fixed_wavelengths"], [])

    def test_combination_analysis_accepts_fixed_wavelengths(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/data/a", "b": "/data/b"}
        value["wavelength"] = {"min": 500, "max": 995, "selective": None}
        value["combination_analysis"] = {
            "enabled": True,
            "min": 500,
            "max": 695,
            "n_wavelengths": 3,
            "fixed_wavelengths": 700,
        }

        cfg = self.load(value)

        self.assertEqual(cfg["combination_analysis"]["fixed_wavelengths"], [700.0])

    def test_combination_analysis_rejects_other_wavelength_modes(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/data/a", "b": "/data/b"}
        value["wavelength"] = {"min": 500, "max": 995, "selective": "500-700"}
        value["combination_analysis"] = {
            "enabled": True,
            "min": 500,
            "max": 700,
            "n_wavelengths": 3,
        }

        with self.assertRaisesRegex(ConfigurationError, "cannot be combined"):
            self.load(value)

    def test_external_testing_accepts_full_experiment_folder_paths(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/training/a", "b": "/training/b"}
        value["external_testing"] = {
            "enabled": True,
            "experiment_folders": {"a": "/external-rat/a-data", "b": "/external-rat/b-data"},
            "labelling_file": {"a": "rat_labels.txt", "b": "rat_labels.txt"},
            "hyperguis": {"a": "_hypergui_rat", "b": "_hypergui_rat"},
        }

        cfg = self.load(value)

        self.assertEqual(cfg["external_testing"]["experiment_folders"]["a"], "/external-rat/a-data")
        self.assertTrue(cfg["external_testing"]["enabled"])

    def test_external_testing_classes_may_differ_from_training(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/training/a", "b": "/training/b"}
        value["external_testing"] = {
            "enabled": True,
            "experiment_folders": {"rat_a": "/external-rat/a-data", "rat_b": "/external-rat/b-data"},
            "labelling_file": {"rat_a": "labels.txt", "rat_b": "labels.txt"},
            "hyperguis": {"rat_a": "_hypergui_1", "rat_b": "_hypergui_1"},
        }

        cfg = self.load(value)

        self.assertEqual(set(cfg["external_testing"]["experiment_folders"]), {"rat_a", "rat_b"})

    def test_external_testing_mappings_must_match_each_other(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/training/a", "b": "/training/b"}
        value["external_testing"] = {
            "enabled": True,
            "experiment_folders": {"rat_a": "/external-rat/a-data", "rat_b": "/external-rat/b-data"},
            "labelling_file": {"rat_a": "labels.txt", "different": "labels.txt"},
            "hyperguis": {"rat_a": "_hypergui_1", "rat_b": "_hypergui_1"},
        }

        with self.assertRaisesRegex(ConfigurationError, "identical keys"):
            self.load(value)

    def test_external_testing_rejects_relative_experiment_folder_paths(self):
        value = base_config()
        value["experiment_folders"] = {"a": "/training/a", "b": "/training/b"}
        value["external_testing"] = {
            "enabled": True,
            "experiment_folders": {"a": "relative/a", "b": "/external-rat/b"},
            "labelling_file": {"a": "labels.txt", "b": "labels.txt"},
            "hyperguis": {"a": "_hypergui_1", "b": "_hypergui_1"},
        }

        with self.assertRaisesRegex(ConfigurationError, "absolute path"):
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
            config_path = scenario / "manifold_settings.yaml"
            config_path.write_text("test", encoding="utf-8")
            layout = scenario_layout(config_path)
            scenario = scenario.resolve()
            self.assertEqual(layout["scenario_root"], scenario)
            self.assertEqual(layout["output"], scenario / "output")
            self.assertEqual(layout["run_config"], scenario / "run_config")
            self.assertEqual(layout["splits"], scenario / "splits")
            self.assertTrue(layout["output"].is_dir())
            self.assertTrue(layout["run_config"].is_dir())
            self.assertTrue(layout["splits"].is_dir())

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
