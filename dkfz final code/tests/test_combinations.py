import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from median_pipeline.combinations import (
    CombinationRun,
    _completed_folder_name,
    build_combination_runs,
    run_combinations,
)


class CombinationTests(unittest.TestCase):
    def test_builds_every_mathematical_combination(self):
        cfg = {
            "combination_analysis": {
                "enabled": True,
                "min": 500,
                "max": 515,
                "n_wavelengths": 3,
            }
        }
        info = {"available_wavelengths": [500, 505, 510, 515, 520]}

        runs = build_combination_runs(cfg, info)

        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[0].selection_name, "500+505+510")
        self.assertEqual(runs[-1].selection_name, "505+510+515")

    def test_completed_folder_uses_zero_padded_test_accuracy_percentage(self):
        run = CombinationRun((500.0, 505.0, 510.0))

        self.assertEqual(_completed_folder_name(run, 0.9312345678), "093_12345678")
        self.assertEqual(_completed_folder_name(run, 0.0525), "005_25000000")
        self.assertEqual(_completed_folder_name(run, 1.0), "100_00000000")

    def test_fixed_wavelength_is_included_in_every_combination(self):
        cfg = {
            "combination_analysis": {
                "enabled": True,
                "min": 500,
                "max": 515,
                "n_wavelengths": 3,
                "fixed_wavelengths": [700],
            }
        }
        info = {"available_wavelengths": [500, 505, 510, 515, 700]}

        runs = build_combination_runs(cfg, info)

        self.assertEqual(len(runs), 6)
        self.assertEqual(runs[0].selection_name, "700+500+505")
        self.assertEqual(runs[-1].selection_name, "700+510+515")
        self.assertTrue(all(run.wavelengths[0] == 700 for run in runs))

    @patch("median_pipeline.combinations.write_split_workbooks")
    @patch("median_pipeline.combinations.prepare_data")
    def test_runs_rename_by_test_accuracy_and_results_are_sorted(
        self,
        prepare_data,
        write_split_workbooks,
    ):
        manifest = pd.DataFrame({"file_path": ["a"], "label": ["organ"]})
        prepare_data.return_value = (
            manifest,
            {"organ": 0},
            {"splits": {}},
            {"available_wavelengths": [500, 505, 510]},
        )
        cfg = {
            "combination_analysis": {
                "enabled": True,
                "min": 500,
                "max": 510,
                "n_wavelengths": 2,
            },
            "wavelength": {"min": 500, "max": 510, "selective": None},
        }
        accuracies = {
            "500; 505": 0.75,
            "500; 510": 0.90,
            "505; 510": 0.825,
        }
        test_accuracies = {
            "500; 505": 0.625,
            "500; 510": 0.875,
            "505; 510": 0.80,
        }
        child_calls = []

        def fake_run_child(_config_path, child_cfg, **kwargs):
            root = kwargs["scenario_root"]
            child_calls.append(root)
            output = root / "output"
            run_config = root / "run_config"
            output.mkdir(parents=True)
            run_config.mkdir(parents=True)
            selection = child_cfg["wavelength"]["selective"]
            pd.DataFrame([{"accuracy": accuracies[selection]}]).to_csv(
                output / "validation_metrics.csv", index=False
            )
            pd.DataFrame([{"accuracy": test_accuracies[selection]}]).to_csv(
                output / "testing_metrics.csv", index=False
            )
            (run_config / "run_parameters.json").write_text(
                json.dumps({"best_checkpoint": str(root / "output/checkpoints/best.ckpt")}),
                encoding="utf-8",
            )
            return root

        with tempfile.TemporaryDirectory() as directory:
            master = Path(directory) / "manifold_settings.yaml"
            master.write_text("test", encoding="utf-8")

            result = run_combinations(master, cfg, False, fake_run_child)

            self.assertEqual(result, master.resolve().parent)
            runs_root = master.parent / "combination runs"
            self.assertEqual(
                sorted(path.name for path in runs_root.iterdir()),
                [
                    "062_50000000",
                    "080_00000000",
                    "087_50000000",
                ],
            )
            self.assertFalse(any(path.name.startswith("pending; ") for path in runs_root.iterdir()))
            results_root = master.parent / "combination results"
            combined = pd.read_csv(results_root / "combined_results.csv")
            self.assertEqual(combined["validation_accuracy"].tolist(), [0.9, 0.825, 0.75])
            self.assertEqual(combined["rank"].tolist(), [1, 2, 3])
            self.assertTrue((results_root / "combined_results.xlsx").is_file())
            parameters = json.loads(
                (
                    runs_root
                    / "087_50000000"
                    / "run_config/run_parameters.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIn("087_50000000", parameters["best_checkpoint"])
            identity = json.loads(
                (
                    runs_root
                    / "087_50000000"
                    / "run_config/combination_run.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(identity["selection"], "500+510")

            # A resumed master run migrates an older validation-accuracy name
            # to the test-accuracy name, then skips retraining completed runs.
            test_named_root = runs_root / "087_50000000"
            validation_named_root = runs_root / "090_00000000"
            old_path = str(test_named_root.resolve())
            test_named_root.rename(validation_named_root)
            parameters_path = validation_named_root / "run_config/run_parameters.json"
            parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
            parameters["best_checkpoint"] = parameters["best_checkpoint"].replace(
                old_path, str(validation_named_root.resolve())
            )
            parameters_path.write_text(json.dumps(parameters), encoding="utf-8")

            run_combinations(master, cfg, False, fake_run_child)
            self.assertEqual(len(child_calls), 3)
            self.assertTrue(test_named_root.is_dir())
            self.assertFalse(validation_named_root.exists())

        self.assertEqual(prepare_data.call_count, 2)
        self.assertEqual(write_split_workbooks.call_count, 2)


if __name__ == "__main__":
    unittest.main()
