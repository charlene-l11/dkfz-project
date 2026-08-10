import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from median_pipeline.selective import (
    SelectiveRun,
    _child_config,
    _child_root,
    build_selective_runs,
    run_selective,
)


class SelectiveTests(unittest.TestCase):
    def test_builds_named_runs_in_yaml_order(self):
        cfg = {"wavelength": {"selective": {"1": "630; 810", "second": "575; 650"}}}

        runs = build_selective_runs(cfg)

        self.assertEqual([run.name for run in runs], ["1", "second"])
        self.assertEqual([run.selection for run in runs], ["630; 810", "575; 650"])
        self.assertEqual([run.folder_name for run in runs], ["1", "second"])

    def test_duplicate_selections_can_use_distinct_test_case_folders(self):
        cfg = {"wavelength": {"selective": {"1": "630; 810", "2": "630; 810"}}}

        runs = build_selective_runs(cfg)

        self.assertEqual([run.folder_name for run in runs], ["1", "2"])

    def test_child_config_contains_only_its_selection(self):
        master = {
            "wavelength": {
                "min": 500,
                "max": 995,
                "selective": {"1": "630; 810", "2": "575; 650"},
            }
        }
        run = SelectiveRun("1", "630; 810")

        child = _child_config(master, Path("master.yaml"), run)

        self.assertEqual(child["wavelength"]["selective"], "630; 810")
        self.assertEqual(child["selective_run"]["name"], "1")
        self.assertIsInstance(master["wavelength"]["selective"], dict)

    def test_child_folder_is_directly_below_master_scenario(self):
        root = Path("scenario")
        self.assertEqual(
            _child_root(root, SelectiveRun("case 1", "630; 810")),
            root / "case 1",
        )

    def test_long_wavelength_selection_uses_short_test_case_folder(self):
        selection = "; ".join(str(value) for value in range(500, 1000, 5))

        self.assertGreater(len(selection), 255)
        self.assertEqual(
            _child_root(Path("scenario"), SelectiveRun("1", selection)),
            Path("scenario") / "1",
        )

    @patch("median_pipeline.selective.write_split_workbooks")
    @patch("median_pipeline.selective.prepare_data")
    def test_batch_reuses_discovery_and_split_and_creates_separate_roots(
        self,
        prepare_data,
        write_split_workbooks,
    ):
        manifest = pd.DataFrame({"file_path": ["a"], "label": ["organ"]})
        available = np.linspace(500, 995, 100).tolist()
        prepare_data.return_value = (
            manifest,
            {"organ": 0},
            {"splits": {}},
            {"available_wavelengths": available},
        )
        cfg = {
            "wavelength": {
                "min": 500,
                "max": 995,
                "selective": {"1": "630; 810", "2": "575; 650"},
            }
        }
        roots = []

        def fake_run_child(_config_path, _cfg, **kwargs):
            root = kwargs["scenario_root"]
            roots.append(root)
            self.assertTrue(kwargs["copy_source_yaml"])
            (root / "output" / "data").mkdir(parents=True)
            wavelength_info = kwargs["prepared"][3]
            (root / "output" / "data" / "wavelengths.json").write_text(
                json.dumps(wavelength_info), encoding="utf-8"
            )
            return root

        with tempfile.TemporaryDirectory() as directory:
            master = Path(directory) / "manifold_settings.yaml"
            master.write_text("test", encoding="utf-8")

            result = run_selective(master, cfg, True, fake_run_child)

            resolved_root = master.resolve().parent
            self.assertEqual(result, resolved_root)
            self.assertEqual(
                roots,
                [
                    resolved_root / "1",
                    resolved_root / "2",
                ],
            )
            status = json.loads((resolved_root / "selective results" / "status.json").read_text())
            self.assertEqual([record["status"] for record in status["runs"]], ["prepared", "prepared"])

        prepare_data.assert_called_once()
        self.assertIsNone(prepare_data.call_args.args[0]["wavelength"]["selective"])
        write_split_workbooks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
