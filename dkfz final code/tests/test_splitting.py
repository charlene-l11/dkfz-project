import unittest

import pandas as pd
from openpyxl import load_workbook

from median_pipeline.excel_outputs import write_split_workbooks
from median_pipeline.preparation import discover, split_manifest


class SplittingTests(unittest.TestCase):
    def test_discovery_uses_configured_hypergui_and_labelling_patterns(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_folders = {}
            for label, subject in (("1_stom_pig", "P001"), ("2_liv_pig", "P002")):
                organ_root = root / label
                sample_dir = organ_root / f"{subject}_Experiment1" / "2026_01_01_00_00_00"
                hypergui_dir = sample_dir / "_hypergui_1"
                hypergui_dir.mkdir(parents=True)
                (sample_dir / "_labelling_001.txt").write_text("configured", encoding="utf-8")
                spectrum = hypergui_dir / "spectrum.csv"
                spectrum.write_text(
                    "\n".join(f"{500 + index * 5},{index / 100}" for index in range(100)),
                    encoding="utf-8",
                )
                experiment_folders[label] = str(organ_root)

            cfg = {
                "experiment_folders": experiment_folders,
                "labelling_file": {
                    "1_stom_pig": "_labelling_*.txt",
                    "2_liv_pig": "_labelling_*.txt",
                },
                "hyperguis": {
                    "1_stom_pig": "_hypergui_*",
                    "2_liv_pig": "_hypergui_*",
                },
                "data": {
                    "patterns": ["spectrum.csv"],
                    "expected_channels": 100,
                    "annotation_name": "csv_spectrum",
                },
                "wavelength": {"min": 500, "max": 995, "selective": None},
            }
            frame, mapping, _ = discover(cfg)
            self.assertEqual(mapping, {"1_stom_pig": 0, "2_liv_pig": 1})
            self.assertEqual(len(frame), 2)
            self.assertTrue(frame["hypergui_dir"].str.endswith("_hypergui_1").all())
            self.assertTrue(frame["labelling_file"].str.endswith("_labelling_001.txt").all())

    def test_subjects_do_not_cross_splits_and_classes_are_covered(self):
        rows = []
        for subject in [f"P{i:03d}" for i in range(1, 9)]:
            for label in ("kidney", "liver"):
                rows.append({"subject_name": subject, "label": label})
        frame = pd.DataFrame(rows)
        cfg = {
            "splitting": {
                "train_ratio": 0.5,
                "val_ratio": 0.25,
                "test_ratio": 0.25,
                "seed": 42,
                "search_attempts": 1000,
                "require_all_classes_in": ["train", "val", "test"],
            }
        }
        result, summary = split_manifest(frame, cfg)
        assignments = result.groupby("subject_name")["split"].nunique()
        self.assertTrue((assignments == 1).all())
        for name in ("train", "val", "test"):
            self.assertEqual(set(result.loc[result["split"].eq(name), "label"]), {"kidney", "liver"})
            self.assertGreater(summary["splits"][name]["rows"], 0)

    def test_impossible_class_coverage_fails(self):
        frame = pd.DataFrame([
            {"subject_name": "P001", "label": "rare"},
            {"subject_name": "P002", "label": "rare"},
            {"subject_name": "P003", "label": "common"},
            {"subject_name": "P004", "label": "common"},
            {"subject_name": "P005", "label": "common"},
        ])
        cfg = {"splitting": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "seed": 1,
                              "search_attempts": 100, "require_all_classes_in": ["train", "val", "test"]}}
        with self.assertRaises(ValueError):
            split_manifest(frame, cfg)

    def test_per_organ_split_workbook_has_three_detail_sheets(self):
        import tempfile
        from pathlib import Path

        frame = pd.DataFrame([
            {"file_path": str(Path("P001") / "2026_01_01_00_00_00" / "_hypergui_1" / "spectrum.csv"),
             "label": "bladder", "subject_name": "P001", "individual_name": "P001_Experiment1",
             "timestamp": "2026_01_01_00_00_00", "image_name": "P001#2026_01_01_00_00_00", "split": split}
            for split in ("train", "val", "test")
        ])
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_split_workbooks(frame, Path(directory))
            self.assertEqual([path.name for path in outputs], ["bladder_splits.xlsx"])
            workbook = load_workbook(outputs[0], data_only=True)
            self.assertEqual(workbook.sheetnames, ["Train Details", "Validation Details", "Test Details"])
            self.assertEqual(workbook["Train Details"]["E3"].value, 1)


if __name__ == "__main__":
    unittest.main()
