import unittest

import pandas as pd

from median_pipeline.preparation import split_manifest


class SplittingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
