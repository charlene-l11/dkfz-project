import unittest
from pathlib import Path

import numpy as np
import yaml

from median_pipeline.stepwise import StepwiseRun, _child_yaml_text, build_stepwise_runs


class StepwiseTests(unittest.TestCase):
    @staticmethod
    def wavelength_info():
        available = np.linspace(500, 995, 100).tolist()
        return {"available_wavelengths": available}

    def test_forward_and_reverse_ranges_are_independent(self):
        cfg = {
            "stepwise_analysis": True,
            "reverse_stepwise": True,
            "forward_stop_wavelength": 520,
            "reverse_stop_wavelength": 980,
        }
        runs = build_stepwise_runs(cfg, self.wavelength_info())
        forward = [run.range_name for run in runs if run.direction == "forward"]
        reverse = [run.range_name for run in runs if run.direction == "reverse"]

        self.assertEqual(forward, ["500-505", "500-510", "500-515", "500-520"])
        self.assertEqual(reverse, ["990-995", "985-995", "980-995"])

    def test_reverse_only_mode_is_supported(self):
        cfg = {
            "stepwise_analysis": False,
            "reverse_stepwise": True,
            "reverse_stop_wavelength": 985,
        }
        runs = build_stepwise_runs(cfg, self.wavelength_info())

        self.assertEqual([run.direction for run in runs], ["reverse", "reverse"])
        self.assertEqual([run.range_name for run in runs], ["990-995", "985-995"])

    def test_stop_wavelength_must_be_on_grid(self):
        cfg = {
            "stepwise_analysis": True,
            "reverse_stepwise": False,
            "forward_stop_wavelength": 512,
        }
        with self.assertRaisesRegex(ValueError, "not on the wavelength grid"):
            build_stepwise_runs(cfg, self.wavelength_info())

    def test_child_yaml_contains_range_but_not_master_stop_controls(self):
        master = {
            "stepwise_analysis": True,
            "reverse_stepwise": True,
            "forward_stop_wavelength": 640,
            "reverse_stop_wavelength": 850,
            "wavelength": {"min": 500, "max": 995, "selective": None},
        }
        run = StepwiseRun("forward", "500-505", 500, 505, (0, 1))
        child = yaml.safe_load(_child_yaml_text(master, Path("master.yaml"), run))

        self.assertFalse(child["stepwise_analysis"])
        self.assertFalse(child["reverse_stepwise"])
        self.assertEqual(child["wavelength"]["selective"], "500-505")
        self.assertNotIn("forward_stop_wavelength", child)
        self.assertNotIn("reverse_stop_wavelength", child)


if __name__ == "__main__":
    unittest.main()
