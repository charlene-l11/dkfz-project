import unittest
from pathlib import Path

import numpy as np
from median_pipeline.stepwise import StepwiseRun, _child_config, build_stepwise_runs


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
            "wavelength": {"selective": None},
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
            "wavelength": {"selective": None},
        }
        runs = build_stepwise_runs(cfg, self.wavelength_info())

        self.assertEqual([run.direction for run in runs], ["reverse", "reverse"])
        self.assertEqual([run.range_name for run in runs], ["990-995", "985-995"])

    def test_forward_stop_uses_highest_selected_wavelength_at_or_below_stop(self):
        cfg = {
            "stepwise_analysis": True,
            "reverse_stepwise": False,
            "forward_stop_wavelength": 512,
            "wavelength": {"selective": None},
        }
        runs = build_stepwise_runs(cfg, self.wavelength_info())
        self.assertEqual([run.range_name for run in runs], ["500-505", "500-510"])

    def test_child_config_contains_range_but_not_master_stop_controls(self):
        master = {
            "stepwise_analysis": True,
            "reverse_stepwise": True,
            "forward_stop_wavelength": 640,
            "reverse_stop_wavelength": 850,
            "wavelength": {"min": 500, "max": 995, "selective": None},
        }
        run = StepwiseRun("forward", "500-505", 500, 505, (0, 1))
        child = _child_config(master, Path("master.yaml"), run)

        self.assertFalse(child["stepwise_analysis"])
        self.assertFalse(child["reverse_stepwise"])
        self.assertEqual(child["wavelength"]["selective"], "500-505")
        self.assertNotIn("forward_stop_wavelength", child)
        self.assertNotIn("reverse_stop_wavelength", child)

    def test_selective_values_build_cumulative_noncontiguous_runs(self):
        cfg = {
            "stepwise_analysis": True,
            "reverse_stepwise": True,
            "forward_stop_wavelength": 630,
            "reverse_stop_wavelength": 510,
            "wavelength": {"selective": "500-515; 580; 600-610; 995"},
        }
        runs = build_stepwise_runs(cfg, self.wavelength_info())
        forward = [run.range_name for run in runs if run.direction == "forward"]
        reverse = [run.range_name for run in runs if run.direction == "reverse"]

        self.assertEqual(
            forward,
            [
                "500-505",
                "500-510",
                "500-515",
                "500-515+580",
                "500-515+580+600",
                "500-515+580+600-605",
                "500-515+580+600-610",
            ],
        )
        self.assertEqual(
            reverse,
            [
                "610+995",
                "605-610+995",
                "600-610+995",
                "580+600-610+995",
                "515+580+600-610+995",
                "510-515+580+600-610+995",
            ],
        )


if __name__ == "__main__":
    unittest.main()
