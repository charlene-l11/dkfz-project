import unittest

from median_pipeline.training import _precomputed_input_settings


class TrainingInputConfigTests(unittest.TestCase):
    def test_original_reflectance_disables_htc_l1_expectation(self):
        settings = _precomputed_input_settings(False)

        self.assertIsNone(settings["normalization"])
        self.assertIsNone(settings["preprocessing"])
        self.assertEqual(settings["precomputed_pixel_normalization"], "none")

    def test_upstream_l1_disables_htc_l1_expectation_and_records_source(self):
        settings = _precomputed_input_settings(True)

        self.assertIsNone(settings["normalization"])
        self.assertIsNone(settings["preprocessing"])
        self.assertEqual(settings["precomputed_pixel_normalization"], "L1")


if __name__ == "__main__":
    unittest.main()
