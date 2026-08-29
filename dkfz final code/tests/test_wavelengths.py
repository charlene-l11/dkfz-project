import unittest

import numpy as np

from median_pipeline.wavelengths import (
    WavelengthSelectionError,
    parse_selective,
    resolve_wavelength_selection,
    scatter_selected_features,
)


class WavelengthTests(unittest.TestCase):
    def test_selected_features_are_scattered_into_fixed_width_input(self):
        selected = np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        result = scatter_selected_features(selected, [0, 1], n_input_channels=100)

        self.assertEqual(result.shape, (2, 100))
        np.testing.assert_array_equal(result[:, :2], selected)
        np.testing.assert_array_equal(result[:, 2:], np.zeros((2, 98), dtype=np.float32))

    def test_scattering_preserves_noncontiguous_wavelength_positions(self):
        selected = np.asarray([[10.0, 20.0]], dtype=np.float32)
        result = scatter_selected_features(selected, [0, 99], n_input_channels=100)

        self.assertEqual(result[0, 0], 10.0)
        self.assertEqual(result[0, 99], 20.0)
        self.assertEqual(np.count_nonzero(result), 2)

    def test_parse_mixed_exact_values_and_ranges(self):
        self.assertEqual(
            parse_selective("500; 560-720; 730-735; 875; 900-930"),
            [(500.0, 500.0), (560.0, 720.0), (730.0, 735.0), (875.0, 875.0), (900.0, 930.0)],
        )

    def test_parse_compact_stepwise_selection(self):
        self.assertEqual(
            parse_selective("500-515+580+600-610+995"),
            [(500.0, 515.0), (580.0, 580.0), (600.0, 610.0), (995.0, 995.0)],
        )

    def test_selective_grid_matches_requested_channels(self):
        result = resolve_wavelength_selection(
            {"min": 500, "max": 995, "selective": "500; 560-720; 730-735; 875; 900-930"},
            n_channels=100,
        )
        self.assertEqual(result["n_selected_channels"], 44)
        self.assertEqual(result["selected_wavelengths"][0], 500.0)
        self.assertIn(720.0, result["selected_wavelengths"])
        self.assertIn(735.0, result["selected_wavelengths"])
        self.assertEqual(result["selected_wavelengths"][-1], 930.0)

    def test_null_selects_full_grid(self):
        result = resolve_wavelength_selection({"min": 500, "max": 995, "selective": None}, 100)
        self.assertEqual(result["selected_indices"], list(range(100)))

    def test_exact_wavelength_must_exist_on_grid(self):
        with self.assertRaises(WavelengthSelectionError):
            resolve_wavelength_selection({"min": 500, "max": 995, "selective": "502"}, 100)

    def test_explicit_csv_grid_is_supported(self):
        grid = np.array([500.0, 507.5, 520.0, 540.0])
        result = resolve_wavelength_selection(
            {"min": 500, "max": 540, "selective": "507.5; 520-540"},
            n_channels=4,
            csv_wavelengths=grid,
        )
        self.assertEqual(result["selected_indices"], [1, 2, 3])
        self.assertEqual(result["grid_source"], "csv_wavelength_column")


if __name__ == "__main__":
    unittest.main()
