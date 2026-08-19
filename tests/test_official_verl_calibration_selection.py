import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "select_calibration_rows.py"
SPEC = importlib.util.spec_from_file_location("official_verl_calibration_selection", MODULE_PATH)
assert SPEC and SPEC.loader
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


class CalibrationSelectionTests(unittest.TestCase):
    def test_parses_ordered_unique_positions(self):
        self.assertEqual(selection.parse_positions("0,1,7"), [0, 1, 7])

    def test_rejects_duplicate_or_negative_positions(self):
        with self.assertRaises(ValueError):
            selection.parse_positions("0,0")
        with self.assertRaises(ValueError):
            selection.parse_positions("-1")


if __name__ == "__main__":
    unittest.main()
