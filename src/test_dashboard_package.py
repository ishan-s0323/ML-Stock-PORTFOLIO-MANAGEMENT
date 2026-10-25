"""Tests that run in a public clone without redistributed vendor data."""

import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from dashboard.data_loader import (REGISTRY, DataUnavailable, ML, load_manifest,
                                   load_table, validate_selected_windows)


class DashboardPackageTests(unittest.TestCase):
    def test_public_bundle_hashes_and_required_tables(self):
        bundle = Path(__file__).resolve().parents[1] / "dashboard/public_outputs"
        declared = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))["files"]
        self.assertEqual(len(declared), len(REGISTRY) + 2)
        for relative, expected in declared.items():
            path = bundle / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        self.assertEqual(len(load_manifest()["symbols"]), 50)
        for key in REGISTRY:
            self.assertFalse(load_table(key).empty, key)

    def test_public_bundle_excludes_vendor_price_csvs(self):
        bundle = Path(__file__).resolve().parents[1] / "dashboard/public_outputs"
        self.assertFalse((bundle / "data/yfinance_rebuild/processed").exists())
        self.assertFalse((bundle / "data/yfinance_rebuild/raw").exists())

    def test_every_registered_source_is_rebuilt_yfinance(self):
        self.assertEqual(len(ML), 8)
        self.assertTrue(REGISTRY)
        self.assertTrue(all("yfinance_rebuild" in str(path) for path, _ in REGISTRY.values()))

    def test_missing_artifact_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "absent.csv"
            with patch.dict(REGISTRY, {"test_missing": (path, {"Value"})}):
                with self.assertRaisesRegex(DataUnavailable, "Missing saved rebuilt-panel output"):
                    load_table("test_missing")

    def test_bad_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("Wrong\n1\n", encoding="utf-8")
            with patch.dict(REGISTRY, {"test_schema": (path, {"Value"})}):
                with self.assertRaisesRegex(DataUnavailable, "Invalid test_schema schema"):
                    load_table("test_schema")

    def test_missing_manifest_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("dashboard.data_loader.DATA", Path(folder)):
                with self.assertRaisesRegex(DataUnavailable, "Missing acquisition manifest"):
                    load_manifest()

    def test_incomplete_window_family_is_rejected(self):
        import pandas as pd

        frame = pd.DataFrame({"Window_Sessions": [100], "Condition": ["worst"]})
        with self.assertRaises(DataUnavailable):
            validate_selected_windows(frame)


if __name__ == "__main__":
    unittest.main()
