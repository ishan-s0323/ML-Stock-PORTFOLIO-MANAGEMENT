"""Offline checks for the documented Yahoo adjustment and feature pipeline."""

import unittest

import numpy as np
import pandas as pd

from build_yfinance_data import features, validate_raw


class YfinanceDataTests(unittest.TestCase):
    def test_adjustment_applies_same_factor_to_all_ohlc(self):
        dates = pd.bdate_range("2020-01-01", periods=220, name="Date")
        close = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
        raw = pd.DataFrame({
            "Open": close - 1, "High": close + 2, "Low": close - 2,
            "Close": close, "Adj Close": close * 0.8,
            "Volume": np.arange(len(dates)) + 1000,
        }, index=dates)
        output = features(validate_raw(raw, "TEST"), "TEST")
        for column in ("Open", "High", "Low", "Close"):
            np.testing.assert_allclose(output[column], raw[column] * 0.8)
        np.testing.assert_allclose(output["Adj Close"], output["Close"])
        self.assertAlmostEqual(output["SMA_20"].iloc[-1], output["Close"].iloc[-20:].mean())
        self.assertEqual(output["Stock"].iloc[-1], "TEST")

    def test_invalid_ohlc_rejected(self):
        raw = pd.DataFrame({"Open": [10.0], "High": [9.0], "Low": [8.0],
                            "Close": [10.0], "Adj Close": [9.0],
                            "Volume": [100]},
                           index=pd.DatetimeIndex(["2020-01-01"], name="Date"))
        with self.assertRaisesRegex(ValueError, "OHLC"):
            validate_raw(raw, "TEST")


if __name__ == "__main__":
    unittest.main()
