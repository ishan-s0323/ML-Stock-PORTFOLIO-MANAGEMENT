"""Small economic checks for exposure-matched portfolio controls."""

import unittest

import numpy as np
import pandas as pd

from analyze_portfolio_risk import matched_path, ratios


class PortfolioRiskTests(unittest.TestCase):
    def test_matched_sleeves_reproduce_cash_and_passive_endpoints(self):
        dates = pd.date_range("2025-01-01", periods=3)
        passive_equity = np.array([110.0, 120.0, 130.0])
        cash_equity = np.array([100.0, 100.0, 100.0])
        passive = pd.DataFrame({"Date": dates, "Equity": passive_equity,
                                "Daily_Return": [0.1, 120 / 110 - 1, 130 / 120 - 1],
                                "Invested_Value": passive_equity})
        cash = pd.DataFrame({"Date": dates, "Equity": cash_equity,
                             "Daily_Return": [0.0, 0.0, 0.0]})
        weight, returns, exposure = matched_path(passive, cash, 0.0)
        self.assertAlmostEqual(weight, 0.0, places=10)
        self.assertAlmostEqual(exposure, 0.0, places=10)
        np.testing.assert_allclose(returns, 0.0, atol=1e-12)
        weight, returns, exposure = matched_path(passive, cash, 100.0)
        self.assertAlmostEqual(weight, 1.0, places=10)
        self.assertAlmostEqual(exposure, 100.0, places=10)
        np.testing.assert_allclose(returns, passive.Daily_Return, atol=1e-12)

    def test_deterministic_cash_has_undefined_sharpe(self):
        sharpe, sortino = ratios(np.repeat(0.07 / 252, 20))
        self.assertTrue(np.isnan(sharpe))
        self.assertTrue(np.isnan(sortino))


if __name__ == "__main__":
    unittest.main()
