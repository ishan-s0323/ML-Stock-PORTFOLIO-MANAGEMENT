import unittest

import numpy as np
import pandas as pd

from portfolio_inference import holm_adjust, paired_block_bootstrap


class PortfolioInferenceTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_bounded(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(adjusted, {"a": 0.03, "c": 0.06, "b": 0.06})

    def test_paired_bootstrap_returns_intervals_and_adjusted_comparisons(self):
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        benchmark_returns = np.full(len(dates), 0.0002)
        strategy_returns = benchmark_returns + 0.0001
        rows = []
        for name, values in (("Buy & Hold", benchmark_returns), ("Model", strategy_returns)):
            rows.extend({"Date": date, "Strategy": name, "Daily_Return": value}
                        for date, value in zip(dates, values))
        result = paired_block_bootstrap(pd.DataFrame(rows), reps=80, block_lengths=(5,), seed=4)
        model = result[result["Strategy"] == "Model"].iloc[0]
        self.assertGreater(model["Observed_Total_Return_Pct"], 0)
        self.assertLessEqual(model["Total_Return_CI_2_5_Pct"], model["Observed_Total_Return_Pct"])
        self.assertGreaterEqual(model["Total_Return_CI_97_5_Pct"], model["Observed_Total_Return_Pct"])
        self.assertEqual(model["Holm_Adjusted_P"], model["Paired_Block_Bootstrap_P"])


if __name__ == "__main__":
    unittest.main()
