"""Unit checks for the shared-cash, fixed-ticket portfolio simulator."""

import unittest

import pandas as pd

from run_shared_capital_portfolio import (
    DAILY_CASH_RATE, DeliveryCostProfile, matched_circular_placebo, simulate_portfolio,
)


def panel(rows):
    return pd.DataFrame(rows, columns=[
        "Date", "Stock", "Signal", "Execution_Open", "Mark_Close",
    ])


class SharedCapitalPortfolioTests(unittest.TestCase):
    def test_cash_only_compounds_once_per_session(self):
        days = pd.date_range("2025-01-01", periods=3, freq="D")
        data = panel([(day, "AAA", False, 10.0, 10.0) for day in days])
        metrics, daily = simulate_portfolio(data, "cash", initial_capital=1000.0, ticket_cash=1000.0)
        expected = 1000.0 * (1 + DAILY_CASH_RATE) ** 3
        self.assertAlmostEqual(metrics["Final_Equity"], expected, places=8)
        self.assertEqual(metrics["Entries"], 0)
        self.assertEqual(daily["Active_Stocks"].sum(), 0)

    def test_whole_share_entry_cost_and_final_liquidation(self):
        days = pd.date_range("2025-01-01", periods=2, freq="D")
        data = panel([(days[0], "AAA", True, 10.0, 11.0),
                      (days[1], "AAA", True, 10.0, 12.0)])
        metrics, _ = simulate_portfolio(data, "one stock", initial_capital=1000.0, ticket_cash=1000.0)
        shares = 99
        cash_after_buy = 1000 - shares * 10 * 1.001
        expected = cash_after_buy * (1 + DAILY_CASH_RATE) ** 2 + shares * 12 * 0.999
        self.assertAlmostEqual(metrics["Final_Equity"], expected, places=8)
        self.assertEqual(metrics["Entries"], 1)
        self.assertEqual(metrics["Exits"], 1)
        self.assertAlmostEqual(metrics["Fees_Paid"], shares * 10 * 0.001 + shares * 12 * 0.001, places=8)

    def test_delivery_profile_applies_asymmetric_charges_dp_and_slippage(self):
        days = pd.date_range("2025-01-01", periods=2, freq="D")
        data = panel([(day, "AAA", True, 10.0, 10.0) for day in days])
        profile = DeliveryCostProfile()
        metrics, _ = simulate_portfolio(data, "delivery", initial_capital=1000.0,
                                        ticket_cash=1000.0, cost_profile=profile)
        buy_price = 10.0 * (1 + profile.slippage)
        sell_price = 10.0 * (1 - profile.slippage)
        shares = int(1000 // (buy_price * (1 + profile.proportional_fee("buy"))))
        buy_fee = profile.charge("buy", shares * buy_price)
        sell_fee = profile.charge("sell", shares * sell_price)
        expected = (1000 - shares * buy_price - buy_fee) * (1 + DAILY_CASH_RATE) ** 2
        expected += shares * sell_price - sell_fee
        self.assertAlmostEqual(metrics["Final_Equity"], expected, places=8)
        self.assertAlmostEqual(metrics["Fees_Paid"], buy_fee + sell_fee, places=8)
        self.assertGreater(profile.proportional_fee("buy"), profile.proportional_fee("sell"))

    def test_simultaneous_entries_are_capped_by_shared_cash(self):
        days = pd.date_range("2025-01-01", periods=2, freq="D")
        data = panel([(day, stock, True, 1000.0, 1000.0)
                      for day in days for stock in ("AAA", "BBB")])
        metrics, daily = simulate_portfolio(data, "cash constrained", initial_capital=1000.0, ticket_cash=500.0)
        self.assertEqual(metrics["Entries"], 0)
        self.assertEqual(metrics["Missed_Entries"], 4)
        self.assertEqual(daily["Active_Stocks"].sum(), 0)

    def test_incomplete_stock_date_panel_is_rejected(self):
        days = pd.date_range("2025-01-01", periods=2, freq="D")
        data = panel([(days[0], "AAA", False, 10.0, 10.0),
                      (days[1], "AAA", False, 10.0, 10.0),
                      (days[0], "BBB", False, 10.0, 10.0)])
        with self.assertRaisesRegex(ValueError, "Incomplete stock/date panel"):
            simulate_portfolio(data, "bad panel", initial_capital=1000.0, ticket_cash=500.0)

    def test_selected_five_day_policy_keeps_five_session_minimum_hold(self):
        days = pd.date_range("2025-01-01", periods=7, freq="D")
        data = panel([(day, "AAA", i == 0, 10.0, 10.0)
                      for i, day in enumerate(days)])
        metrics, daily = simulate_portfolio(
            data, "XGBoost 5D — 2021 selected", initial_capital=1000.0, ticket_cash=1000.0)
        self.assertEqual(metrics["Entries"], 1)
        self.assertEqual(daily["Exits"].tolist(), [0, 0, 0, 0, 0, 1, 0])

    def test_turnover_matched_placebo_preserves_each_stock_exposure_and_trade_legs(self):
        days = pd.date_range("2025-01-01", periods=10, freq="D")
        signals = [False, False, True, True, False, False, False, True, True, False]
        data = panel([(day, "AAA", signal, 10.0, 10.0)
                      for day, signal in zip(days, signals)])
        placebo = matched_circular_placebo(data, seed=4)
        original = data.Signal.astype(bool).to_numpy()
        shifted = placebo.Signal.astype(bool).to_numpy()

        def event_counts(values):
            entries = int(values[0]) + int(((~values[:-1]) & values[1:]).sum())
            exits = int((values[:-1] & (~values[1:])).sum()) + int(values[-1])
            return int(values.sum()), entries, exits

        self.assertEqual(event_counts(original), event_counts(shifted))
        self.assertNotEqual(original.tolist(), shifted.tolist())


if __name__ == "__main__":
    unittest.main()
