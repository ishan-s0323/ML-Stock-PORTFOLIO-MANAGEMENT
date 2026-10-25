"""Build fixed-ticket, shared-cash portfolio paths from full daily signal ledgers.

The portfolio starts with INR 5,000,000 and permits at most INR 100,000 of cash
including fees per stock entry. Positions are long-only, whole-share, and held
until the supplied signal exits (subject to the model's minimum hold). Unused
capital stays in a shared cash account. There is no daily rebalancing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "yfinance_rebuild"
INITIAL_CAPITAL = 5_000_000.0
TICKET_CASH = INITIAL_CAPITAL / 50
FEE_PER_LEG = 0.001
DAILY_CASH_RATE = 0.07 / 252
MIN_HOLD = {"XGBoost 1D": 1, "XGBoost 3D": 3, "XGBoost 5D": 5,
            "Logistic Regression 1D": 1}


@dataclass(frozen=True)
class DeliveryCostProfile:
    """Illustrative contemporary NSE delivery charges; rates are fractions."""

    stt: float = 0.001
    stamp_buy: float = 0.00015
    exchange: float = 0.0000307
    sebi: float = 0.000001
    gst: float = 0.18
    brokerage: float = 0.0
    dp_sell: float = 13.0 * 1.18
    slippage: float = 0.0005

    def proportional_fee(self, side: str) -> float:
        return (self.stt + (self.stamp_buy if side == "buy" else 0.0)
                + self.exchange + self.sebi
                + self.gst * (self.exchange + self.sebi + self.brokerage)
                + self.brokerage)

    def charge(self, side: str, notional: float) -> float:
        return notional * self.proportional_fee(side) + (self.dp_sell if side == "sell" else 0.0)


def validate_panel(panel: pd.DataFrame) -> tuple[list[str], list[pd.Timestamp]]:
    required = {"Date", "Stock", "Signal", "Execution_Open", "Mark_Close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Daily ledger is missing required fields: {sorted(missing)}")
    if panel.duplicated(["Date", "Stock"]).any():
        raise ValueError("Duplicate stock/date rows in portfolio ledger")
    stocks = sorted(panel["Stock"].unique())
    dates = sorted(pd.to_datetime(panel["Date"]).unique())
    expected = len(stocks) * len(dates)
    if len(panel) != expected:
        raise ValueError(f"Incomplete stock/date panel: expected {expected}, found {len(panel)}")
    if panel[["Execution_Open", "Mark_Close"]].isna().any().any():
        raise ValueError("Missing execution or mark price in portfolio ledger")
    if (panel[["Execution_Open", "Mark_Close"]] <= 0).any().any():
        raise ValueError("Nonpositive execution or mark price in portfolio ledger")
    return stocks, dates


def _filled_signals(values: pd.Series) -> np.ndarray:
    return values.astype("float64").ffill().fillna(0).to_numpy(dtype=bool)


def simulate_portfolio(panel: pd.DataFrame, strategy: str, fee_per_leg: float = FEE_PER_LEG,
                       initial_capital: float = INITIAL_CAPITAL,
                       ticket_cash: float | None = None,
                       cost_profile: DeliveryCostProfile | None = None) -> tuple[dict, pd.DataFrame]:
    """Simulate one shared cash account with equal fixed-cash entry tickets."""
    stocks, dates = validate_panel(panel)
    ticket = float(ticket_cash if ticket_cash is not None else initial_capital / len(stocks))
    frame = panel.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index(["Date", "Stock"]).sort_index()
    signal = {stock: _filled_signals(frame.xs(stock, level="Stock")["Signal"]) for stock in stocks}
    opens = {stock: frame.xs(stock, level="Stock")["Execution_Open"].to_numpy(float) for stock in stocks}
    closes = {stock: frame.xs(stock, level="Stock")["Mark_Close"].to_numpy(float) for stock in stocks}

    cash = float(initial_capital)
    shares = {stock: 0 for stock in stocks}
    entry_at = {stock: None for stock in stocks}
    hold = MIN_HOLD.get(strategy.split(" — ")[0], 1)
    rows = []
    equity_path = [cash]
    buy_notional_total = sell_notional_total = fees_total = 0.0
    entries = exits = missed_entries = 0

    for day, date in enumerate(dates):
        day_buys = day_sells = day_fees = 0.0
        day_entries = day_exits = day_missed = 0

        # Process all eligible exits at the same next-session open.
        for stock in stocks:
            if shares[stock] and not signal[stock][day] and day - entry_at[stock] >= hold:
                raw_price = opens[stock][day]
                price = raw_price * (1 - cost_profile.slippage) if cost_profile else raw_price
                notional = shares[stock] * price
                fee = cost_profile.charge("sell", notional) if cost_profile else notional * fee_per_leg
                cash += notional - fee
                day_sells += notional
                day_fees += fee
                shares[stock] = 0
                entry_at[stock] = None
                exits += 1
                day_exits += 1

        entrants = [stock for stock in stocks if shares[stock] == 0 and signal[stock][day]]
        if entrants:
            # Scale simultaneous entries pro rata when shared cash cannot fund
            # every full ticket. Fee is included inside each ticket budget.
            per_entry_cash = min(ticket, cash / len(entrants))
            for stock in entrants:
                raw_price = opens[stock][day]
                price = raw_price * (1 + cost_profile.slippage) if cost_profile else raw_price
                rate = cost_profile.proportional_fee("buy") if cost_profile else fee_per_leg
                quantity = int(np.floor(per_entry_cash / (price * (1 + rate))))
                if quantity < 1:
                    day_missed += 1
                    continue
                notional = quantity * price
                fee = cost_profile.charge("buy", notional) if cost_profile else notional * fee_per_leg
                cash -= notional + fee
                shares[stock] = quantity
                entry_at[stock] = day
                day_buys += notional
                day_fees += fee
                entries += 1
                day_entries += 1
            missed_entries += day_missed

        cash = max(0.0, cash)
        # Cash held after today's open trades earns the daily rate through close.
        cash *= 1 + DAILY_CASH_RATE
        stock_value = sum(shares[s] * closes[s][day] for s in stocks)
        day_exposure = stock_value
        day_active = sum(v > 0 for v in shares.values())
        equity = cash + stock_value
        # Positions are forcibly liquidated at the final available close.
        if day == len(dates) - 1:
            for stock in stocks:
                if shares[stock]:
                    raw_price = closes[stock][day]
                    price = raw_price * (1 - cost_profile.slippage) if cost_profile else raw_price
                    notional = shares[stock] * price
                    fee = cost_profile.charge("sell", notional) if cost_profile else notional * fee_per_leg
                    cash += notional - fee
                    day_sells += notional
                    day_fees += fee
                    shares[stock] = 0
                    exits += 1
                    day_exits += 1
            equity = cash
        # The final liquidation is at the close; it earns no post-test interest.
        equity_path.append(equity)
        buy_notional_total += day_buys
        sell_notional_total += day_sells
        fees_total += day_fees
        rows.append({
            "Date": date, "Strategy": strategy, "Equity": equity,
            "Cash_Balance": cash, "Invested_Value": day_exposure,
            "Exposure_Pct": 100 * day_exposure / equity if equity else np.nan,
            "Active_Stocks": day_active,
            "Entries": day_entries, "Exits": day_exits,
            "Missed_Entries": day_missed,
            "Buy_Notional": day_buys, "Sell_Notional": day_sells,
            "Fees": day_fees,
        })

    equity_path = np.asarray(equity_path, dtype=float)
    daily_returns = equity_path[1:] / equity_path[:-1] - 1
    daily = pd.DataFrame(rows)
    daily["Daily_Return"] = daily_returns
    drawdown = equity_path[1:] / np.maximum.accumulate(equity_path) [1:] - 1
    n_days = len(daily_returns)
    years = n_days / 252
    volatility = np.std(daily_returns, ddof=1) if n_days > 1 else 0.0
    excess = daily_returns - DAILY_CASH_RATE
    metrics = {
        "Strategy": strategy, "Stocks": len(stocks), "Sessions": n_days,
        "Initial_Capital": initial_capital, "Final_Equity": equity_path[-1],
        "Total_Return_Pct": 100 * (equity_path[-1] / initial_capital - 1),
        "Annual_Return_Pct": 100 * ((equity_path[-1] / initial_capital) ** (1 / years) - 1),
        "Sharpe": np.sqrt(252) * np.mean(excess) / volatility if volatility else np.nan,
        "Max_Drawdown_Pct": 100 * np.min(drawdown),
        "Mean_Exposure_Pct": daily["Exposure_Pct"].mean(),
        "Entries": entries, "Exits": exits, "Trade_Legs": entries + exits,
        "Missed_Entries": missed_entries,
        "Turnover_Pct_Of_Initial_Capital": 100 * (buy_notional_total + sell_notional_total) / initial_capital,
        "Fees_Paid": fees_total,
    }
    return metrics, daily


def matched_circular_placebo(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Circularly shift each stock's signal path, preserving exposure and legs.

    The shift is accepted only if linear-period exposure and entry/exit counts
    exactly match the original signal for that stock. A stock with no eligible
    nonzero shift keeps its original signal for this draw.
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    out["Signal"] = pd.to_numeric(out["Signal"], errors="coerce").astype(float)
    for stock, indices in out.groupby("Stock", sort=True).groups.items():
        idx = np.asarray(list(indices))
        values = _filled_signals(out.loc[idx, "Signal"])
        n = len(values)

        def counts(path):
            entries = int(path[0]) + int(np.sum((~path[:-1]) & path[1:]))
            exits = int(np.sum(path[:-1] & (~path[1:]))) + int(path[-1])
            return int(path.sum()), entries, exits

        target = counts(values)
        for _ in range(64):
            shifted = np.roll(values, int(rng.integers(1, n)))
            if counts(shifted) == target:
                out.loc[idx, "Signal"] = shifted.astype(int)
                break
    return out


def load_strategy_panels() -> dict[str, pd.DataFrame]:
    base_path = RESULTS / "daily_strategy_returns.csv"
    threshold_path = RESULTS / "threshold_test_daily.csv"
    base = pd.read_csv(base_path, parse_dates=["Date"])
    required_base = {"Signal", "Execution_Open", "Mark_Close"}
    if not required_base.issubset(base.columns):
        raise ValueError("Rerun the corrected benchmark first to generate the execution ledger")
    panels = {}
    for strategy, group in base.groupby("Strategy", sort=False):
        panels[strategy] = group[["Date", "Stock", "Signal", "Execution_Open", "Mark_Close"]].copy()
    cash_panel = panels["Buy & Hold"].copy()
    cash_panel["Signal"] = 0
    panels["Cash at 7%"] = cash_panel
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path, parse_dates=["Date"])
        if not required_base.issubset(thresholds.columns):
            raise ValueError("Rerun the corrected threshold experiment to generate its execution ledger")
        selected = thresholds[thresholds["Policy"] == "2021 selected"]
        for model, group in selected.groupby("Model", sort=False):
            panels[f"{model} — 2021 selected"] = group[
                ["Date", "Stock", "Signal", "Execution_Open", "Mark_Close"]
            ].copy()
    return panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "portfolio_level")
    parser.add_argument("--placebo-replicates", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = load_strategy_panels()
    summaries = []
    daily_paths = []
    cost_rows = []
    placebo_summary = None
    for strategy, panel in panels.items():
        metrics, daily = simulate_portfolio(panel, strategy)
        summaries.append(metrics)
        daily_paths.append(daily)
        for all_in_cost in (0.0, 0.001, 0.002, 0.004):
            cost_metrics, _ = simulate_portfolio(panel, strategy, fee_per_leg=all_in_cost)
            cost_rows.append({"Strategy": strategy, "All_In_Cost_Per_Leg": all_in_cost,
                              "Total_Return_Pct": cost_metrics["Total_Return_Pct"],
                              "Fees_Paid": cost_metrics["Fees_Paid"]})
        if strategy == "Logistic Regression 1D — 2021 selected":
            placebo_rows = []
            actual_return = float(metrics["Total_Return_Pct"])
            for seed in range(args.placebo_replicates):
                placebo = matched_circular_placebo(panel, seed)
                result, path = simulate_portfolio(placebo, f"LR turnover-matched placebo {seed}")
                placebo_rows.append({"Seed": seed, **result})
            placebo_frame = pd.DataFrame(placebo_rows)
            placebo_frame.to_csv(args.output_dir / "logistic_turnover_matched_placebos.csv", index=False)
            placebo_returns = placebo_frame["Total_Return_Pct"].to_numpy(float)
            placebo_summary = {
                "Strategy": "LR selected vs turnover-matched placebo",
                "Placebo_Replicates": len(placebo_returns),
                "Observed_Total_Return_Pct": actual_return,
                "Placebo_Mean_Return_Pct": float(np.mean(placebo_returns)),
                "Placebo_CI_2_5_Pct": float(np.quantile(placebo_returns, 0.025)),
                "Placebo_CI_97_5_Pct": float(np.quantile(placebo_returns, 0.975)),
                "Observed_Placebo_Percentile": float(np.mean(placebo_returns <= actual_return) * 100),
                "One_Sided_Randomization_P": float((1 + np.sum(placebo_returns >= actual_return)) / (1 + len(placebo_returns))),
                "Observed_Portfolio_Trade_Legs": int(metrics["Trade_Legs"]),
                "Placebo_Mean_Portfolio_Trade_Legs": float(placebo_frame["Trade_Legs"].mean()),
                "Observed_Portfolio_Mean_Exposure_Pct": float(metrics["Mean_Exposure_Pct"]),
                "Placebo_Mean_Portfolio_Exposure_Pct": float(placebo_frame["Mean_Exposure_Pct"].mean()),
                "Signal_Invested_Days_And_Transitions_Matched_Per_Stock": True,
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(args.output_dir / "portfolio_cost_sensitivity.csv", index=False)
    if placebo_summary is not None:
        pd.DataFrame([placebo_summary]).to_csv(args.output_dir / "logistic_placebo_summary.csv", index=False)
    pd.concat(daily_paths, ignore_index=True).to_csv(args.output_dir / "portfolio_daily_paths.csv", index=False)
    print(pd.DataFrame(summaries).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
