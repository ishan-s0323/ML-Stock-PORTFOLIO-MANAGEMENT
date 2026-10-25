"""Quantify how separate-stock means differ from shared-capital returns."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "yfinance_rebuild"
PORT = RESULTS / "portfolio_level"


def main() -> None:
    stocks = pd.read_csv(RESULTS / "strategy_summary.csv").set_index("Strategy")
    selected = pd.read_csv(RESULTS / "threshold_test_summary.csv")
    selected = selected[selected.Policy == "2021 selected"].set_index("Model")
    flat = pd.read_csv(PORT / "portfolio_summary.csv").set_index("Strategy")
    delivery = pd.read_csv(PORT / "delivery_cost_portfolio_summary.csv").set_index("Strategy")
    rows = []
    for name in flat.index:
        if name == "Cash at 7%":
            continue
        if name.endswith(" — 2021 selected"):
            model = name.replace(" — 2021 selected", "")
            stock_mean = float(selected.loc[model, "Mean_Return_Pct"])
        else:
            stock_mean = float(stocks.loc[name, "Mean_Return_Pct"])
        shared = float(flat.loc[name, "Total_Return_Pct"])
        rows.append({"Policy": name, "Separate_Stock_Mean_Pct": stock_mean,
                     "Shared_10bp_Pct": shared,
                     "Accounting_Gap_Points": stock_mean - shared,
                     "Shared_Delivery_Profile_Pct": float(delivery.loc[name, "Total_Return_Pct"])})
    result = pd.DataFrame(rows)
    result.to_csv(PORT / "accounting_gap_summary.csv", index=False)
    print(result.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
