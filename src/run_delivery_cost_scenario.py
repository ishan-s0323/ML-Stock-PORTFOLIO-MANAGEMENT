"""Rerun stored signals with an illustrative contemporary Indian delivery profile.

The default signals use the documented yfinance rebuild. This cost scenario
is not a historically verified broker contract-note reconstruction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from run_shared_capital_portfolio import DeliveryCostProfile, load_strategy_panels, simulate_portfolio


def main() -> None:
    from run_shared_capital_portfolio import RESULTS

    output = RESULTS / "portfolio_level"
    profile = DeliveryCostProfile()
    rows, paths = [], []
    for strategy, panel in load_strategy_panels().items():
        metrics, daily = simulate_portfolio(panel, strategy, cost_profile=profile)
        rows.append(metrics)
        paths.append(daily)
    pd.DataFrame(rows).to_csv(output / "delivery_cost_portfolio_summary.csv", index=False)
    pd.concat(paths, ignore_index=True).to_csv(output / "delivery_cost_portfolio_paths.csv", index=False)
    config = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile.__dict__,
        "rate_sources": {
            "official_STT_stamp_SEBI_GST": "https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies",
            "illustrative_broker_exchange_and_zero_delivery_brokerage": "https://zerodha.com/charges/",
            "illustrative_broker_DP_sell": "https://support.zerodha.com/category/account-opening/resident-individual/ri-charges/articles/what-do-dp-charges-mean",
        },
        "slippage_assumption": "5 basis points adverse on each executed buy/sell, unmeasured",
        "charge_timing_caveat": "Current published broker rates applied to all 2022-2025 days for sensitivity; historical rate changes not reconstructed",
        "price_provenance": "Documented yfinance 1.7.0 acquisition; adjusted OHLC reconstructed from raw OHLC and Adj Close; point-in-time index membership unverified",
        "execution_assumption": "Full fill at next open adjusted by assumed slippage; final close liquidation similarly adjusted",
        "not_included": ["capital gains tax", "time-varying spread", "market impact", "partial fills", "settlement restrictions"],
    }
    (output / "delivery_cost_profile.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["Strategy", "Total_Return_Pct", "Trade_Legs", "Fees_Paid"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
