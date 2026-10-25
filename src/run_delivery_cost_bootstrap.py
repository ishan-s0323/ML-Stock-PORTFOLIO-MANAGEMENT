"""Reproduce the 20-session paired bootstrap on saved delivery-cost paths."""

from pathlib import Path

import pandas as pd

from portfolio_inference import paired_block_bootstrap


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/yfinance_rebuild/portfolio_level/delivery_cost_portfolio_paths.csv"
OUTPUT = ROOT / "results/yfinance_rebuild/portfolio_level/delivery_cost_log_excess_bootstrap.csv"
SEED = 42
REPLICATES = 5000
BLOCK_LENGTH = 20


def main() -> None:
    paths = pd.read_csv(INPUT, parse_dates=["Date"])
    paths = paths.loc[paths["Strategy"] != "Cash at 7%"].copy()
    result = paired_block_bootstrap(
        paths, benchmark="Buy & Hold", reps=REPLICATES,
        block_lengths=(BLOCK_LENGTH,), seed=SEED,
    )
    result = result.loc[result["Strategy"] != "Buy & Hold"].copy()
    if len(result) != 12:
        raise ValueError(f"Expected 12 active policies, found {len(result)}")
    result = result.rename(columns={
        "Strategy": "Policy",
        "Observed_Annualized_Log_Excess_Pct": "Annualized_Log_Excess_Pct",
        "Annualized_Log_Excess_CI_2_5_Pct": "CI_95_Lower_Pct",
        "Annualized_Log_Excess_CI_97_5_Pct": "CI_95_Upper_Pct",
        "Paired_Block_Bootstrap_P": "Unadjusted_P",
        "Holm_Adjusted_P": "Holm_P",
    })
    result = result[[
        "Policy", "Annualized_Log_Excess_Pct", "CI_95_Lower_Pct",
        "CI_95_Upper_Pct", "Unadjusted_P", "Holm_P",
    ]]
    result["Policy"] = result["Policy"].str.replace("�", "—", regex=False)
    result["Seed"] = SEED
    result["Bootstrap_RNG_Seed"] = SEED + BLOCK_LENGTH
    result["Family_Policies"] = 12
    result["Replicates"] = REPLICATES
    result["Block_Length_Sessions"] = BLOCK_LENGTH
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.round(4).to_string(index=False))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
