"""Rerun the established experiment on the isolated yfinance rebuild panel."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import run_50_stock_benchmark as benchmark
import run_threshold_experiment as thresholds
import run_shared_capital_portfolio as shared
import run_delivery_cost_scenario as delivery
import analyze_portfolio_risk as risk
import summarize_accounting_gaps as gaps
from portfolio_inference import paired_block_bootstrap


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/yfinance_rebuild/processed"
OUT = ROOT / "results/yfinance_rebuild"


def call_cli(function, argv: list[str]) -> None:
    prior = sys.argv
    try:
        sys.argv = argv
        function()
    finally:
        sys.argv = prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("benchmark", "threshold", "portfolio", "delivery", "inference", "analysis", "all"), default="all")
    parser.add_argument("--placebo-replicates", type=int, default=500)
    args = parser.parse_args()
    if not (ROOT / "data/yfinance_rebuild/manifest.json").exists():
        raise FileNotFoundError("Run build_yfinance_data.py before the experiment")
    if len(list(DATA.glob("*_features.csv"))) != 50:
        raise ValueError("The rebuilt 50-stock panel is incomplete")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("benchmark", "all"):
        benchmark.DATA_DIR = DATA
        benchmark.run(OUT)
    if args.stage in ("threshold", "all"):
        benchmark.DATA_DIR = DATA
        thresholds.OUTPUT = OUT
        thresholds.main()
    if args.stage in ("portfolio", "all"):
        shared.RESULTS = OUT
        call_cli(shared.main, ["run_shared_capital_portfolio.py", "--output-dir",
                               str(OUT / "portfolio_level"), "--placebo-replicates",
                               str(args.placebo_replicates)])
    if args.stage in ("delivery", "all"):
        shared.RESULTS = OUT
        delivery.main()
    if args.stage in ("inference", "all"):
        path = OUT / "portfolio_level/delivery_cost_portfolio_paths.csv"
        frame = pd.read_csv(path, parse_dates=["Date"])
        active = frame.loc[frame.Strategy != "Cash at 7%"]
        result = paired_block_bootstrap(active, reps=5000, block_lengths=(20,), seed=42)
        result.loc[result.Strategy != "Buy & Hold"].to_csv(
            OUT / "portfolio_level/delivery_cost_bootstrap.csv", index=False)
    if args.stage in ("analysis", "all"):
        risk.RESULTS = OUT / "portfolio_level"
        call_cli(risk.main, ["analyze_portfolio_risk.py", "--input",
                             str(OUT / "portfolio_level/delivery_cost_portfolio_paths.csv"),
                             "--prefix", "delivery_cost"])
        gaps.RESULTS = OUT
        gaps.PORT = OUT / "portfolio_level"
        gaps.main()
    (OUT / "rerun_manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest": "data/yfinance_rebuild/manifest.json",
        "stage": args.stage,
        "placebo_replicates": args.placebo_replicates,
        "old_results_untouched": True,
        "test_status": "Exploratory reused 2022-2025 sample, not an untouched holdout",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved rebuilt experiment results in {OUT}")


if __name__ == "__main__":
    main()
