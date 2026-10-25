"""Paired moving-block bootstrap for shared-capital portfolio paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "yfinance_rebuild" / "portfolio_level" / "portfolio_daily_paths.csv"
DEFAULT_OUTPUT = ROOT / "results" / "yfinance_rebuild" / "portfolio_level" / "bootstrap_inference.csv"


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, returned under original keys."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[name] = running
    return adjusted


def moving_block_indices(n: int, block_length: int, reps: int, rng: np.random.Generator,
                         batch_size: int = 200) -> np.ndarray:
    """Draw circular moving-block bootstrap indices in bounded-size batches."""
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    batches = []
    for start in range(0, reps, batch_size):
        size = min(batch_size, reps - start)
        starts = rng.integers(0, n, size=(size, blocks))
        indices = (starts[:, :, None] + offsets[None, None, :]) % n
        batches.append(indices.reshape(size, -1)[:, :n])
    return np.concatenate(batches, axis=0)


def paired_block_bootstrap(daily: pd.DataFrame, benchmark: str = "Buy & Hold",
                           reps: int = 5000, block_lengths: tuple[int, ...] = (10, 20, 40),
                           seed: int = 42) -> pd.DataFrame:
    required = {"Date", "Strategy", "Daily_Return"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Daily portfolio paths are missing fields: {sorted(missing)}")
    if daily.duplicated(["Date", "Strategy"]).any():
        raise ValueError("Duplicate date/strategy rows in portfolio paths")
    matrix = daily.pivot(index="Date", columns="Strategy", values="Daily_Return").sort_index()
    if matrix.isna().any().any():
        raise ValueError("Portfolio strategies must share one complete date panel")
    if benchmark not in matrix.columns:
        raise ValueError(f"Benchmark not found: {benchmark}")
    returns = matrix.to_numpy(float)
    if (returns <= -1).any() or not np.isfinite(returns).all():
        raise ValueError("Daily returns must be finite and greater than -100%")
    n, k = returns.shape
    if n < max(block_lengths):
        raise ValueError("The evaluation period is shorter than the longest bootstrap block")
    names = matrix.columns.tolist()
    baseline_i = names.index(benchmark)
    log_returns = np.log1p(returns)
    observed_total = np.prod(1 + returns, axis=0) - 1
    observed_excess = (log_returns - log_returns[:, [baseline_i]]).mean(axis=0) * 252
    comparisons = [name for name in names if name != benchmark]
    records = []
    for block_length in block_lengths:
        rng = np.random.default_rng(seed + block_length)
        indices = moving_block_indices(n, block_length, reps, rng)
        boot_total = np.empty((reps, k), dtype=float)
        boot_excess = np.empty((reps, k), dtype=float)
        null_excess = np.empty((reps, len(comparisons)), dtype=float)
        comparison_columns = [names.index(name) for name in comparisons]
        centered_delta = log_returns[:, comparison_columns] - log_returns[:, [baseline_i]]
        centered_delta -= centered_delta.mean(axis=0, keepdims=True)
        for start in range(0, reps, 200):
            stop = min(start + 200, reps)
            sample = returns[indices[start:stop], :]
            sample_log = np.log1p(sample)
            boot_total[start:stop] = np.prod(1 + sample, axis=1) - 1
            boot_excess[start:stop] = (
                sample_log - sample_log[:, :, baseline_i:baseline_i + 1]
            ).mean(axis=1) * 252
            null_excess[start:stop] = centered_delta[indices[start:stop], :].mean(axis=1) * 252

        raw_p = {}
        for column, name in enumerate(comparisons):
            observed = observed_excess[names.index(name)]
            raw_p[name] = (1 + int((np.abs(null_excess[:, column]) >= abs(observed)).sum())) / (reps + 1)
        adjusted = holm_adjust(raw_p)
        for j, name in enumerate(names):
            record = {
                "Strategy": name, "Benchmark": benchmark,
                "Block_Length_Sessions": block_length, "Bootstrap_Replicates": reps,
                "Observed_Total_Return_Pct": 100 * observed_total[j],
                "Total_Return_CI_2_5_Pct": 100 * np.quantile(boot_total[:, j], 0.025),
                "Total_Return_CI_97_5_Pct": 100 * np.quantile(boot_total[:, j], 0.975),
                "Observed_Annualized_Log_Excess_Pct": 100 * observed_excess[j],
                "Annualized_Log_Excess_CI_2_5_Pct": 100 * np.quantile(boot_excess[:, j], 0.025),
                "Annualized_Log_Excess_CI_97_5_Pct": 100 * np.quantile(boot_excess[:, j], 0.975),
                "Paired_Block_Bootstrap_P": raw_p.get(name, np.nan),
                "Holm_Adjusted_P": adjusted.get(name, np.nan),
            }
            records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--benchmark", default="Buy & Hold")
    args = parser.parse_args()
    paths = pd.read_csv(args.input, parse_dates=["Date"])
    result = paired_block_bootstrap(paths, args.benchmark, args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
