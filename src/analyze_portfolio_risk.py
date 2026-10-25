"""Risk diagnostics and a buy-and-hold/cash exposure-matched control.

The control allocates capital once between the existing after-cost buy-and-hold
portfolio and the existing cash strategy. It does not rebalance daily. The
starting sleeve weight is solved to match each policy's *realized* mean value
exposure, so this is a retrospective diagnostic, not a tradable ex-ante rule.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from portfolio_inference import holm_adjust, moving_block_indices

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "yfinance_rebuild" / "portfolio_level"
CASH_RATE = 0.07 / 252


def ratios(returns: np.ndarray) -> tuple[float, float]:
    excess = returns - CASH_RATE
    volatility = np.std(returns, ddof=1)
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    # A deterministic cash series has only floating-point roundoff variance.
    return (float(np.sqrt(252) * excess.mean() / volatility) if volatility > 1e-10 else np.nan,
            float(np.sqrt(252) * excess.mean() / downside) if downside else np.nan)


def attribution(returns: np.ndarray, benchmark: np.ndarray) -> tuple[float, float]:
    x = benchmark - CASH_RATE
    y = returns - CASH_RATE
    variance = np.var(x, ddof=1)
    beta = float(np.cov(x, y, ddof=1)[0, 1] / variance) if variance else np.nan
    alpha = float(252 * (y.mean() - beta * x.mean())) if np.isfinite(beta) else np.nan
    return alpha, beta


def matched_path(passive: pd.DataFrame, cash: pd.DataFrame, target_exposure: float) -> tuple[float, np.ndarray, float]:
    start = float(passive.Equity.iloc[0] / (1 + passive.Daily_Return.iloc[0]))
    passive_equity = np.r_[start, passive.Equity.to_numpy(float)]
    cash_equity = np.r_[start, cash.Equity.to_numpy(float)]
    invested = passive.Invested_Value.to_numpy(float)

    def exposure(weight: float) -> float:
        equity = weight * passive_equity[1:] + (1 - weight) * cash_equity[1:]
        return float(np.mean(100 * weight * invested / equity))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if exposure(mid) < target_exposure:
            lo = mid
        else:
            hi = mid
    weight = (lo + hi) / 2
    equity = weight * passive_equity + (1 - weight) * cash_equity
    returns = equity[1:] / equity[:-1] - 1
    return weight, returns, exposure(weight)


def analyze(paths: pd.DataFrame, reps: int = 2000, block: int = 20, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paths.duplicated(["Date", "Strategy"]).any():
        raise ValueError("Duplicate strategy/date rows")
    groups = {name: group.sort_values("Date").reset_index(drop=True)
              for name, group in paths.groupby("Strategy")}
    passive = groups["Buy & Hold"]
    cash = groups["Cash at 7%"]
    dates = passive.Date.to_numpy()
    if any(not np.array_equal(frame.Date.to_numpy(), dates) for frame in groups.values()):
        raise ValueError("Paths do not share one calendar")
    passive_returns = passive.Daily_Return.to_numpy(float)
    rows, boot_rows = [], []
    indices = moving_block_indices(len(dates), block, reps, np.random.default_rng(seed))
    p_values = {}
    for name, frame in groups.items():
        returns = frame.Daily_Return.to_numpy(float)
        target = float(frame.Exposure_Pct.mean())
        weight, control, control_exposure = matched_path(passive, cash, target)
        sharpe, sortino = ratios(returns)
        control_sharpe, control_sortino = ratios(control)
        alpha, beta = attribution(returns, passive_returns)
        rows.append({
            "Strategy": name, "Mean_Exposure_Pct": target,
            "Matched_Initial_Passive_Weight": weight,
            "Matched_Exposure_Pct": control_exposure,
            "Return_Pct": 100 * (np.prod(1 + returns) - 1),
            "Matched_Return_Pct": 100 * (np.prod(1 + control) - 1),
            "Return_Gap_vs_Matched_Points": 100 * (np.prod(1 + returns) - np.prod(1 + control)),
            "Sharpe": sharpe, "Matched_Sharpe": control_sharpe,
            "Sortino": sortino, "Matched_Sortino": control_sortino,
            "Alpha_Annual_Pct": 100 * alpha, "Beta_vs_Passive": beta,
        })
        if name in ("Buy & Hold", "Cash at 7%"):
            continue
        observed = sharpe - control_sharpe
        sampled = np.empty(reps)
        for i in range(reps):
            a, _ = ratios(returns[indices[i]])
            b, _ = ratios(control[indices[i]])
            sampled[i] = a - b
        centered = sampled - sampled.mean()
        p_value = float((1 + np.sum(np.abs(centered) >= abs(observed))) / (reps + 1))
        p_values[name] = p_value
        boot_rows.append({"Strategy": name, "Block_Length_Sessions": block,
                          "Bootstrap_Replicates": reps, "Observed_Sharpe_Difference": observed,
                          "CI_2_5": np.quantile(sampled, 0.025),
                          "CI_97_5": np.quantile(sampled, 0.975), "Raw_P": p_value})
    adjusted = holm_adjust(p_values)
    for row in boot_rows:
        row["Holm_Adjusted_P"] = adjusted[row["Strategy"]]
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RESULTS / "portfolio_daily_paths.csv")
    parser.add_argument("--prefix", default="portfolio")
    args = parser.parse_args()
    paths = pd.read_csv(args.input, parse_dates=["Date"])
    metrics, tests = analyze(paths)
    metrics.to_csv(RESULTS / f"{args.prefix}_risk_attribution.csv", index=False)
    tests.to_csv(RESULTS / f"{args.prefix}_sharpe_bootstrap.csv", index=False)
    print(metrics[["Strategy", "Return_Pct", "Matched_Return_Pct", "Sharpe", "Matched_Sharpe", "Sortino", "Alpha_Annual_Pct", "Beta_vs_Passive"]].round(3).to_string(index=False))
    print(tests[["Strategy", "Observed_Sharpe_Difference", "CI_2_5", "CI_97_5", "Holm_Adjusted_P"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
