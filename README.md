# Cost-Aware ML Trading Signals Across 50 Indian Stocks

Research code for a documented 50-stock Indian equity experiment. It acquires daily Yahoo Finance data through yfinance, builds 17 technical features, evaluates XGBoost at 1-, 3-, and 5-session horizons and logistic regression at 1 session, then simulates per-stock and shared-capital trading with transaction costs.

**Research status:** exploratory. The fixed [50-symbol list](config/symbols.txt) is not point-in-time Nifty 50 membership, and the 2022–2025 evaluation period was examined during development. Do not interpret the results as an untouched holdout or evidence of live trading performance.

## Repository contents

| Path | Purpose |
| --- | --- |
| src/build_yfinance_data.py | Acquire raw daily OHLCV and actions; adjust OHLC consistently; create features and a hash manifest. |
| src/run_50_stock_benchmark.py | Chronological model training, diagnostics, technical baselines, and per-stock simulation. |
| src/run_threshold_experiment.py | Select entry/exit thresholds on 2021 validation data and evaluate them later. |
| src/run_shared_capital_portfolio.py | Shared INR 5 million cash ledger, sizing, cash constraints, and flat-cost sensitivity. |
| src/run_delivery_cost_scenario.py | Illustrative Indian delivery charges and 5 bp adverse slippage per leg. |
| src/portfolio_inference.py and src/analyze_portfolio_risk.py | Paired block bootstrap, risk metrics, and exposure-matched controls. |
| src/run_feature_diagnostics.py | Exploratory 2021 feature importance, group ablation, and probability calibration. |
| src/test_*.py | Accounting, portfolio, inference, risk, and data tests. |
| examples/ | Saved metric summaries and the October 2026 acquisition manifest; no vendor price series. |

This repository contains no paper, PDF, document builder, Yahoo raw prices, processed stock histories, or per-session trading paths.

## Setup

Use Python 3.12. Run these commands from the repository root. In Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s src -p 'test_*.py' -q
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s src -p 'test_*.py' -q
```

In the commands below, `python` means the virtual environment's Python executable (`.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux). You can activate the environment first, or substitute that executable for `python` in each command.

The model source, [requirements](requirements.txt), and random seeds are fixed in code. Installing the pinned packages is recommended; OS and hardware differences can still affect exact numeric reproduction.

## Acquire and validate data

```bash
python src/build_yfinance_data.py --download
python src/verify_yfinance_rebuild.py
```

The download uses yfinance 1.7.0, .NS symbols, daily data from 2018-01-01 through 2025-12-30, and explicit auto_adjust=False, actions=True, repair=False, and threads=False. The pipeline multiplies each raw OHLC field by that day's Adj Close / Close factor, retains raw volume, and recomputes the 17 features. It writes local raw and processed CSVs plus `data/yfinance_rebuild/manifest.json` with retrieval settings, timestamp, SHA-256 hashes, row counts, and dividend/split event counts.

The [example manifest](examples/acquisition_manifest_2026-10-25.json) records the documented 2026-10-25 acquisition. A fresh vendor download may contain revisions and need not match those hashes or the example metrics. After downloading, rerunning the builder without `--download` reconstructs processed files from local raw CSVs and verifies them against the current local manifest.

Yahoo data are not redistributed here. Check the provider's terms before sharing downloaded or derived per-session prices.

## Run the experiment

```bash
python src/run_yfinance_rebuild_experiments.py --stage all --placebo-replicates 500
python src/run_feature_diagnostics.py
```

The first command runs the chronological benchmark, 2021 threshold selection, shared-cash portfolios, 500 signal rotations, delivery-cost scenario, 5,000-replicate paired 20-session block bootstrap (seed 42), and risk/accounting summaries. The second command is a separate exploratory diagnostic: it fits on pre-2021 data and scores the **already used** 2021 validation year for importance, ablation, and calibration. It does not change trading-policy results. The full run can take substantial time.

To run the bootstrap alone **after** generating delivery-cost portfolio paths, use `python src/run_delivery_cost_bootstrap.py`. That script writes a separate focused CSV; it is not needed after `--stage all`.

Outputs are written to `results/yfinance_rebuild/` and ignored by Git. Summary snapshots from the documented acquisition are in [examples/metrics](examples/metrics).

## Main assumptions and interpretation

- Signals use data available after close t and first trade at the next open. The XGBoost policies enforce 1-, 3-, and 5-session minimum holds, respectively.
- The shared portfolio starts with INR 5 million, caps each entry ticket at INR 100,000, uses whole shares, allocates simultaneous buys within available cash, and credits idle cash at an assumed 7% annual rate.
- The delivery profile uses illustrative contemporary Indian delivery charges and assumed 5 bp adverse slippage per leg. It is not a reconstruction of historical broker contract notes. Full next-open fills, adjusted-price whole-share rounding, and missing impact/settlement details limit realism.
- The 50 names were selected retrospectively. The available data do not establish historical index membership or eliminate survivorship bias. The model-ready evaluation stops in 2025; no 2026 holdout result is claimed.
- In the saved delivery-cost shared-account snapshot, buy-and-hold returned **+97.11%**, 2021-selected logistic regression **+61.12%**, and fixed 1-day XGBoost **+24.65%**. These are historical simulated returns, not forecasts. The latter shortfall was the only annualized log-return contrast with Holm-adjusted p below 0.05 among 12 active policies; the comparison is still exploratory because of repeated test-period inspection.
- Feature importance and ablation are descriptive. The 2021 calibration diagnostic shows raw probabilities are imperfectly calibrated; no calibration transform was applied to the trading policies.

## Reproduction limits

An exact rerun of the example numbers requires the archived vendor inputs represented by the example acquisition manifest. Those CSVs are intentionally excluded. This repository supports a fresh, documented acquisition and full pipeline rerun, but Yahoo can revise historical prices and adjustment factors. A stronger study needs licensed, point-in-time constituents and corporate-action records, executable-price accounting, and a genuinely fresh evaluation period.
