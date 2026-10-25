"""Validated, read-only access to the documented yfinance rebuild outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/yfinance_rebuild"
DATA = ROOT / "data/yfinance_rebuild"
REGISTRY: dict[str, tuple[Path, set[str]]] = {
    "model_summary": (BASE / "model_summary.csv", {"Model", "Mean_Fold_AUC", "Mean_Fold_Accuracy", "Mean_Fold_LogLoss", "Folds"}),
    "folds": (BASE / "fold_diagnostics.csv", {"Stock", "Fold", "Start", "Model", "AUC", "Accuracy", "LogLoss"}),
    "stock_metrics": (BASE / "stock_metrics.csv", {"Stock", "Strategy", "Total_Return_Pct", "Max_Drawdown_Pct"}),
    "portfolio_summary": (BASE / "portfolio_level/delivery_cost_portfolio_summary.csv", {"Strategy", "Initial_Capital", "Total_Return_Pct", "Sharpe", "Max_Drawdown_Pct", "Mean_Exposure_Pct", "Trade_Legs", "Fees_Paid"}),
    "portfolio_paths": (BASE / "portfolio_level/delivery_cost_portfolio_paths.csv", {"Date", "Strategy", "Equity", "Daily_Return", "Exposure_Pct", "Fees"}),
    "risk": (BASE / "portfolio_level/delivery_cost_risk_attribution.csv", {"Strategy", "Return_Pct", "Matched_Return_Pct", "Sharpe", "Sortino", "Matched_Sharpe"}),
    "accounting_gap": (BASE / "portfolio_level/accounting_gap_summary.csv", {"Policy", "Separate_Stock_Mean_Pct", "Shared_10bp_Pct", "Shared_Delivery_Profile_Pct"}),
    "cost_sweep": (BASE / "portfolio_level/portfolio_cost_sensitivity.csv", {"Strategy", "All_In_Cost_Per_Leg", "Total_Return_Pct", "Fees_Paid"}),
    "bootstrap": (BASE / "portfolio_level/delivery_cost_log_excess_bootstrap.csv", {"Policy", "Annualized_Log_Excess_Pct", "CI_95_Lower_Pct", "CI_95_Upper_Pct", "Unadjusted_P", "Holm_P"}),
    "sharpe_bootstrap": (BASE / "portfolio_level/delivery_cost_sharpe_bootstrap.csv", {"Strategy", "Observed_Sharpe_Difference", "CI_2_5", "CI_97_5", "Holm_Adjusted_P"}),
    "placebo": (BASE / "portfolio_level/logistic_placebo_summary.csv", {"Observed_Total_Return_Pct", "Placebo_Mean_Return_Pct", "One_Sided_Randomization_P"}),
    "importance": (BASE / "feature_diagnostics/feature_importance_summary.csv", {"Model", "Feature", "Normalized_Importance"}),
    "ablation": (BASE / "feature_diagnostics/group_ablation_summary.csv", {"Model", "Removed_Group", "AUC_Drop", "Brier_Increase"}),
    "calibration": (BASE / "feature_diagnostics/calibration_summary.csv", {"Model", "Pooled_AUC", "Pooled_Brier", "Pooled_ECE"}),
    "calibration_bins": (BASE / "feature_diagnostics/calibration_bins.csv", {"Model", "Mean_Probability", "Observed_Frequency", "Count"}),
    "validation_probabilities": (BASE / "feature_diagnostics/validation_probabilities.csv", {"Stock", "Model", "Date", "Probability", "Outcome"}),
    "selected_windows": (BASE / "regime_analysis_v3/direct_ml_vs_buy_hold_windows.csv", {"Condition", "Window_Sessions", "Start_Date", "End_Date", "Strategy", "Buy_Hold_Return_Pct", "Return_Pct", "Return_Diff_vs_Buy_Hold_Points", "Max_Drawdown_Pct", "Sharpe", "Mean_Exposure_Pct", "Trade_Legs", "Fee_Charges_INR"}),
    "window_summary": (BASE / "regime_analysis_v3/ml_outperformance_summary.csv", {"Condition", "Window_Sessions", "ML_Beating_Buy_Hold_Count"}),
    "regimes": (BASE / "regime_analysis_v3/direct_ml_vs_buy_hold_regimes.csv", {"Regime", "Strategy", "Sessions", "Mean_Daily_Return_Pct", "Mean_Exposure_Pct", "Mean_Daily_Gap_vs_Buy_Hold_Points"}),
    "regime_all": (BASE / "regime_analysis_v2/regime_conditioned_metrics_delivery.csv", {"Regime", "Strategy", "Sessions", "Mean_Daily_Return_Pct", "Mean_Exposure_Pct", "Volatility_Annual_Pct"}),
    "regime_episodes": (BASE / "regime_analysis_v2/contiguous_regime_episodes_all_labels.csv", {"Regime", "Episode", "Start_Date", "End_Date", "Strategy", "Return_Pct"}),
    "matched_windows": (BASE / "regime_analysis_v3/secondary_exposure_matched_windows.csv", {"Condition", "Window_Sessions", "Strategy", "Return_Pct", "Matched_Return_Pct", "Excess_vs_Matched_Points"}),
}

ML = ["XGBoost 1D", "XGBoost 3D", "XGBoost 5D", "XGBoost 1D — 2021 selected",
      "XGBoost 3D — 2021 selected", "XGBoost 5D — 2021 selected",
      "Logistic Regression 1D", "Logistic Regression 1D — 2021 selected"]


class DataUnavailable(Exception):
    """A required saved artifact is missing or incompatible."""


def load_table(key: str) -> pd.DataFrame:
    if key not in REGISTRY:
        raise KeyError(key)
    path, required = REGISTRY[key]
    if not path.is_file():
        raise DataUnavailable(f"Missing saved rebuilt-panel output: {path}. Restore the documented results; do not substitute legacy-panel results.")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise DataUnavailable(f"Invalid {key} schema: missing {sorted(missing)} in {path}")
    if frame.empty:
        raise DataUnavailable(f"Saved {key} result is empty: {path}")
    return frame


def load_manifest() -> dict:
    path = DATA / "manifest.json"
    if not path.is_file():
        raise DataUnavailable(f"Missing acquisition manifest: {path}. Run `python src/build_yfinance_data.py --download` only when a fresh acquisition is intended.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("retrieval_mode") != "fresh_yfinance" or len(manifest.get("symbols", [])) != 50:
        raise DataUnavailable("Acquisition manifest is not the documented 50-symbol yfinance rebuild")
    return manifest


def symbols() -> list[str]:
    return sorted(item["symbol"] for item in load_manifest()["symbols"])


def load_stock(symbol: str) -> pd.DataFrame:
    if symbol not in symbols():
        raise DataUnavailable(f"{symbol!r} is outside the documented 50-stock panel")
    path = DATA / "processed" / f"{symbol}_features.csv"
    if not path.is_file():
        raise DataUnavailable(f"Missing processed stock history: {path}. Rebuild from locally archived raw CSVs with `python src/build_yfinance_data.py` or acquire fresh data with `--download`.")
    frame = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "SMA_20", "SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "BBU_20_2.0", "BBL_20_2.0"}
    if not required.issubset(frame.columns) or frame.Date.duplicated().any():
        raise DataUnavailable(f"Invalid processed stock schema or duplicate dates: {path}")
    return frame.sort_values("Date").reset_index(drop=True)


def strategies() -> list[str]:
    names = load_table("portfolio_summary").Strategy.tolist()
    expected = set(ML) | {"Buy & Hold", "Cash at 7%", "SMA 9/21", "SMA Threshold 1%", "SMA 200", "RSI Mean Reversion"}
    if set(names) != expected or len(names) != 14:
        raise DataUnavailable("Delivery-cost portfolio policy family differs from the documented 14 policies")
    return ["Buy & Hold", "Cash at 7%"] + [n for n in names if n not in ("Buy & Hold", "Cash at 7%")]


def validate_selected_windows(frame: pd.DataFrame) -> None:
    if set(frame.Window_Sessions) != {100, 150, 200, 252, 378} or set(frame.Condition) != {"best", "worst"}:
        raise DataUnavailable("REGIME_V3 windows do not contain all five lengths and both conditions")
    expected = set(ML) | {"Buy & Hold"}
    for _, subset in frame.groupby(["Condition", "Window_Sessions"]):
        if set(subset.Strategy) != expected or len(subset) != 9:
            raise DataUnavailable("REGIME_V3 window is missing an ML policy or Buy & Hold")
        passive = float(subset.loc[subset.Strategy.eq("Buy & Hold"), "Return_Pct"].iloc[0])
        if not ((subset.Return_Pct - passive - subset.Return_Diff_vs_Buy_Hold_Points).abs() < 1e-7).all():
            raise DataUnavailable("REGIME_V3 return gaps disagree with Buy & Hold")


def source_label(key: str) -> str:
    path = REGISTRY[key][0]
    return str(path.relative_to(ROOT)).replace("\\", "/")
