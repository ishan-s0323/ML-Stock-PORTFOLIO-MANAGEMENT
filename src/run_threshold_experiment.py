"""Select fixed hysteresis thresholds in 2021, compare on 2022-25 predictions.

The later years have already been inspected in this project, so their results
are a reused-test exploratory comparison, not fresh independent confirmation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from run_50_stock_benchmark import (
    FEATURES, HORIZONS, ROOT, XGB_PARAMS, backtest, load_stock,
    predict_walk_forward, source_files,
)

OUTPUT = ROOT / "results" / "yfinance_rebuild"
VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_END_EXCLUSIVE = pd.Timestamp("2021-12-16")
TEST_START = pd.Timestamp("2022-01-01")
MODELS = ("XGBoost 1D", "XGBoost 3D", "XGBoost 5D", "Logistic Regression 1D")
ENTRIES = (0.5, 0.525, 0.55, 0.575)
EXITS = (0.425, 0.45, 0.475, 0.5)
CANDIDATES = tuple((entry, exit_) for entry in ENTRIES for exit_ in EXITS if exit_ <= entry)
FEES = (0.0, 0.001, 0.002, 0.004)


def desired_positions(probabilities: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    held = False
    signals = np.empty(len(probabilities), dtype=bool)
    for i, probability in enumerate(probabilities):
        if not held and probability >= entry:
            held = True
        elif held and probability < exit_:
            held = False
        signals[i] = held
    return signals


def validation_predictions(df: pd.DataFrame) -> pd.DataFrame:
    features_valid = df.loc[:, list(FEATURES)].notna().all(axis=1)
    next_session = pd.Series(df.index, index=df.index).shift(-1)
    validation_mask = ((df.index >= VALIDATION_START) &
                       (df.index < VALIDATION_END_EXCLUSIVE) & next_session.notna())
    validation = df.loc[validation_mask].copy()
    if int((features_valid.loc[validation.index] & validation["forward_return_5d"].notna()).sum()) < 150:
        raise ValueError("Insufficient validation rows")
    for horizon in HORIZONS:
        outcome_end = pd.Series(df.index, index=df.index).shift(-(horizon + 1))
        train_mask = ((df.index < VALIDATION_START) & (outcome_end < VALIDATION_START) &
                      features_valid & df[f"target_{horizon}d"].notna())
        train = df.loc[train_mask]
        if len(train) < 252:
            raise ValueError("Insufficient pre-2021 training rows")
        predict_mask = validation.loc[:, list(FEATURES)].notna().all(axis=1)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train.loc[:, list(FEATURES)])
        x_val = scaler.transform(validation.loc[predict_mask, list(FEATURES)])
        y_train = train[f"target_{horizon}d"].to_numpy(dtype=int)
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(x_train, y_train)
        validation[f"XGBoost {horizon}D_Probability"] = np.nan
        validation.loc[predict_mask, f"XGBoost {horizon}D_Probability"] = model.predict_proba(x_val)[:, 1]
        if horizon == 1:
            linear = LogisticRegression(C=0.1, max_iter=500, random_state=42)
            linear.fit(x_train, y_train)
            validation["Logistic Regression 1D_Probability"] = np.nan
            validation.loc[predict_mask, "Logistic Regression 1D_Probability"] = linear.predict_proba(x_val)[:, 1]
    return validation


def hold_days(model: str) -> int:
    return 3 if model == "XGBoost 3D" else 5 if model == "XGBoost 5D" else 1


def main() -> None:
    files = source_files()
    if len(files) != 50:
        raise ValueError(f"Expected 50 stocks, got {len(files)}")
    datasets = {}
    validation_rows = []
    for number, path in enumerate(files, start=1):
        stock = path.name.removesuffix("_features.csv")
        df, _ = load_stock(path)
        validation = validation_predictions(df)
        test, _ = predict_walk_forward(df, stock)
        datasets[stock] = (df, test)
        for model in MODELS:
            probabilities = validation[f"{model}_Probability"].to_numpy()
            for entry, exit_ in CANDIDATES:
                signals = desired_positions(probabilities, entry, exit_)
                metrics, _ = backtest(df, validation, signals, hold_days(model))
                validation_rows.append({
                    "Stock": stock, "Model": model, "Entry_Threshold": entry,
                    "Exit_Threshold": exit_, "Net_Return_Pct": metrics["Total_Return_Pct"],
                    "Entries": metrics["Entries"], "Trade_Legs": metrics["Trade_Legs"],
                    "Exposure_Pct": metrics["Exposure_Pct"],
                })
        print(f"Prepared {number}/50: {stock}", flush=True)
    validation_frame = pd.DataFrame(validation_rows)
    candidate_summary = validation_frame.groupby(
        ["Model", "Entry_Threshold", "Exit_Threshold"], sort=False).agg(
        Stocks=("Stock", "nunique"), Mean_Net_Return_Pct=("Net_Return_Pct", "mean"),
        Mean_Entries=("Entries", "mean"), Mean_Trade_Legs=("Trade_Legs", "mean"),
        Mean_Exposure_Pct=("Exposure_Pct", "mean"),
    ).reset_index()
    selected = {}
    for model in MODELS:
        group = candidate_summary[candidate_summary["Model"] == model].copy()
        baseline = group[(group["Entry_Threshold"] == 0.5) &
                         (group["Exit_Threshold"] == 0.5)].iloc[0]
        eligible = group[(group["Mean_Exposure_Pct"].between(20, 80)) &
                         (group["Mean_Trade_Legs"] <= baseline["Mean_Trade_Legs"] + 1e-9)]
        if eligible.empty:
            chosen = baseline
        else:
            chosen = eligible.sort_values(
                ["Mean_Net_Return_Pct", "Mean_Trade_Legs", "Entry_Threshold"],
                ascending=[False, True, True], kind="stable").iloc[0]
        selected[model] = (float(chosen["Entry_Threshold"]), float(chosen["Exit_Threshold"]))
        print(f"2021 selected {model}: entry {selected[model][0]:.3f}, exit {selected[model][1]:.3f}; "
              f"validation mean net {chosen['Mean_Net_Return_Pct']:.2f}%")
    test_rows, fee_rows, daily_rows = [], [], []
    for stock, (df, test) in datasets.items():
        for model in MODELS:
            probabilities = test[f"{model}_Probability"].to_numpy()
            for policy, (entry, exit_) in (("Fixed 0.5", (0.5, 0.5)),
                                           ("2021 selected", selected[model])):
                signals = desired_positions(probabilities, entry, exit_)
                for fee in FEES:
                    metrics, path = backtest(df, test, signals, hold_days(model), fee_per_leg=fee)
                    row = {"Stock": stock, "Model": model, "Policy": policy,
                           "Entry_Threshold": entry, "Exit_Threshold": exit_,
                           "Fee_Per_Leg": fee, **metrics}
                    fee_rows.append(row)
                    if fee == 0.001:
                        test_rows.append(row)
                        path.insert(1, "Stock", stock)
                        path.insert(2, "Model", model)
                        path.insert(3, "Policy", policy)
                        daily_rows.append(path)
    test_frame = pd.DataFrame(test_rows)
    fee_frame = pd.DataFrame(fee_rows)
    baseline = test_frame[test_frame["Policy"] == "Fixed 0.5"][
        ["Stock", "Model", "Total_Return_Pct"]].rename(
        columns={"Total_Return_Pct": "Fixed_Baseline_Return_Pct"})
    test_frame = test_frame.merge(baseline, on=["Stock", "Model"], validate="many_to_one")
    test_frame["Excess_vs_Fixed_Pct_Points"] = (
        test_frame["Total_Return_Pct"] - test_frame["Fixed_Baseline_Return_Pct"])
    summary = test_frame.groupby(["Model", "Policy"], sort=False).agg(
        Stocks=("Stock", "nunique"), Mean_Return_Pct=("Total_Return_Pct", "mean"),
        Median_Return_Pct=("Total_Return_Pct", "median"),
        Mean_Excess_vs_Fixed_Pct_Points=("Excess_vs_Fixed_Pct_Points", "mean"),
        Stocks_Beating_Fixed=("Excess_vs_Fixed_Pct_Points", lambda x: int((x > 0).sum())),
        Mean_Max_Drawdown_Pct=("Max_Drawdown_Pct", "mean"),
        Mean_Entries=("Entries", "mean"), Mean_Trade_Legs=("Trade_Legs", "mean"),
        Mean_Exposure_Pct=("Exposure_Pct", "mean"),
    ).reset_index()
    fee_summary = fee_frame.groupby(["Model", "Policy", "Fee_Per_Leg"], sort=False).agg(
        Mean_Return_Pct=("Total_Return_Pct", "mean"),
        Mean_Trade_Legs=("Trade_Legs", "mean"),
    ).reset_index()
    validation_frame.to_csv(OUTPUT / "threshold_validation_stock_metrics.csv", index=False)
    candidate_summary.to_csv(OUTPUT / "threshold_validation_candidates.csv", index=False)
    test_frame.to_csv(OUTPUT / "threshold_test_stock_metrics.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(OUTPUT / "threshold_test_daily.csv", index=False)
    summary.to_csv(OUTPUT / "threshold_test_summary.csv", index=False)
    fee_summary.to_csv(OUTPUT / "threshold_cost_sensitivity.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validation_start": str(VALIDATION_START.date()),
        "validation_end_exclusive": str(VALIDATION_END_EXCLUSIVE.date()),
        "test_start": str(TEST_START.date()),
        "selection_rule": "Per-model global threshold pair maximizing mean 2021 stock net return among candidates with 20%-80% mean exposure and no more mean trade legs than fixed 0.5; fallback fixed 0.5 if none eligible",
        "candidate_entries": ENTRIES, "candidate_exits": EXITS,
        "selected": {model: {"entry": pair[0], "exit": pair[1]} for model, pair in selected.items()},
        "fee_per_leg_values": FEES,
        "caveat": "2022-2025 test years were previously examined for other project experiments; this is exploratory reused-test evidence.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / "threshold_experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.round(3).to_string(index=False))
    print(fee_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
