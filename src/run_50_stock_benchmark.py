"""Reproducible 50-stock, nine-strategy daily walk-forward benchmark.

Signals use information available at the close of day t. Trades execute at
the next trading day's open. H-day classifiers predict the open-to-open return
from t+1 to t+1+H and hold a new position for at least H trading sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "yfinance_rebuild" / "processed"
DEFAULT_OUTPUT = ROOT / "results" / "yfinance_rebuild"
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2026-01-01")
INITIAL_CASH = 100_000.0
FEE_PER_LEG = 0.001
DAILY_CASH_RATE = 0.07 / 252
HORIZONS = (1, 3, 5)
FEATURES = (
    "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0",
    "SMA_10", "SMA_20", "SMA_50", "SMA_200", "Return_1d", "Volume_Change",
    "Close_vs_SMA20", "Close_vs_SMA50",
)
XGB_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=4,
)
STRATEGIES = (
    "XGBoost 1D", "XGBoost 3D", "XGBoost 5D", "Logistic Regression 1D",
    "SMA 9/21", "SMA Threshold 1%", "SMA 200", "RSI Mean Reversion", "Buy & Hold",
)


def source_files() -> list[Path]:
    files = sorted(DATA_DIR.glob("*_features.csv"))
    return [p for p in files if p.name not in
            {"all_stocks_features.csv", "market_regime_features.csv"}]


def load_stock(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    if not df.index.is_unique:
        raise ValueError(f"Duplicate dates in {path.name}")
    required = set(FEATURES) | {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    ohlc = df[["Open", "High", "Low", "Close"]]
    if ohlc.isna().any().any() or (ohlc <= 0).any().any():
        raise ValueError(f"Missing or nonpositive OHLC prices in {path.name}")
    if df["Volume"].isna().any() or (df["Volume"] < 0).any():
        raise ValueError(f"Missing or negative volume in {path.name}")
    if ((df["High"] < ohlc[["Open", "Close", "Low"]].max(axis=1)) |
            (df["Low"] > ohlc[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError(f"OHLC values are internally inconsistent in {path.name}")
    df.loc[:, list(FEATURES)] = df.loc[:, list(FEATURES)].replace([np.inf, -np.inf], np.nan)

    # Historical files contain two incompatible Bollinger conventions. Use one
    # trailing, close-only definition for every stock before fitting models.
    middle = df["Close"].rolling(20, min_periods=20).mean()
    spread = 2 * df["Close"].rolling(20, min_periods=20).std(ddof=1)
    lower, upper = middle - spread, middle + spread
    df["BBL_20_2.0"] = lower
    df["BBM_20_2.0"] = middle
    df["BBU_20_2.0"] = upper
    df["BBB_20_2.0"] = (upper - lower) / middle
    df["BBP_20_2.0"] = (df["Close"] - lower) / (upper - lower)

    for horizon in HORIZONS:
        forward_return = df["Open"].shift(-(horizon + 1)) / df["Open"].shift(-1) - 1
        df[f"forward_return_{horizon}d"] = forward_return
        df[f"target_{horizon}d"] = np.where(
            forward_return.notna(), (forward_return > 0).astype(float), np.nan)

    df["SMA9"] = df["Close"].rolling(9).mean()
    df["SMA21"] = df["Close"].rolling(21).mean()
    df["SMA 9/21"] = (df["SMA9"] > df["SMA21"]).astype(int)
    spread = (df["SMA9"] - df["SMA21"]) / df["SMA21"]
    state = 0
    threshold = []
    for value in spread:
        if pd.notna(value):
            if value > 0.01:
                state = 1
            elif value < -0.01:
                state = 0
        threshold.append(state)
    df["SMA Threshold 1%"] = threshold
    df["SMA 200"] = (df["Close"] > df["SMA_200"]).astype(int)
    state = 0
    rsi_signal = []
    for value in df["RSI_14"]:
        if pd.notna(value):
            if value < 30:
                state = 1
            elif value > 50:
                state = 0
        rsi_signal.append(state)
    df["RSI Mean Reversion"] = rsi_signal

    usable = df.dropna(subset=list(FEATURES) + ["forward_return_5d"]).copy()
    audit = {
        "Stock": path.name.removesuffix("_features.csv"),
        "Source_Rows": len(df), "Usable_Rows": len(usable),
        "First_Date": df.index.min().date().isoformat(),
        "Last_Date": df.index.max().date().isoformat(),
        "Missing_OHLCV_Values": int(df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum()),
        "Nonpositive_OHLC_Rows": int((ohlc <= 0).any(axis=1).sum()),
        "Negative_Volume_Rows": int((df["Volume"] < 0).sum()),
        "Inconsistent_OHLC_Rows": int(((df["High"] < ohlc[["Open", "Close", "Low"]].max(axis=1)) |
                                        (df["Low"] > ohlc[["Open", "Close", "High"]].min(axis=1))).sum()),
        "Absolute_Close_Return_Over_50pct": int((df["Close"].pct_change().abs() > 0.5).sum()),
        "Source_SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return df, audit


def predict_walk_forward(df: pd.DataFrame, stock: str) -> tuple[pd.DataFrame, list[dict]]:
    boundaries = pd.date_range(START, END, freq="QS-JAN")
    pieces = []
    fold_rows = []
    features_valid = df.loc[:, list(FEATURES)].notna().all(axis=1)
    for fold, (begin, finish) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        next_session = pd.Series(df.index, index=df.index).shift(-1)
        test_mask = (df.index >= begin) & (df.index < finish) & next_session.notna()
        test = df.loc[test_mask].copy()
        if len(test) < 10:
            raise ValueError(f"{stock} fold {fold} has only {len(test)} test rows")
        for horizon in HORIZONS:
            # Keep only labels whose outcome endpoint is strictly before the fold.
            outcome_end = pd.Series(df.index, index=df.index).shift(-(horizon + 1))
            train_mask = ((df.index < begin) & (outcome_end < begin) & features_valid &
                          df[f"target_{horizon}d"].notna())
            train = df.loc[train_mask]
            if len(train) < 252:
                raise ValueError(f"{stock} fold {fold} has insufficient training history")
            target = f"target_{horizon}d"
            y_train = train[target].to_numpy(dtype=int)
            if len(np.unique(y_train)) != 2:
                raise ValueError(f"{stock} fold {fold} has only one training class")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train.loc[:, list(FEATURES)])
            model = XGBClassifier(**XGB_PARAMS)
            model.fit(x_train, y_train)
            name = f"XGBoost {horizon}D"
            predict_mask = test.loc[:, list(FEATURES)].notna().all(axis=1)
            probabilities = model.predict_proba(
                scaler.transform(test.loc[predict_mask, list(FEATURES)]))[:, 1]
            test[name] = np.nan
            test[f"{name}_Probability"] = np.nan
            test.loc[predict_mask, name] = (probabilities >= 0.5).astype(float)
            test.loc[predict_mask, f"{name}_Probability"] = probabilities
            scored = predict_mask & test[target].notna()
            y_test = test.loc[scored, target].to_numpy(dtype=int)
            scored_probabilities = test.loc[scored, f"{name}_Probability"].to_numpy()
            fold_rows.append({
                "Stock": stock, "Fold": fold, "Start": begin.date().isoformat(),
                "End_Exclusive": finish.date().isoformat(), "Model": name,
                "Horizon_Days": horizon, "Observations": len(y_test),
                "Training_Observations": len(train),
                "Training_Last_Date": train.index.max().date().isoformat(),
                "Training_Last_Label_End": outcome_end.loc[train.index].max().date().isoformat(),
                "Test_Decision_Rows": len(test), "Predicted_Rows": int(predict_mask.sum()),
                "AUC": roc_auc_score(y_test, scored_probabilities) if len(np.unique(y_test)) == 2 else np.nan,
                "Accuracy": accuracy_score(y_test, scored_probabilities >= 0.5) if len(y_test) else np.nan,
                "LogLoss": log_loss(y_test, scored_probabilities, labels=[0, 1]) if len(y_test) else np.nan,
            })
            if horizon == 1:
                linear = LogisticRegression(C=0.1, max_iter=500, random_state=42)
                linear.fit(x_train, y_train)
                lr_probabilities = linear.predict_proba(
                    scaler.transform(test.loc[predict_mask, list(FEATURES)]))[:, 1]
                test["Logistic Regression 1D"] = np.nan
                test["Logistic Regression 1D_Probability"] = np.nan
                test.loc[predict_mask, "Logistic Regression 1D"] = (lr_probabilities >= 0.5).astype(float)
                test.loc[predict_mask, "Logistic Regression 1D_Probability"] = lr_probabilities
                scored_lr = predict_mask & test["target_1d"].notna()
                y_lr = test.loc[scored_lr, "target_1d"].to_numpy(dtype=int)
                scored_lr_probabilities = test.loc[scored_lr, "Logistic Regression 1D_Probability"].to_numpy()
                fold_rows.append({
                    "Stock": stock, "Fold": fold, "Start": begin.date().isoformat(),
                    "End_Exclusive": finish.date().isoformat(),
                    "Model": "Logistic Regression 1D", "Horizon_Days": 1,
                    "Observations": len(y_lr),
                    "Training_Observations": len(train),
                    "Training_Last_Date": train.index.max().date().isoformat(),
                    "Training_Last_Label_End": outcome_end.loc[train.index].max().date().isoformat(),
                    "Test_Decision_Rows": len(test), "Predicted_Rows": int(predict_mask.sum()),
                    "AUC": roc_auc_score(y_lr, scored_lr_probabilities) if len(np.unique(y_lr)) == 2 else np.nan,
                    "Accuracy": accuracy_score(y_lr, scored_lr_probabilities >= 0.5) if len(y_lr) else np.nan,
                    "LogLoss": log_loss(y_lr, scored_lr_probabilities, labels=[0, 1]) if len(y_lr) else np.nan,
                })
        pieces.append(test)
    oos = pd.concat(pieces).sort_index()
    if oos.index.has_duplicates:
        raise ValueError(f"Overlapping test folds for {stock}")
    return oos, fold_rows


def backtest(df: pd.DataFrame, oos: pd.DataFrame, signals: np.ndarray,
             min_hold_days: int, fee_per_leg: float = FEE_PER_LEG) -> tuple[dict, pd.DataFrame]:
    positions = df.index.get_indexer(oos.index)
    if (positions < 0).any() or (positions + 1 >= len(df)).any():
        raise ValueError("An execution date is missing")
    opens = df["Open"].to_numpy()[positions + 1]
    closes = df["Close"].to_numpy()[positions + 1]
    cash, shares = INITIAL_CASH, 0.0
    equity = [INITIAL_CASH]
    entries = exits = invested_days = 0
    entry_index = -1
    entry_capital = 0.0
    trade_returns = []
    daily_rows = []
    for i, (signal, open_price, close_price) in enumerate(zip(signals, opens, closes)):
        day_entries = day_exits = 0
        entry_fee = exit_fee = 0.0
        if pd.notna(signal):
            want_long = bool(signal)
            if shares == 0 and want_long:
                entry_capital = cash
                shares = cash / (open_price * (1 + fee_per_leg))
                entry_fee = shares * open_price * fee_per_leg
                cash = 0.0
                entries += 1
                day_entries = 1
                entry_index = i
            elif shares > 0 and not want_long and i - entry_index >= min_hold_days:
                exit_fee = shares * open_price * fee_per_leg
                cash = shares * open_price * (1 - fee_per_leg)
                trade_returns.append(cash / entry_capital - 1)
                shares = 0.0
                exits += 1
                day_exits = 1
        if shares > 0:
            invested_days += 1
        else:
            cash *= 1 + DAILY_CASH_RATE
        equity.append(cash + shares * close_price)
        daily_rows.append({"Decision_Date": oos.index[i], "Date": df.index[positions[i] + 1],
                           "Equity": equity[-1], "Invested": int(shares > 0),
                           "Entries": day_entries, "Trade_Legs": day_entries + day_exits,
                           "Signal": np.nan if pd.isna(signal) else int(bool(signal)),
                           "Execution_Open": open_price, "Mark_Close": close_price,
                           "Entry_Fee": entry_fee, "Exit_Fee": exit_fee,
                           "Cash_Balance": cash, "Shares": shares})
    if shares > 0:
        # All strategies are fully liquidated at the final test close.
        cash = shares * closes[-1] * (1 - fee_per_leg)
        daily_rows[-1]["Exit_Fee"] += shares * closes[-1] * fee_per_leg
        daily_rows[-1]["Cash_Balance"] = cash
        daily_rows[-1]["Shares"] = 0.0
        trade_returns.append(cash / entry_capital - 1)
        equity[-1] = cash
        exits += 1
        daily_rows[-1]["Equity"] = cash
        daily_rows[-1]["Trade_Legs"] += 1
    equity = np.asarray(equity, dtype=float)
    returns = equity[1:] / equity[:-1] - 1
    daily = pd.DataFrame(daily_rows)
    daily["Daily_Return"] = returns
    excess = returns - DAILY_CASH_RATE
    volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    drawdown = equity / np.maximum.accumulate(equity) - 1
    years = len(returns) / 252
    total_return = equity[-1] / INITIAL_CASH - 1
    metrics = {
        "Total_Return_Pct": 100 * total_return,
        "Annual_Return_Pct": 100 * ((1 + total_return) ** (1 / years) - 1),
        "Sharpe": np.sqrt(252) * np.mean(excess) / volatility if volatility > 0 else np.nan,
        "Sortino": np.sqrt(252) * np.mean(excess) / downside if downside > 0 else np.nan,
        "Max_Drawdown_Pct": 100 * np.min(drawdown),
        "Annual_Volatility_Pct": 100 * np.sqrt(252) * volatility,
        "Exposure_Pct": 100 * invested_days / len(returns),
        "Entries": entries, "Exits": exits, "Trade_Legs": entries + exits,
        "Win_Rate_Pct": 100 * np.mean(np.asarray(trade_returns) > 0) if trade_returns else np.nan,
        "Test_Days": len(returns),
    }
    return metrics, daily


def run(output: Path, stocks: list[str] | None = None) -> None:
    files = source_files()
    if stocks is not None:
        wanted = set(stocks)
        files = [p for p in files if p.name.removesuffix("_features.csv") in wanted]
        if len(files) != len(wanted):
            raise ValueError("One or more requested stocks lack processed feature files")
    elif len(files) != 50:
        raise ValueError(f"Expected 50 processed stock files, found {len(files)}")
    stock_rows, fold_rows, data_rows, daily_rows = [], [], [], []
    for number, path in enumerate(files, start=1):
        stock = path.name.removesuffix("_features.csv")
        df, audit = load_stock(path)
        oos, folds = predict_walk_forward(df, stock)
        data_rows.append(audit)
        fold_rows.extend(folds)
        for strategy in STRATEGIES:
            signals = np.ones(len(oos), dtype=bool) if strategy == "Buy & Hold" else oos[strategy].to_numpy()
            hold = 3 if strategy == "XGBoost 3D" else 5 if strategy == "XGBoost 5D" else 1
            metrics, daily = backtest(df, oos, signals, hold)
            _, gross_daily = backtest(df, oos, signals, hold, fee_per_leg=0.0)
            daily["Gross_Daily_Return"] = gross_daily["Daily_Return"].to_numpy()
            daily.insert(1, "Stock", stock)
            daily.insert(2, "Strategy", strategy)
            daily_rows.append(daily)
            stock_rows.append({"Stock": stock, "Strategy": strategy, "Horizon_Days": hold,
                               "First_Execution": df.index[df.index.get_indexer(oos.index)[0] + 1].date().isoformat(),
                               "Last_Execution": df.index[df.index.get_indexer(oos.index)[-1] + 1].date().isoformat(),
                               **metrics})
        print(f"[{number}/{len(files)}] {stock}: {len(oos)} out-of-sample days", flush=True)
    stock_df = pd.DataFrame(stock_rows)
    if len(stock_df) != len(files) * len(STRATEGIES):
        raise RuntimeError("Incomplete strategy results")
    fold_df = pd.DataFrame(fold_rows)
    summary = stock_df.groupby("Strategy", sort=False).agg(
        Stocks=("Stock", "nunique"), Mean_Return_Pct=("Total_Return_Pct", "mean"),
        Median_Return_Pct=("Total_Return_Pct", "median"),
        Mean_Annual_Return_Pct=("Annual_Return_Pct", "mean"),
        Mean_Sharpe=("Sharpe", "mean"), Mean_Max_Drawdown_Pct=("Max_Drawdown_Pct", "mean"),
        Mean_Entries=("Entries", "mean"), Mean_Trade_Legs=("Trade_Legs", "mean"),
        Mean_Win_Rate_Pct=("Win_Rate_Pct", "mean"),
        Mean_Exposure_Pct=("Exposure_Pct", "mean"),
    ).reset_index()
    predictive = fold_df.groupby("Model").agg(
        Mean_Fold_AUC=("AUC", "mean"), Mean_Fold_Accuracy=("Accuracy", "mean"),
        Mean_Fold_LogLoss=("LogLoss", "mean"), Folds=("Fold", "count"),
    ).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    stock_df.to_csv(output / "stock_metrics.csv", index=False)
    summary.to_csv(output / "strategy_summary.csv", index=False)
    fold_df.to_csv(output / "fold_diagnostics.csv", index=False)
    predictive.to_csv(output / "model_summary.csv", index=False)
    pd.DataFrame(data_rows).to_csv(output / "data_audit.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(output / "daily_strategy_returns.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stock_count": len(files), "strategy_count": len(STRATEGIES),
        "expected_rows": len(files) * len(STRATEGIES),
        "oos_start": START.date().isoformat(), "oos_end_exclusive": END.date().isoformat(),
        "fee_per_leg": FEE_PER_LEG, "annual_cash_rate": 0.07,
        "initial_cash": INITIAL_CASH, "features": FEATURES,
        "bollinger_definition": "20-session trailing close, sample standard deviation (ddof=1), ratio bandwidth; recomputed for every stock",
        "test_status": "exploratory reused 2022-2025 period, not untouched holdout",
        "xgboost_parameters": XGB_PARAMS,
        "logistic_regression_parameters": {"C": 0.1, "max_iter": 500, "random_state": 42},
        "minimum_hold_days": {"XGBoost 1D": 1, "XGBoost 3D": 3, "XGBoost 5D": 5},
        "early_stopping": False,
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nStrategy summary (means across stocks):")
    print(summary.round(3).to_string(index=False))
    print("\nPredictive diagnostics (means across folds):")
    print(predictive.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stocks", nargs="+", help="Optional subset for a smoke run")
    args = parser.parse_args()
    run(args.output_dir, args.stocks)
