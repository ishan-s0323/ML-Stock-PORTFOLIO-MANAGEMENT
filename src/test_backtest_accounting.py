"""Hand-calculated checks for next-open backtest accounting."""

import ast
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(__file__).with_name("run_50_stock_benchmark.py")
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
names = {"INITIAL_CASH", "FEE_PER_LEG", "DAILY_CASH_RATE", "FEATURES", "HORIZONS",
         "START", "END", "XGB_PARAMS"}
selected = [node for node in tree.body if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
selected += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "backtest"]
selected += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_stock"]
selected += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict_walk_forward"]


class IdentityScaler:
    def fit_transform(self, values):
        return np.asarray(values, dtype=float)

    def transform(self, values):
        return np.asarray(values, dtype=float)


class ConstantClassifier:
    def __init__(self, **kwargs):
        pass

    def fit(self, values, labels):
        return self

    def predict_proba(self, values):
        positive = np.full(len(values), 0.6)
        return np.column_stack([1 - positive, positive])


def simple_accuracy(labels, predictions):
    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def simple_log_loss(y_true, y_pred, labels=None):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def simple_auc(labels, probabilities):
    return 0.5


namespace = {"np": np, "pd": pd, "Path": Path, "hashlib": hashlib,
             "StandardScaler": IdentityScaler, "XGBClassifier": ConstantClassifier,
             "LogisticRegression": ConstantClassifier, "accuracy_score": simple_accuracy,
             "log_loss": simple_log_loss, "roc_auc_score": simple_auc}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
backtest = namespace["backtest"]
load_stock = namespace["load_stock"]
predict_walk_forward = namespace["predict_walk_forward"]
INITIAL_CASH = namespace["INITIAL_CASH"]
FEE_PER_LEG = namespace["FEE_PER_LEG"]
DAILY_CASH_RATE = namespace["DAILY_CASH_RATE"]


def fixture_inputs():
    # Decision rows are d0, d1, d2; their executions are at d1, d2, d3.
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {"Open": [10.0, 10.0, 12.0, 12.0], "Close": [10.0, 11.0, 12.0, 13.0]},
        index=dates,
    )
    oos = pd.DataFrame(index=dates[:3])
    return df, oos


class BacktestAccountingTests(unittest.TestCase):
    def test_all_cash_accrues_one_daily_rate_per_execution_session(self):
        df, oos = fixture_inputs()
        metrics, daily = backtest(df, oos, np.array([False, False, False]), 1)
        expected = INITIAL_CASH * (1 + DAILY_CASH_RATE) ** 3
        self.assertAlmostEqual(metrics["Total_Return_Pct"], 100 * (expected / INITIAL_CASH - 1), places=9)
        self.assertEqual(daily["Invested"].sum(), 0)

    def test_entry_and_forced_final_close_apply_each_fee_once(self):
        df, oos = fixture_inputs()
        metrics, daily = backtest(df, oos, np.array([True, True, True]), 1)
        shares = INITIAL_CASH / (10.0 * (1 + FEE_PER_LEG))
        expected_final = shares * 13.0 * (1 - FEE_PER_LEG)
        self.assertAlmostEqual(metrics["Total_Return_Pct"], 100 * (expected_final / INITIAL_CASH - 1), places=9)
        self.assertEqual(metrics["Entries"], 1)
        self.assertEqual(metrics["Exits"], 1)
        self.assertEqual(metrics["Trade_Legs"], 2)
        self.assertEqual(daily["Date"].tolist(), list(df.index[1:]))

    def test_exit_executes_at_next_open_and_flat_cash_earns_through_close(self):
        df, oos = fixture_inputs()
        metrics, daily = backtest(df, oos, np.array([True, False, False]), 1)
        shares = INITIAL_CASH / (10.0 * (1 + FEE_PER_LEG))
        proceeds = shares * 12.0 * (1 - FEE_PER_LEG)
        expected_final = proceeds * (1 + DAILY_CASH_RATE) ** 2
        self.assertAlmostEqual(metrics["Total_Return_Pct"], 100 * (expected_final / INITIAL_CASH - 1), places=9)
        self.assertEqual(metrics["Entries"], 1)
        self.assertEqual(metrics["Exits"], 1)
        self.assertEqual(daily["Trade_Legs"].tolist(), [1, 1, 0])

    def test_minimum_holding_period_blocks_early_exit(self):
        df, oos = fixture_inputs()
        metrics, daily = backtest(df, oos, np.array([True, False, False]), 2)
        self.assertEqual(metrics["Entries"], 1)
        # The exit request at d1 is deferred until the d2 decision, then executes
        # at that decision's next open (row d3).
        self.assertEqual(metrics["Exits"], 1)
        self.assertEqual(daily["Invested"].tolist(), [1, 1, 0])

    def test_missing_signal_holds_position_and_keeps_every_session(self):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Open": [10.0] * 5, "Close": [10.0, 10.0, 11.0, 12.0, 12.0]}, index=dates)
        oos = pd.DataFrame(index=dates[:4])
        metrics, daily = backtest(df, oos, np.array([True, np.nan, False, False]), 1)
        self.assertEqual(len(daily), 4)
        self.assertEqual(daily["Date"].tolist(), list(dates[1:]))
        self.assertEqual(daily["Invested"].tolist(), [1, 1, 0, 0])
        self.assertEqual(metrics["Entries"], 1)
        self.assertEqual(metrics["Exits"], 1)

    def test_forward_targets_remain_missing_when_future_open_is_unavailable(self):
        dates = pd.date_range("2025-01-01", periods=12, freq="D")
        prices = np.arange(10.0, 22.0)
        frame = pd.DataFrame({"Date": dates, "Open": prices, "High": prices + 1,
                              "Low": prices - 1, "Close": prices, "Volume": np.ones(12)})
        for feature in namespace["FEATURES"]:
            frame[feature] = 1.0
        with tempfile.TemporaryDirectory(dir=SOURCE.parents[1]) as directory:
            path = Path(directory) / "SAMPLE_features.csv"
            frame.set_index("Date").to_csv(path)
            loaded, _ = load_stock(path)
        self.assertEqual(loaded.loc[dates[0], "target_1d"], 1.0)
        self.assertTrue(loaded.loc[dates[-2:], "target_1d"].isna().all())
        self.assertTrue(loaded.loc[dates[-4:], "target_3d"].isna().all())

    def test_bollinger_features_are_rebuilt_to_one_convention(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        close = np.linspace(100.0, 115.0, 30) + np.sin(np.arange(30))
        frame = pd.DataFrame({"Date": dates, "Open": close, "High": close + 1,
                              "Low": close - 1, "Close": close, "Volume": np.ones(30)})
        for feature in namespace["FEATURES"]:
            frame[feature] = 1.0
        frame["BBB_20_2.0"] = 999.0
        with tempfile.TemporaryDirectory(dir=SOURCE.parents[1]) as directory:
            path = Path(directory) / "SAMPLE_features.csv"
            frame.set_index("Date").to_csv(path)
            loaded, _ = load_stock(path)
        middle = pd.Series(close).rolling(20).mean().iloc[-1]
        deviation = pd.Series(close).rolling(20).std(ddof=1).iloc[-1]
        self.assertAlmostEqual(loaded["BBB_20_2.0"].iloc[-1], 4 * deviation / middle)

    def test_walk_forward_keeps_every_executable_session_and_masks_missing_features(self):
        dates = pd.bdate_range("2018-01-01", "2025-12-30")
        rng = np.random.default_rng(17)
        opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
        frame = pd.DataFrame(index=dates)
        frame["Open"] = opens
        for feature in namespace["FEATURES"]:
            frame[feature] = rng.normal(size=len(dates))
        missing_date = pd.Timestamp("2023-05-10")
        frame.loc[missing_date, "RSI_14"] = np.nan
        for horizon in namespace["HORIZONS"]:
            forward = frame["Open"].shift(-(horizon + 1)) / frame["Open"].shift(-1) - 1
            frame[f"forward_return_{horizon}d"] = forward
            frame[f"target_{horizon}d"] = np.where(forward.notna(), (forward > 0).astype(float), np.nan)
        oos, folds = predict_walk_forward(frame, "SAMPLE")
        expected = dates[(dates >= pd.Timestamp("2022-01-01")) &
                         (dates < pd.Timestamp("2026-01-01"))][:-1]
        self.assertEqual(oos.index.tolist(), expected.tolist())
        self.assertTrue(pd.isna(oos.loc[missing_date, "XGBoost 1D_Probability"]))
        self.assertTrue(pd.notna(oos.loc[missing_date + pd.offsets.BDay(1), "XGBoost 1D_Probability"]))
        self.assertTrue(pd.isna(oos.iloc[-1]["target_5d"]))
        self.assertTrue(pd.notna(oos.iloc[-1]["XGBoost 5D_Probability"]))
        self.assertTrue(all(row["Training_Last_Label_End"] < row["Start"] for row in folds))
        self.assertTrue(all(50 <= row["Test_Decision_Rows"] <= 70 for row in folds))


if __name__ == "__main__":
    unittest.main()
