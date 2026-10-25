"""Fetch and rebuild the 50-stock 2018-2025 panel from pinned yfinance.

The Yahoo download is a new, documented acquisition. It need not reproduce the
unlogged original byte for byte. Models are not run by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/yfinance_rebuild"
START = "2018-01-01"
END_EXCLUSIVE = "2025-12-31"
EXPECTED_VERSION = "1.7.0"
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbols() -> list[str]:
    path = ROOT / "config/symbols.txt"
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate symbols in config/symbols.txt")
    return sorted(names)


def flatten(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if ticker in set(frame.columns.get_level_values(level)):
                frame = frame.xs(ticker, axis=1, level=level)
                break
    frame.columns = [str(x) for x in frame.columns]
    return frame


def validate_raw(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{symbol}: empty download")
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: missing raw fields {sorted(missing)}")
    frame = frame.loc[:, list(FIELDS) + [x for x in ("Dividends", "Stock Splits") if x in frame]].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "Date"
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate dates")
    if frame[list(FIELDS)].isna().any().any():
        raise ValueError(f"{symbol}: missing OHLCV or adjusted close")
    if (frame[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: nonpositive price")
    if (frame["Volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume")
    if ((frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)) |
            (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError(f"{symbol}: invalid raw OHLC relationship")
    return frame


def features(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    factor = raw["Adj Close"] / raw["Close"]
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ValueError(f"{symbol}: invalid adjustment factor")
    out = pd.DataFrame(index=raw.index)
    out["Raw_Close"] = raw["Close"]
    out["Adjustment_Factor"] = factor
    for column in ("Open", "High", "Low", "Close"):
        out[column] = raw[column] * factor
    out["Adj Close"] = raw["Adj Close"]
    out["Volume"] = raw["Volume"]
    close = out["Close"]
    out["Return_1d"] = close.pct_change(fill_method=None)
    out["Volume_Change"] = out["Volume"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    for period in (10, 20, 50, 200):
        out[f"SMA_{period}"] = close.rolling(period, min_periods=period).mean()
    out["Close_vs_SMA20"] = close / out["SMA_20"] - 1
    out["Close_vs_SMA50"] = close / out["SMA_50"] - 1
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["RSI_14"] = 100 - 100 / (1 + up / down)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    out["MACD_12_26_9"] = fast - slow
    out["MACDs_12_26_9"] = out["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
    out["MACDh_12_26_9"] = out["MACD_12_26_9"] - out["MACDs_12_26_9"]
    middle = out["SMA_20"]
    spread = 2 * close.rolling(20, min_periods=20).std(ddof=1)
    out["BBM_20_2.0"] = middle
    out["BBU_20_2.0"] = middle + spread
    out["BBL_20_2.0"] = middle - spread
    out["BBB_20_2.0"] = 2 * spread / middle
    out["BBP_20_2.0"] = (close - out["BBL_20_2.0"]) / (2 * spread)
    out["Stock"] = symbol
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch fresh Yahoo data; otherwise rebuild from archived raw CSVs")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-run symbol count; zero means all 50")
    args = parser.parse_args()
    names = symbols()
    if len(names) != 50:
        raise ValueError(f"Expected 50 original symbols, found {len(names)}")
    if args.limit:
        names = names[:args.limit]
    raw_dir, processed_dir = BASE / "raw", BASE / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    prior_manifest = None
    if not args.download:
        manifest_path = BASE / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Archived rebuild requires the original acquisition manifest")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    yf = None
    if args.download:
        # Standard installation uses requirements.txt. The local .deps target
        # is a fallback; numpy/pandas were imported before that path is added.
        try:
            import yfinance as yf  # type: ignore[no-redef]
        except ModuleNotFoundError:
            sys.path.insert(0, str(ROOT / ".deps"))
            import yfinance as yf  # type: ignore[no-redef]
        if yf.__version__ != EXPECTED_VERSION:
            raise RuntimeError(f"Expected yfinance {EXPECTED_VERSION}, got {yf.__version__}")
    records = []
    for i, symbol in enumerate(names, 1):
        path = raw_dir / f"{symbol}.csv"
        if args.download:
            ticker = f"{symbol}.NS"
            last_error = None
            for attempt in range(3):
                try:
                    frame = flatten(yf.download(
                        ticker, start=START, end=END_EXCLUSIVE, interval="1d",
                        auto_adjust=False, actions=True, repair=False,
                        threads=False, progress=False, timeout=30), ticker)
                    frame = validate_raw(frame, symbol)
                    frame.to_csv(path, float_format="%.15g")
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"{symbol}: three download attempts failed") from last_error
        elif not path.exists():
            raise FileNotFoundError(f"Missing archived raw file: {path}")
        frame = validate_raw(pd.read_csv(path, index_col="Date", parse_dates=True), symbol)
        output_path = processed_dir / f"{symbol}_features.csv"
        features(frame, symbol).to_csv(output_path, float_format="%.15g")
        records.append({
            "symbol": symbol, "ticker": f"{symbol}.NS", "rows": len(frame),
            "first_date": frame.index.min().date().isoformat(),
            "last_date": frame.index.max().date().isoformat(),
            "raw_sha256": file_hash(path), "processed_sha256": file_hash(output_path),
            "dividend_events": int((frame.get("Dividends", pd.Series(0, index=frame.index)) != 0).sum()),
            "split_events": int((frame.get("Stock Splits", pd.Series(0, index=frame.index)) != 0).sum()),
        })
        print(f"[{i}/{len(names)}] {symbol}: {len(frame)} rows", flush=True)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_mode": "fresh_yfinance" if args.download else "archived_raw_rebuild",
        "vendor": "Yahoo Finance via yfinance" if args.download else "Archived CSVs from prior documented fresh_yfinance retrieval",
        "yfinance_version": EXPECTED_VERSION,
        "start_inclusive": START, "end_exclusive": END_EXCLUSIVE,
        "download_arguments": {"interval": "1d", "auto_adjust": False,
                               "actions": True, "repair": False, "threads": False},
        "transformation": "Multiply each raw OHLC field by same-day Adj Close / raw Close; retain raw volume. Recompute all 17 features from adjusted OHLCV.",
        "universe": "Fixed 50 symbols from config/symbols.txt; not point-in-time Nifty membership",
        "symbols": records,
    }
    if args.download:
        (BASE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {BASE / 'manifest.json'}")
    else:
        previous = {item["symbol"]: item for item in prior_manifest["symbols"]}
        for item in records:
            old = previous[item["symbol"]]
            if (item["raw_sha256"] != old["raw_sha256"] or
                    item["processed_sha256"] != old["processed_sha256"]):
                raise ValueError(f"{item['symbol']}: archived rebuild hash differs from acquisition manifest")
        print(f"Rebuilt {len(records)} symbols without changing the acquisition manifest")


if __name__ == "__main__":
    main()
