"""Verify archived yfinance raw files reproduce the rebuilt processed panel."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_yfinance_data import BASE, features, file_hash, validate_raw


def main() -> None:
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    checks = []
    for entry in manifest["symbols"]:
        symbol = entry["symbol"]
        raw_path = BASE / "raw" / f"{symbol}.csv"
        processed_path = BASE / "processed" / f"{symbol}_features.csv"
        if file_hash(raw_path) != entry["raw_sha256"] or file_hash(processed_path) != entry["processed_sha256"]:
            raise ValueError(f"{symbol}: file hash differs from acquisition manifest")
        raw = validate_raw(pd.read_csv(raw_path, index_col="Date", parse_dates=True), symbol)
        expected = features(raw, symbol)
        actual = pd.read_csv(processed_path, index_col="Date", parse_dates=True)
        pd.testing.assert_index_equal(actual.index, expected.index)
        numeric = expected.select_dtypes(include="number").columns
        if not np.allclose(actual[numeric], expected[numeric], rtol=1e-10, atol=1e-8, equal_nan=True):
            raise ValueError(f"{symbol}: processed numerical values do not reproduce")
        checks.append({"symbol": symbol, "rows": len(raw), "verified": True})
    output = Path(__file__).resolve().parents[1] / "results/yfinance_rebuild/data_reproduction_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(output, index=False)
    print(f"Verified raw-to-feature reproduction for {len(checks)} stocks")


if __name__ == "__main__":
    main()
