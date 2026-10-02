"""Explicit acquisition of public BTC reference prices for retrospective cycles only."""

import hashlib
import io
import json
from pathlib import Path
import httpx
import numpy as np
import pandas as pd


def main():
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    response = httpx.get(url, timeout=45, follow_redirects=True)
    response.raise_for_status()
    raw = pd.read_csv(io.BytesIO(response.content), usecols=["time", "PriceUSD"])
    dates = pd.to_datetime(raw.time, utc=True).dt.normalize()
    price = pd.to_numeric(raw.PriceUSD, errors="coerce")
    frame = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "close": price})
    frame = frame[(dates < pd.Timestamp.now(tz="UTC").normalize()) & price.notna()].copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True))
    if (
        index.has_duplicates
        or not index.is_monotonic_increasing
        or not ((index[1:] - index[:-1]).days == 1).all()
        or not np.isfinite(frame.close).all()
        or (frame.close <= 0).any()
    ):
        raise ValueError("Reference history is not consecutive valid daily observations")
    directory = Path("data/ml_verified/cycles")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "BTC.csv"
    frame.to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "file": "BTC.csv",
        "sha256": sha,
        "dataset_id": "coinmetrics-btc-reference-" + sha[:16],
        "provider": "Coin Metrics Community / PriceUSD daily reference rate",
        "source_url": url,
        "raw_response_sha256": hashlib.sha256(response.content).hexdigest(),
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "Vendor USD reference rate, distinct from Binance USDT exchange closes. Retrospective historical cycle comparison only; not used to train or infer volatility forecasts.",
        "license": "CC BY-NC 4.0",
        "attribution": "Coin Metrics Community https://github.com/coinmetrics/data",
    }
    (directory / "BTC.json").write_text(json.dumps(receipt, indent=2))
    print(
        json.dumps(
            {
                "first": frame.date.iloc[0],
                "last": frame.date.iloc[-1],
                "rows": len(frame),
                "provider": receipt["provider"],
            }
        )
    )


if __name__ == "__main__":
    main()
