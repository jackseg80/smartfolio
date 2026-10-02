"""Append-only public observations; immutable evaluation datasets are never rewritten."""

import hashlib
import io
import json
import os
import time
from pathlib import Path
from functools import lru_cache
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd
from filelock import FileLock


_last_attempt = {}


def extend_crypto(root, asset, close, receipt, now):
    if receipt.get("provider") != "binance_spot_public_market_data":
        return close, receipt
    directory = Path(root) / "cache/ml_observations/crypto"
    directory.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(now).normalize()
    with FileLock(str(directory / (asset + ".lock")), timeout=30):
        metadata = directory / (asset + ".json")
        frame = pd.DataFrame({"date": close.index.strftime("%Y-%m-%d"), "close": close.to_numpy()})
        current = dict(receipt)
        if metadata.exists():
            cached = json.loads(metadata.read_text())
            path = (directory / cached["file"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Unsafe observation receipt path")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != cached["sha256"]:
                raise ValueError("Observation receipt hash mismatch")
            if cached["evaluation_sha256"] == receipt["sha256"]:
                candidate = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
                prefix = candidate.iloc[: len(frame)]
                if not prefix.equals(frame):
                    raise ValueError("Observed extension changes the evaluated historical prefix")
                frame, current = candidate, cached
        existing = _series(frame)
        if existing.index[-1] >= end:
            raise ValueError("Observation extension includes incomplete closes")
        start = pd.Timestamp(frame.date.iloc[-1], tz="UTC") + pd.Timedelta(days=1)
        if start < end:
            # Bounded pagination: an abandoned cache must not trigger unlimited HTTP work.
            if (end - start).days > 1000:
                raise ValueError(
                    "Observation extension exceeds 1000 days; explicit acquisition is required"
                )
            key = (str(directory.resolve()), asset, receipt["sha256"], end.isoformat())
            if time.monotonic() - _last_attempt.get(key, -float("inf")) < 60:
                return _series(frame), current
            _last_attempt[key] = time.monotonic()
            try:
                response = httpx.get(
                    "https://data-api.binance.vision/api/v3/klines",
                    params={
                        "symbol": asset + "USDT",
                        "interval": "1d",
                        "startTime": int(start.timestamp() * 1000),
                        "endTime": int(end.timestamp() * 1000) - 1,
                        "limit": 1000,
                    },
                    timeout=8,
                )
                response.raise_for_status()
            except httpx.HTTPError:
                # Preserve the actual last observation; the caller still enforces freshness.
                return _series(frame), current
            rows = response.json()
            additions = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp(r[0], unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                        "close": float(r[4]),
                    }
                    for r in rows
                    if r[6] < int(end.timestamp() * 1000)
                ]
            )
            if additions.empty:
                return _series(frame), current
            extension = _series(additions)
            expected = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
            if (
                not extension.index.equals(expected)
                or not np.isfinite(extension).all()
                or (extension <= 0).any()
            ):
                raise ValueError("Provider returned incomplete or invalid daily observations")
            frame = pd.concat([frame, additions], ignore_index=True)
            content = frame.to_csv(index=False).encode()
            sha = hashlib.sha256(content).hexdigest()
            filename = asset + "-" + sha[:16] + ".csv"
            (directory / filename).write_bytes(content)
            raw_sha = hashlib.sha256(response.content).hexdigest()
            raw_file = asset + "-vendor-" + raw_sha[:16] + ".json"
            (directory / raw_file).write_bytes(response.content)
            current = {
                **receipt,
                "file": filename,
                "sha256": sha,
                "dataset_id": f"verified-live-crypto-{asset}-{sha[:16]}",
                "evaluation_dataset_id": receipt["dataset_id"],
                "evaluation_sha256": receipt["sha256"],
                "fetched_at": pd.Timestamp(now).isoformat(),
                "raw_response_sha256": raw_sha,
                "raw_response_file": raw_file,
            }
            temporary = metadata.with_suffix("." + uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(current, indent=2))
            os.replace(temporary, metadata)
        result = _series(frame)
        if (
            result.index.has_duplicates
            or not result.index.is_monotonic_increasing
            or not ((result.index[1:] - result.index[:-1]).days == 1).all()
        ):
            raise ValueError("Observation extension is not consecutive daily history")
        if not np.isfinite(result).all() or (result <= 0).any() or result.index[-1] >= end:
            raise ValueError("Observation extension contains invalid or incomplete closes")
        return result, current


def _series(frame):
    result = pd.Series(
        frame.close.to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True)),
        name="close",
    )
    if (
        result.empty
        or result.index.has_duplicates
        or not result.index.is_monotonic_increasing
        or not ((result.index[1:] - result.index[:-1]).days == 1).all()
        or not np.isfinite(result).all()
        or (result <= 0).any()
    ):
        raise ValueError("Provider returned incomplete or invalid daily observations")
    return result


@lru_cache(maxsize=64)
def completed_stock_sessions(calendar_name, start_date, end_date):
    """Cache only the public schedule, keyed by venue and exact date bounds.

    No prices, portfolio, validation or predictions are cached here. A new UTC
    day uses a different key; holidays and session membership remain unchanged.
    """
    import exchange_calendars as xcals

    calendar = xcals.get_calendar(calendar_name, start=start_date, end=end_date)
    sessions = pd.DatetimeIndex(calendar.sessions)
    if sessions.tz is None:
        sessions = sessions.tz_localize("UTC")
    return sessions[sessions < pd.Timestamp(end_date, tz="UTC")]


def extend_stocks(root, asset, close, receipt, now):
    """Append complete exchange sessions, rejecting vendor adjustment revisions."""
    if receipt.get("provider") != "Yahoo Finance via yfinance":
        return close, receipt
    import yfinance as yf

    directory = Path(root) / "cache/ml_observations/stocks"
    directory.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(now).normalize()
    sessions = completed_stock_sessions(
        receipt["exchange_calendar"], close.index[0].date(), end.date()
    )
    with FileLock(str(directory / (asset + ".lock")), timeout=30):
        metadata = directory / (asset + ".json")
        frame = pd.DataFrame({"date": close.index.strftime("%Y-%m-%d"), "close": close.to_numpy()})
        current = dict(receipt)
        if metadata.exists():
            cached = json.loads(metadata.read_text())
            path = (directory / cached["file"]).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != cached["sha256"]
            ):
                raise ValueError("Invalid stock observation receipt")
            if cached["evaluation_sha256"] == receipt["sha256"]:
                candidate = pd.read_csv(path, float_precision="round_trip")
                if not candidate.iloc[: len(frame)].equals(frame):
                    raise ValueError("Stock extension changes the evaluated historical prefix")
                frame, current = candidate, cached
        series = pd.Series(
            frame.close.to_numpy(),
            index=pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True)),
            name="close",
        )
        if (
            not series.index.equals(sessions[sessions <= series.index[-1]])
            or not np.isfinite(series).all()
            or (series <= 0).any()
        ):
            raise ValueError("Stock extension is not complete valid exchange sessions")
        expected = sessions[sessions > series.index[-1]]
        if expected.empty:
            return series, current
        key = (str(directory.resolve()), asset, receipt["sha256"], end.isoformat())
        if time.monotonic() - _last_attempt.get(key, -float("inf")) < 60:
            return series, current
        _last_attempt[key] = time.monotonic()
        try:
            data = yf.download(
                asset,
                start=(series.index[-1] - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                actions=True,
                progress=False,
                threads=False,
                timeout=8,
            )
        except Exception:
            return series, current
        if data.empty:
            return series, current
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        downloaded = pd.Series(
            data["Close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(data.index, utc=True)).normalize(),
            name="close",
        )
        overlap = downloaded.index.intersection(series.index)
        if overlap.empty or not np.allclose(
            downloaded.loc[overlap], series.loc[overlap], rtol=1e-10, atol=1e-8, equal_nan=False
        ):
            raise ValueError(
                "Vendor revised adjusted historical prices; explicit dataset acquisition and reevaluation are required"
            )
        additions = downloaded[downloaded.index > series.index[-1]]
        if (
            not additions.index.equals(expected)
            or not np.isfinite(additions).all()
            or (additions <= 0).any()
        ):
            raise ValueError("Provider returned missing or invalid completed exchange sessions")
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {"date": additions.index.strftime("%Y-%m-%d"), "close": additions.to_numpy()}
                ),
            ],
            ignore_index=True,
        )
        content = frame.to_csv(index=False).encode()
        sha = hashlib.sha256(content).hexdigest()
        filename = asset + "-" + sha[:16] + ".csv"
        (directory / filename).write_bytes(content)
        # Preserve the SDK's returned observation evidence as a separate public receipt.
        raw = downloaded.to_csv().encode()
        raw_sha = hashlib.sha256(raw).hexdigest()
        raw_file = asset + "-vendor-" + raw_sha[:16] + ".csv"
        (directory / raw_file).write_bytes(raw)
        current = {
            **receipt,
            "file": filename,
            "sha256": sha,
            "dataset_id": f"verified-live-stocks-{asset}-{sha[:16]}",
            "evaluation_dataset_id": receipt["dataset_id"],
            "evaluation_sha256": receipt["sha256"],
            "fetched_at": pd.Timestamp(now).isoformat(),
            "raw_response_sha256": raw_sha,
            "raw_response_file": raw_file,
            "yfinance_version": yf.__version__,
        }
        temporary = metadata.with_suffix("." + uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(current, indent=2))
        os.replace(temporary, metadata)
        return pd.Series(
            frame.close.to_numpy(),
            index=pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True)),
            name="close",
        ), current
