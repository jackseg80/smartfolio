"""Reproducible Binance/OKX daily market-data consistency validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

CROSS_EXCHANGE_SCHEMA_VERSION = "crypto-forecast-cross-exchange-validation-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode("utf-8")


def _finite_or_none(value: object) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2:
        return None
    result = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return _finite_or_none(result)


def stable_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash acquisition identity while excluding the non-semantic generation timestamp."""
    stable = {key: value for key, value in manifest.items() if key != "generated_at_utc"}
    return _sha256_bytes(_json_bytes(stable))


def _verified_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "acquisition_manifest.json"
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise ValueError(f"Invalid acquisition manifest: {path}")
    return manifest, stable_manifest_sha256(manifest)


def load_binance_daily(root: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    directory = Path(root)
    manifest, manifest_sha256 = _verified_manifest(directory)
    if manifest.get("provider") != "binance_spot_public_market_data":
        raise ValueError("The Binance input has unexpected provider provenance")
    frames: dict[str, pd.DataFrame] = {}
    for item in manifest["inputs"]:
        symbol = str(item["symbol"])
        path = directory / str(item["ohlcv_file"])
        if file_sha256(path) != item["ohlcv_file_sha256"]:
            raise ValueError(f"Binance OHLCV hash mismatch for {symbol}")
        frame = pd.read_csv(path, usecols=["date", "close", "quote_asset_volume"])
        frame = frame.rename(columns={"quote_asset_volume": "quote_volume"})
        frames[symbol] = _validate_daily_frame(frame, f"Binance {symbol}")
    return frames, {
        "provider": manifest["provider"],
        "artifact_id": manifest["artifact_id"],
        "manifest_identity_sha256": manifest_sha256,
    }


def load_okx_daily(root: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    directory = Path(root)
    manifest, manifest_sha256 = _verified_manifest(directory)
    if manifest.get("provider") != "okx_spot_public_market_data":
        raise ValueError("The OKX input has unexpected provider provenance")
    frames: dict[str, pd.DataFrame] = {}
    for item in manifest["inputs"]:
        symbol = str(item["symbol"])
        path = directory / str(item["ohlcv_file"])
        if file_sha256(path) != item["ohlcv_file_sha256"]:
            raise ValueError(f"OKX OHLCV hash mismatch for {symbol}")
        frame = pd.read_csv(path, usecols=["date", "close", "quote_volume", "confirmed"])
        if not (frame["confirmed"] == 1).all():
            raise ValueError(f"OKX contains unconfirmed daily candles for {symbol}")
        frames[symbol] = _validate_daily_frame(frame.drop(columns="confirmed"), f"OKX {symbol}")
    return frames, {
        "provider": manifest["provider"],
        "artifact_id": manifest["artifact_id"],
        "manifest_identity_sha256": manifest_sha256,
    }


def _validate_daily_frame(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result["date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {context}")
    for column in ("close", "quote_volume"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"Non-finite {column} in {context}")
    if (result["close"] <= 0).any() or (result["quote_volume"] < 0).any():
        raise ValueError(f"Invalid price or volume in {context}")
    return result.sort_values("date").set_index("date")


def compare_asset(
    symbol: str,
    binance: pd.DataFrame,
    okx: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    aligned = binance.join(okx, how="inner", lsuffix="_binance", rsuffix="_okx")
    if aligned.empty:
        raise ValueError(f"No common observations for {symbol}")
    aligned["return_binance"] = aligned["close_binance"].pct_change(fill_method=None)
    aligned["return_okx"] = aligned["close_okx"].pct_change(fill_method=None)
    aligned["close_difference_bps"] = (
        aligned["close_okx"] / aligned["close_binance"] - 1.0
    ) * 10_000.0
    aligned["absolute_close_difference_bps"] = aligned["close_difference_bps"].abs()
    aligned["return_difference_bps"] = (
        aligned["return_okx"] - aligned["return_binance"]
    ) * 10_000.0
    aligned["absolute_return_difference_bps"] = aligned["return_difference_bps"].abs()

    volume_correlations: dict[str, float | None] = {}
    for horizon in config["volume_change_horizons_days"]:
        days = int(horizon)
        left = aligned["quote_volume_binance"].pct_change(days, fill_method=None)
        right = aligned["quote_volume_okx"].pct_change(days, fill_method=None)
        volume_correlations[f"quote_volume_change_{days}d_correlation"] = _correlation(left, right)

    absolute_close = aligned["absolute_close_difference_bps"]
    max_index = absolute_close.idxmax()
    daily_return_correlation = _correlation(aligned["return_binance"], aligned["return_okx"])
    median_close = float(absolute_close.median())
    p95_close = float(absolute_close.quantile(0.95))
    checks = {
        "minimum_common_observations": len(aligned) >= int(config["minimum_common_observations"]),
        "minimum_daily_return_correlation": daily_return_correlation is not None
        and daily_return_correlation >= float(config["minimum_daily_return_correlation"]),
        "maximum_median_absolute_close_difference_bps": median_close
        <= float(config["maximum_median_absolute_close_difference_bps"]),
        "maximum_p95_absolute_close_difference_bps": p95_close
        <= float(config["maximum_p95_absolute_close_difference_bps"]),
    }
    summary = {
        "symbol": symbol,
        "common_observations": len(aligned),
        "common_start": aligned.index.min().date().isoformat(),
        "common_end": aligned.index.max().date().isoformat(),
        "binance_observations_outside_common_period": int(len(binance) - len(aligned)),
        "okx_observations_outside_common_period": int(len(okx) - len(aligned)),
        "daily_return_correlation": daily_return_correlation,
        "median_absolute_close_difference_bps": median_close,
        "p95_absolute_close_difference_bps": p95_close,
        "maximum_absolute_close_difference_bps": float(absolute_close.max()),
        "maximum_absolute_close_difference_date": max_index.date().isoformat(),
        "median_absolute_return_difference_bps": _finite_or_none(
            aligned["absolute_return_difference_bps"].median()
        ),
        **volume_correlations,
        "quality_checks": checks,
        "price_consistency_passed": all(checks.values()),
    }
    aligned.index.name = "date"
    output = aligned.reset_index()
    output.insert(1, "symbol", symbol)
    return summary, output


def _aligned_csv_bytes(frames: Sequence[pd.DataFrame]) -> bytes:
    combined = pd.concat(frames, ignore_index=True)
    columns = [
        "date",
        "symbol",
        "close_binance",
        "close_okx",
        "quote_volume_binance",
        "quote_volume_okx",
        "return_binance",
        "return_okx",
        "close_difference_bps",
        "return_difference_bps",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in combined[columns].itertuples(index=False, name=None):
        serialized = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                serialized.append(value.date().isoformat())
            elif pd.isna(value):
                serialized.append("")
            elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise ValueError("Aligned daily output contains an infinite value")
            else:
                serialized.append(value)
        writer.writerow(serialized)
    return buffer.getvalue().encode("utf-8")


def run_validation(
    *,
    binance_root: str | Path,
    okx_root: str | Path,
    config: Mapping[str, Any],
    config_sha256: str,
    validation_code_sha256: str,
    output_root: str | Path,
) -> Path:
    if config.get("schema_version") != CROSS_EXCHANGE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported validation schema: {config.get('schema_version')}")
    binance, binance_identity = load_binance_daily(binance_root)
    okx, okx_identity = load_okx_daily(okx_root)
    symbols = sorted(set(binance) & set(okx))
    if not symbols:
        raise ValueError("No common symbols between Binance and OKX")
    summaries = []
    aligned_frames = []
    for symbol in symbols:
        summary, aligned = compare_asset(symbol, binance[symbol], okx[symbol], config=config)
        summaries.append(summary)
        aligned_frames.append(aligned)

    aligned_payload = _aligned_csv_bytes(aligned_frames)
    identity = {
        "schema_version": CROSS_EXCHANGE_SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "validation_code_sha256": validation_code_sha256,
        "binance_input": binance_identity,
        "okx_input": okx_identity,
        "symbols": symbols,
        "summary": summaries,
    }
    identity_sha256 = _sha256_bytes(_json_bytes(identity))
    artifact_id = f"{CROSS_EXCHANGE_SCHEMA_VERSION}-{identity_sha256[:16]}"
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final_directory = root / artifact_id
    if final_directory.exists():
        raise FileExistsError(f"Validation artifact already exists: {final_directory}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=root) as temporary:
        temporary_directory = Path(temporary)
        aligned_name = "aligned_daily.csv"
        (temporary_directory / aligned_name).write_bytes(aligned_payload)
        results = {
            **identity,
            "artifact_id": artifact_id,
            "validation_identity_sha256": identity_sha256,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "aligned_daily_file": aligned_name,
            "aligned_daily_sha256": _sha256_bytes(aligned_payload),
            "all_symbols_passed_price_consistency": all(
                item["price_consistency_passed"] for item in summaries
            ),
            "model_selection_performed": False,
            "production_configuration_changed": False,
            "real_orders_created": False,
        }
        (temporary_directory / "results.json").write_bytes(_json_bytes(results, pretty=True))
        Path(temporary).replace(final_directory)
    return final_directory
