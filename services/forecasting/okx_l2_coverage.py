"""Metadata-only coverage and size survey for OKX historical L2 archives."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import math
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

OKX_L2_COVERAGE_SCHEMA_VERSION = "crypto-forecast-okx-l2-coverage-v1"


class PublicMetadataClient(Protocol):
    async def get_json(self, path: str, params: Mapping[str, object]) -> object: ...


@dataclass(frozen=True)
class ArchiveMetadata:
    date_utc: str
    date_timestamp_ms: int
    instrument: str
    filename: str
    size_mb: str
    url: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _date_ms(date_text: str) -> int:
    value = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _positive_decimal(value: object, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"Invalid {field}: {value!r}")
    return number


def _payload_data(payload: object, context: str) -> list[object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid OKX response for {context}")
    if str(payload.get("code")) != "0":
        raise ValueError(f"OKX returned an error for {context}: {payload.get('msg')!r}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Invalid OKX data for {context}")
    return data


def parse_metadata_response(
    payload: object,
    *,
    date_utc: str,
    instruments: Sequence[str],
    maximum_single_file_mb: Decimal,
) -> tuple[list[ArchiveMetadata], list[str]]:
    data = _payload_data(payload, date_utc)
    if len(data) != 1 or not isinstance(data[0], Mapping):
        raise ValueError(f"Expected one daily metadata group for {date_utc}")
    group = data[0]
    if str(group.get("dateAggrType")) != "daily":
        raise ValueError(f"Unexpected date aggregation for {date_utc}")
    details = group.get("details")
    if not isinstance(details, list):
        raise ValueError(f"Invalid metadata details for {date_utc}")

    expected = set(instruments)
    date_timestamp_ms = _date_ms(date_utc)
    found: dict[str, ArchiveMetadata] = {}
    for detail in details:
        if not isinstance(detail, Mapping):
            raise ValueError(f"Invalid instrument metadata for {date_utc}")
        instrument = str(detail.get("instId", ""))
        if instrument not in expected:
            raise ValueError(f"Unexpected instrument metadata: {instrument!r}")
        if instrument in found:
            raise ValueError(f"Duplicate instrument metadata: {instrument} on {date_utc}")
        files = detail.get("groupDetails")
        if not isinstance(files, list) or len(files) > 1:
            raise ValueError(f"Expected at most one archive for {instrument} on {date_utc}")
        if not files:
            continue
        file_record = files[0]
        if not isinstance(file_record, Mapping):
            raise ValueError(f"Invalid archive metadata for {instrument} on {date_utc}")
        timestamp_value = file_record.get("dateTs", file_record.get("dataTs"))
        if int(str(timestamp_value)) != date_timestamp_ms:
            raise ValueError(f"Archive date mismatch for {instrument} on {date_utc}")
        expected_filename = f"{instrument}-L2orderbook-400lv-{date_utc}.tar.gz"
        filename = str(file_record.get("filename", ""))
        if filename != expected_filename:
            raise ValueError(f"Unexpected filename for {instrument} on {date_utc}")
        url = str(file_record.get("url", ""))
        parsed_url = urlsplit(url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "static.okx.com":
            raise ValueError(f"Unexpected archive URL for {instrument} on {date_utc}")
        size = _positive_decimal(file_record.get("sizeMB"), "archive size")
        if size > maximum_single_file_mb:
            raise ValueError(
                f"Archive size exceeds the frozen limit for {instrument} on {date_utc}"
            )
        found[instrument] = ArchiveMetadata(
            date_utc=date_utc,
            date_timestamp_ms=date_timestamp_ms,
            instrument=instrument,
            filename=filename,
            size_mb=str(size),
            url=url,
        )
    missing = sorted(expected - set(found))
    return [found[key] for key in sorted(found)], missing


async def collect_coverage(
    client: PublicMetadataClient, config: Mapping[str, Any]
) -> tuple[list[ArchiveMetadata], list[dict[str, str]]]:
    if config.get("schema_version") != OKX_L2_COVERAGE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported coverage schema: {config.get('schema_version')}")
    instruments = [str(value) for value in config["instruments"]]
    dates = [str(value) for value in config["dates_utc"]]
    if len(set(instruments)) != len(instruments) or len(set(dates)) != len(dates):
        raise ValueError("Coverage instruments and dates must be unique")
    maximum_file_size = _positive_decimal(
        config["maximum_single_file_mb"], "maximum single-file size"
    )
    records: list[ArchiveMetadata] = []
    missing: list[dict[str, str]] = []
    for index, date_utc in enumerate(dates):
        timestamp_ms = _date_ms(date_utc)
        payload = await client.get_json(
            str(config["endpoint"]),
            {
                "module": int(config["module"]),
                "instType": str(config["instrument_type"]),
                "instIdList": ",".join(instruments),
                "dateAggrType": str(config["date_aggregation"]),
                "begin": timestamp_ms,
                "end": timestamp_ms,
            },
        )
        parsed, absent = parse_metadata_response(
            payload,
            date_utc=date_utc,
            instruments=instruments,
            maximum_single_file_mb=maximum_file_size,
        )
        records.extend(parsed)
        missing.extend({"date_utc": date_utc, "instrument": item} for item in absent)
        if index + 1 < len(dates) and float(config["request_delay_seconds"]) > 0:
            await asyncio.sleep(float(config["request_delay_seconds"]))
    return records, missing


def _percentile(values: Sequence[Decimal], fraction: float) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    position = Decimal(len(ordered) - 1) * Decimal(str(fraction))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - Decimal(lower)
    return ordered[lower] * (Decimal(1) - weight) + ordered[upper] * weight


def _size_summary(values: Sequence[Decimal]) -> dict[str, float | int | None]:
    return {
        "observations": len(values),
        "minimum_mb": float(min(values)) if values else None,
        "median_mb": float(_percentile(values, 0.5)) if values else None,
        "p95_mb": float(_percentile(values, 0.95)) if values else None,
        "maximum_mb": float(max(values)) if values else None,
    }


def build_coverage_result(
    records: Sequence[ArchiveMetadata],
    missing: Sequence[Mapping[str, str]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    instruments = [str(value) for value in config["instruments"]]
    dates = [str(value) for value in config["dates_utc"]]
    expected_total = len(instruments) * len(dates)
    counts = Counter(record.instrument for record in records)
    rows_by_date: dict[str, list[ArchiveMetadata]] = {date: [] for date in dates}
    for record in records:
        rows_by_date[record.date_utc].append(record)

    instrument_summaries = {}
    for instrument in instruments:
        sizes = [Decimal(record.size_mb) for record in records if record.instrument == instrument]
        instrument_summaries[instrument] = {
            "coverage": counts[instrument] / len(dates),
            **_size_summary(sizes),
        }

    complete_daily_totals: dict[str, Decimal] = {}
    for date, rows in rows_by_date.items():
        if len(rows) == len(instruments):
            complete_daily_totals[date] = sum(Decimal(row.size_mb) for row in rows)
    daily_values = list(complete_daily_totals.values())
    daily_summary = _size_summary(daily_values)
    median_daily = _percentile(daily_values, 0.5)

    candidate_dates = [str(value) for value in config["candidate_dates_utc"]]
    candidate_complete = all(date in complete_daily_totals for date in candidate_dates)
    candidate_size = (
        sum(complete_daily_totals[date] for date in candidate_dates) if candidate_complete else None
    )
    coverage_valid = len(records) >= int(config["minimum_total_files"]) and all(
        counts[instrument] >= int(config["minimum_dates_per_instrument"])
        for instrument in instruments
    )
    candidate_feasible = (
        coverage_valid
        and candidate_size is not None
        and candidate_size <= Decimal(str(config["candidate_download_limit_mb"]))
    )

    return {
        "schema_version": OKX_L2_COVERAGE_SCHEMA_VERSION,
        "provider": config["provider"],
        "query": {
            "endpoint": config["endpoint"],
            "module": config["module"],
            "instrument_type": config["instrument_type"],
            "date_aggregation": config["date_aggregation"],
            "instruments": instruments,
            "dates_utc": dates,
        },
        "coverage": {
            "expected_files": expected_total,
            "present_files": len(records),
            "missing_files": len(missing),
            "coverage": len(records) / expected_total if expected_total else 0.0,
            "missing": list(missing),
            "valid_at_frozen_thresholds": coverage_valid,
        },
        "instrument_summaries": instrument_summaries,
        "complete_date_totals_mb": {
            date: float(value) for date, value in sorted(complete_daily_totals.items())
        },
        "complete_daily_size_summary": daily_summary,
        "size_extrapolation": {
            "basis": "median complete three-instrument sampled day",
            "thirty_days_mb": float(median_daily * Decimal(30)) if median_daily else None,
            "annual_365_25_days_mb": (
                float(median_daily * Decimal("365.25")) if median_daily else None
            ),
            "warning": "Indicative extrapolation only; archive size varies with market activity",
        },
        "candidate_sample": {
            "dates_utc": candidate_dates,
            "all_dates_complete": candidate_complete,
            "total_size_mb": float(candidate_size) if candidate_size is not None else None,
            "limit_mb": float(Decimal(str(config["candidate_download_limit_mb"]))),
            "materially_feasible": candidate_feasible,
        },
        "decision": "GO_BOUNDED_SAMPLE_METADATA" if candidate_feasible else "NO_GO_BOUNDED_SAMPLE",
        "archive_downloads_performed": False,
        "credentials_used": False,
    }


def _metadata_csv(records: Sequence[ArchiveMetadata]) -> bytes:
    buffer = io.StringIO(newline="")
    fieldnames = ["date_utc", "date_timestamp_ms", "instrument", "filename", "size_mb", "url"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in sorted(records, key=lambda item: (item.date_utc, item.instrument)):
        writer.writerow(asdict(record))
    return buffer.getvalue().encode("utf-8")


def write_coverage_artifact(
    result: Mapping[str, Any],
    records: Sequence[ArchiveMetadata],
    *,
    config_sha256: str,
    acquisition_code_sha256: str,
    output_root: Path,
) -> Path:
    csv_payload = _metadata_csv(records)
    identity = {
        **result,
        "config_sha256": config_sha256,
        "acquisition_code_sha256": acquisition_code_sha256,
        "metadata_csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
    }
    artifact_id = (
        f"{OKX_L2_COVERAGE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Coverage artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "coverage_result.json").write_bytes(_json_bytes(identity, pretty=True))
        (temporary_path / "archive_metadata.csv").write_bytes(csv_payload)
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "public_read_only_endpoint": f"GET {result['query']['endpoint']}",
            "archive_downloads_performed": False,
            "credentials_used": False,
            "orders_placed": False,
            "model_trained": False,
            "production_touched": False,
            "result_file": "coverage_result.json",
            "result_sha256": file_sha256(temporary_path / "coverage_result.json"),
            "metadata_file": "archive_metadata.csv",
            "metadata_sha256": file_sha256(temporary_path / "archive_metadata.csv"),
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
