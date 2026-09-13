"""Checksum-verified Binance USD-M funding history acquisition for offline research."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

FUNDING_ACQUISITION_SCHEMA_VERSION = "crypto-forecast-binance-funding-history-v1"
EXPECTED_HEADER = ("calc_time", "funding_interval_hours", "last_funding_rate")
HOUR_MS = 3_600_000
SHA256_PATTERN = re.compile(r"^(?P<sha>[0-9a-fA-F]{64})\s+\*?(?P<name>[^\s]+)\s*$")


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    market_symbol: str


@dataclass(frozen=True)
class FundingObservation:
    calc_time_ms: int
    funding_interval_hours: int
    funding_rate: str

    @property
    def timestamp_utc(self) -> str:
        return datetime.fromtimestamp(self.calc_time_ms / 1000, timezone.utc).isoformat()

    @property
    def utc_date(self) -> str:
        return datetime.fromtimestamp(self.calc_time_ms / 1000, timezone.utc).date().isoformat()


@dataclass(frozen=True)
class ValidatedArchive:
    spec: InstrumentSpec
    month: str
    filename: str
    archive_bytes: int
    uncompressed_bytes: int
    official_sha256: str
    calculated_sha256: str
    observations: tuple[FundingObservation, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: object, *, pretty: bool = False) -> bytes:
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


def parse_month(value: str) -> tuple[datetime, datetime]:
    try:
        start = datetime.strptime(value, "%Y-%m").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"Invalid month: {value!r}") from exc
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def iter_months(first_month: str, last_month: str) -> list[str]:
    cursor, _ = parse_month(first_month)
    final, _ = parse_month(last_month)
    if cursor > final:
        raise ValueError("first_month must not be after last_month")
    months = []
    while cursor <= final:
        months.append(cursor.strftime("%Y-%m"))
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return months


def parse_official_checksum(payload: bytes, expected_filename: str) -> str:
    try:
        text = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Checksum payload must be ASCII") from exc
    match = SHA256_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("Checksum payload has an invalid format")
    if match.group("name") != expected_filename:
        raise ValueError("Checksum references the wrong archive")
    return match.group("sha").lower()


def _safe_single_member(archive: zipfile.ZipFile, expected_csv: str) -> zipfile.ZipInfo:
    members = archive.infolist()
    if len(members) != 1 or members[0].is_dir():
        raise ValueError("Funding archive must contain exactly one CSV file")
    info = members[0]
    path = PurePosixPath(info.filename.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError("Funding archive contains an unsafe path")
    if path.name != expected_csv:
        raise ValueError("Funding archive contains an unexpected CSV filename")
    return info


def _parse_observations(payload: bytes, month: str) -> tuple[FundingObservation, ...]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Funding CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
        raise ValueError(f"Unrecognized funding CSV schema: {reader.fieldnames!r}")
    month_start, month_end = parse_month(month)
    start_ms = int(month_start.timestamp() * 1000)
    end_ms = int(month_end.timestamp() * 1000)
    observations: list[FundingObservation] = []
    previous_time: int | None = None
    for row_number, row in enumerate(reader, start=2):
        try:
            calc_time_ms = int(row["calc_time"])
            interval_hours = int(row["funding_interval_hours"])
            funding_rate = str(row["last_funding_rate"])
            funding_rate_number = float(funding_rate)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid funding row {row_number}") from exc
        if not start_ms <= calc_time_ms < end_ms:
            raise ValueError(f"Funding row {row_number} falls outside {month}")
        if not 1 <= interval_hours <= 24:
            raise ValueError(f"Invalid funding interval at row {row_number}")
        if not math.isfinite(funding_rate_number):
            raise ValueError(f"Non-finite funding rate at row {row_number}")
        if previous_time is not None and calc_time_ms <= previous_time:
            raise ValueError("Funding timestamps must be strictly increasing and unique")
        observations.append(
            FundingObservation(
                calc_time_ms=calc_time_ms,
                funding_interval_hours=interval_hours,
                funding_rate=funding_rate,
            )
        )
        previous_time = calc_time_ms
    if not observations:
        raise ValueError("Funding CSV contains no observations")
    return tuple(observations)


def validate_archive(
    spec: InstrumentSpec,
    month: str,
    archive_payload: bytes,
    checksum_payload: bytes,
    *,
    maximum_archive_bytes: int,
    maximum_uncompressed_bytes: int,
) -> ValidatedArchive:
    filename = f"{spec.market_symbol}-fundingRate-{month}.zip"
    expected_csv = filename.removesuffix(".zip") + ".csv"
    if len(archive_payload) > maximum_archive_bytes:
        raise ValueError(f"Archive exceeds the frozen byte limit: {filename}")
    official_sha256 = parse_official_checksum(checksum_payload, filename)
    calculated_sha256 = sha256_bytes(archive_payload)
    if calculated_sha256 != official_sha256:
        raise ValueError(f"Checksum mismatch for {filename}")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            info = _safe_single_member(archive, expected_csv)
            if info.file_size > maximum_uncompressed_bytes:
                raise ValueError(f"CSV exceeds the frozen byte limit: {expected_csv}")
            csv_payload = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {filename}") from exc
    observations = _parse_observations(csv_payload, month)
    return ValidatedArchive(
        spec=spec,
        month=month,
        filename=filename,
        archive_bytes=len(archive_payload),
        uncompressed_bytes=len(csv_payload),
        official_sha256=official_sha256,
        calculated_sha256=calculated_sha256,
        observations=observations,
    )


def load_instrument_specs(config: Mapping[str, Any]) -> list[InstrumentSpec]:
    if config.get("schema_version") != FUNDING_ACQUISITION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported acquisition schema: {config.get('schema_version')}")
    specs = [InstrumentSpec(**item) for item in config["instruments"]]
    if len(specs) < 5:
        raise ValueError("At least five instruments are required")
    symbols = [item.symbol for item in specs]
    markets = [item.market_symbol for item in specs]
    if len(set(symbols)) != len(symbols) or len(set(markets)) != len(markets):
        raise ValueError("Instrument symbols and market symbols must be unique")
    if any(not item.market_symbol.endswith("USDT") for item in specs):
        raise ValueError("Every funding market must be quoted in USDT")
    months = iter_months(str(config["first_month"]), str(config["last_month"]))
    expected = len(specs) * len(months)
    if int(config["expected_archive_count"]) != expected:
        raise ValueError("expected_archive_count does not match instruments and months")
    return specs


def _date_range(start: date, end: date) -> list[str]:
    return [
        (start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)
    ]


def validate_history(
    archives: Sequence[ValidatedArchive], config: Mapping[str, Any]
) -> dict[str, tuple[FundingObservation, ...]]:
    specs = load_instrument_specs(config)
    months = iter_months(str(config["first_month"]), str(config["last_month"]))
    expected_pairs = {(spec.market_symbol, month) for spec in specs for month in months}
    actual_pairs = {(item.spec.market_symbol, item.month) for item in archives}
    if len(actual_pairs) != len(archives):
        raise ValueError("Duplicate instrument-month archive")
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(f"Archive contract mismatch; missing={missing}, extra={extra}")
    total_archive_bytes = sum(item.archive_bytes for item in archives)
    if total_archive_bytes > int(config["maximum_total_archive_bytes"]):
        raise ValueError("Total archives exceed the frozen byte limit")

    start = date.fromisoformat(str(config["start_date"]))
    end = date.fromisoformat(str(config["end_date"]))
    expected_dates = _date_range(start, end)
    if len(expected_dates) < int(config["minimum_calendar_days"]):
        raise ValueError("Configured period is shorter than the frozen minimum")
    result: dict[str, tuple[FundingObservation, ...]] = {}
    for spec in specs:
        observations = tuple(
            observation
            for archive in sorted(
                (item for item in archives if item.spec == spec), key=lambda item: item.month
            )
            for observation in archive.observations
        )
        timestamps = [item.calc_time_ms for item in observations]
        if len(set(timestamps)) != len(timestamps) or timestamps != sorted(timestamps):
            raise ValueError(f"Duplicate or unordered timestamps for {spec.market_symbol}")
        counts = Counter(
            item.utc_date
            for item in observations
            if start <= date.fromisoformat(item.utc_date) <= end
        )
        missing_dates = [
            day
            for day in expected_dates
            if counts[day] < int(config["minimum_observations_per_day"])
        ]
        if missing_dates:
            raise ValueError(f"Missing funding days for {spec.market_symbol}: {missing_dates[:5]}")
        if (
            observations[0].utc_date != start.isoformat()
            or observations[-1].utc_date != end.isoformat()
        ):
            raise ValueError(f"Contract boundaries are missing for {spec.market_symbol}")
        maximum_gap_ms = int(config["maximum_gap_hours"]) * HOUR_MS
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if gaps and max(gaps) > maximum_gap_ms:
            raise ValueError(f"Funding gap exceeds the frozen limit for {spec.market_symbol}")
        result[spec.market_symbol] = observations
    return result


def _events_csv_bytes(observations: Sequence[FundingObservation]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["timestamp_utc", "calc_time_ms", "utc_date", "funding_interval_hours", "funding_rate"]
    )
    for observation in observations:
        writer.writerow(
            [
                observation.timestamp_utc,
                observation.calc_time_ms,
                observation.utc_date,
                observation.funding_interval_hours,
                observation.funding_rate,
            ]
        )
    return buffer.getvalue().encode("utf-8")


def _coverage_csv_bytes(observations: Sequence[FundingObservation]) -> bytes:
    counts = Counter(item.utc_date for item in observations)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utc_date", "observation_count"])
    for day in sorted(counts):
        writer.writerow([day, counts[day]])
    return buffer.getvalue().encode("utf-8")


def write_acquisition_artifact(
    archives: Sequence[ValidatedArchive],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    acquisition_code_sha256: str,
    output_root: str | Path,
) -> Path:
    history = validate_history(archives, config)
    files: dict[str, bytes] = {}
    inputs = []
    for market_symbol in sorted(history):
        observations = history[market_symbol]
        events_name = f"events/{market_symbol}_funding.csv"
        coverage_name = f"coverage/{market_symbol}_daily.csv"
        events_payload = _events_csv_bytes(observations)
        coverage_payload = _coverage_csv_bytes(observations)
        files[events_name] = events_payload
        files[coverage_name] = coverage_payload
        archive_records = [
            {
                "month": item.month,
                "filename": item.filename,
                "archive_bytes": item.archive_bytes,
                "uncompressed_bytes": item.uncompressed_bytes,
                "official_sha256": item.official_sha256,
                "calculated_sha256": item.calculated_sha256,
                "observations": len(item.observations),
            }
            for item in sorted(
                (entry for entry in archives if entry.spec.market_symbol == market_symbol),
                key=lambda entry: entry.month,
            )
        ]
        inputs.append(
            {
                "symbol": next(
                    item.spec.symbol
                    for item in archives
                    if item.spec.market_symbol == market_symbol
                ),
                "market_symbol": market_symbol,
                "observations": len(observations),
                "first_timestamp_utc": observations[0].timestamp_utc,
                "last_timestamp_utc": observations[-1].timestamp_utc,
                "calendar_days": len({item.utc_date for item in observations}),
                "events_file": events_name,
                "events_file_sha256": sha256_bytes(events_payload),
                "coverage_file": coverage_name,
                "coverage_file_sha256": sha256_bytes(coverage_payload),
                "archives": archive_records,
            }
        )
    identity = {
        "schema_version": FUNDING_ACQUISITION_SCHEMA_VERSION,
        "provider": config["provider"],
        "base_url": config["base_url"],
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "config_sha256": config_sha256,
        "acquisition_code_sha256": acquisition_code_sha256,
        "total_archive_bytes": sum(item.archive_bytes for item in archives),
        "inputs": inputs,
    }
    artifact_id = f"{FUNDING_ACQUISITION_SCHEMA_VERSION}-{sha256_bytes(json_bytes(identity))[:16]}"
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final_directory = root / artifact_id
    if final_directory.exists():
        raise FileExistsError(f"Acquisition artifact already exists: {final_directory}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=root) as temporary:
        temporary_directory = Path(temporary)
        for relative_name, payload in files.items():
            destination = temporary_directory / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        manifest = {
            **identity,
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "credentials_used": False,
            "orders_created": False,
            "raw_archives_retained": False,
            "targets_created": False,
            "predictive_features_created": False,
            "normalization": (
                "official monthly ZIP SHA-256 verified; exact published funding-rate strings "
                "retained; no missing day filled; coverage output contains counts only"
            ),
        }
        (temporary_directory / "acquisition_manifest.json").write_bytes(
            json_bytes(manifest, pretty=True)
        )
        Path(temporary).replace(final_directory)
    return final_directory
