"""Bounded, offline analysis of one pre-registered OKX historical L2 archive."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

OKX_L2_PILOT_SCHEMA_VERSION = "crypto-forecast-okx-l2-pilot-v1"
DAY_MS = 86_400_000


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


def _utc_day_bounds(date_text: str) -> tuple[int, int]:
    day = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(day.timestamp() * 1000)
    return start_ms, start_ms + DAY_MS


def _decimal(value: object, field: str, *, allow_zero: bool) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not number.is_finite() or number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"Invalid {field}: {value!r}")
    return number


def _level(value: object) -> tuple[Decimal, Decimal, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"Invalid order-book level: {value!r}")
    price = _decimal(value[0], "price", allow_zero=False)
    size = _decimal(value[1], "size", allow_zero=True)
    try:
        auxiliary_count = int(str(value[2]))
    except ValueError as exc:
        raise ValueError(f"Invalid auxiliary level count: {value[2]!r}") from exc
    if auxiliary_count < 0:
        raise ValueError(f"Inconsistent order-book level: {value!r}")
    return price, size, auxiliary_count


class OrderBook:
    def __init__(self) -> None:
        self.bids: dict[Decimal, tuple[Decimal, int]] = {}
        self.asks: dict[Decimal, tuple[Decimal, int]] = {}
        self.initialized = False

    def apply(self, action: str, bids: object, asks: object) -> None:
        if action == "snapshot":
            self.bids.clear()
            self.asks.clear()
            self.initialized = True
        elif action != "update":
            raise ValueError(f"Unsupported order-book action: {action!r}")
        elif not self.initialized:
            raise ValueError("An update was received before the first snapshot")
        self._apply_side(self.bids, bids, "bids")
        self._apply_side(self.asks, asks, "asks")

    @staticmethod
    def _apply_side(
        destination: dict[Decimal, tuple[Decimal, int]], values: object, field: str
    ) -> None:
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        for raw_level in values:
            price, size, auxiliary_count = _level(raw_level)
            if size == 0:
                destination.pop(price, None)
            else:
                destination[price] = (size, auxiliary_count)

    def metrics(self, bands_bps: Sequence[int]) -> dict[str, object]:
        if not self.bids or not self.asks:
            return {"valid": False, "reason": "empty_side"}
        best_bid = max(self.bids)
        best_ask = min(self.asks)
        mid = (best_bid + best_ask) / Decimal(2)
        if mid <= 0:
            return {"valid": False, "reason": "invalid_mid"}
        spread_bps = (best_ask - best_bid) / mid * Decimal(10_000)
        result: dict[str, object] = {
            "valid": best_bid <= best_ask,
            "reason": "ok" if best_bid <= best_ask else "crossed_book",
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "mid": float(mid),
            "spread_bps": float(spread_bps),
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
        }
        for band in bands_bps:
            fraction = Decimal(band) / Decimal(10_000)
            bid_floor = mid * (Decimal(1) - fraction)
            ask_ceiling = mid * (Decimal(1) + fraction)
            bid_notional = sum(
                price * size for price, (size, _count) in self.bids.items() if price >= bid_floor
            )
            ask_notional = sum(
                price * size for price, (size, _count) in self.asks.items() if price <= ask_ceiling
            )
            total = bid_notional + ask_notional
            imbalance = (bid_notional - ask_notional) / total if total else Decimal(0)
            result[f"bid_depth_{band}bps_usdt"] = float(bid_notional)
            result[f"ask_depth_{band}bps_usdt"] = float(ask_notional)
            result[f"imbalance_{band}bps"] = float(imbalance)
        return result


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(samples: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    values = [float(row[field]) for row in samples if row.get("valid") and field in row]
    return {
        "observations": len(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def analyze_archive(
    archive_path: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    if config.get("schema_version") != OKX_L2_PILOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pilot schema: {config.get('schema_version')}")
    archive_size = archive_path.stat().st_size
    if archive_size > int(config["download_limit_bytes"]):
        raise ValueError("Compressed archive exceeds the pre-registered byte limit")
    archive_hash = file_sha256(archive_path)
    if archive_hash != str(config["expected_archive_sha256"]):
        raise ValueError("Archive SHA-256 does not match the pinned pilot input")

    day_start_ms, day_end_ms = _utc_day_bounds(str(config["date_utc"]))
    sample_interval_ms = int(config["sample_interval_ms"])
    if sample_interval_ms <= 0 or DAY_MS % sample_interval_ms:
        raise ValueError("sample_interval_ms must divide one UTC day")
    bands = [int(value) for value in config["depth_bands_bps"]]
    if not bands or any(value <= 0 for value in bands):
        raise ValueError("depth_bands_bps must contain positive values")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError("The pilot archive must contain exactly one regular member")
        member = members[0]
        if member.size > int(config["uncompressed_limit_bytes"]):
            raise ValueError("Uncompressed member exceeds the pre-registered byte limit")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Unable to open the archive member")

        book = OrderBook()
        record_count = 0
        action_counts: Counter[str] = Counter()
        record_keys: set[str] = set()
        previous_ts: int | None = None
        first_ts: int | None = None
        last_ts: int | None = None
        maximum_timestamp_gap_ms = 0
        records_with_sequence_fields = 0
        sequence_links_checked = 0
        sequence_links_valid = 0
        previous_seq_id: int | None = None
        snapshot_timestamps: list[int] = []
        samples: list[dict[str, object]] = []
        next_sample_ms = day_start_ms

        def emit_until(bound_ms: int) -> None:
            nonlocal next_sample_ms
            while next_sample_ms < bound_ms and next_sample_ms < day_end_ms:
                if book.initialized:
                    row = {"timestamp_ms": next_sample_ms, **book.metrics(bands)}
                    samples.append(row)
                next_sample_ms += sample_interval_ms

        for raw_line in stream:
            record_count += 1
            if record_count > int(config["max_records"]):
                raise ValueError("Record count exceeds the pre-registered limit")
            if len(raw_line) > int(config["max_line_bytes"]):
                raise ValueError("A JSON record exceeds the pre-registered line-size limit")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON record at line {record_count}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record {record_count} is not an object")
            record_keys.update(str(key) for key in record)
            if record.get("instId") != config["instrument"]:
                raise ValueError(f"Unexpected instrument at line {record_count}")
            try:
                timestamp_ms = int(str(record["ts"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid timestamp at line {record_count}") from exc
            if not day_start_ms <= timestamp_ms < day_end_ms:
                raise ValueError(f"Timestamp outside the frozen UTC day at line {record_count}")
            if previous_ts is not None:
                if timestamp_ms < previous_ts:
                    raise ValueError(f"Timestamps moved backwards at line {record_count}")
                maximum_timestamp_gap_ms = max(maximum_timestamp_gap_ms, timestamp_ms - previous_ts)
                if timestamp_ms > previous_ts:
                    emit_until(timestamp_ms)
            first_ts = timestamp_ms if first_ts is None else first_ts
            previous_ts = timestamp_ms
            last_ts = timestamp_ms

            action = str(record.get("action", ""))
            action_counts[action] += 1
            book.apply(action, record.get("bids"), record.get("asks"))
            if action == "snapshot":
                snapshot_timestamps.append(timestamp_ms)

            if "seqId" in record and "prevSeqId" in record:
                records_with_sequence_fields += 1
                seq_id = int(str(record["seqId"]))
                prev_seq_id = int(str(record["prevSeqId"]))
                if previous_seq_id is not None:
                    sequence_links_checked += 1
                    if prev_seq_id == previous_seq_id:
                        sequence_links_valid += 1
                previous_seq_id = seq_id

        emit_until(day_end_ms)

    sequence_field_coverage = records_with_sequence_fields / record_count if record_count else 0.0
    sequence_link_coverage = (
        sequence_links_valid / sequence_links_checked if sequence_links_checked else None
    )
    valid_samples = [row for row in samples if row.get("valid")]
    valid_sample_ratio = len(valid_samples) / len(samples) if samples else 0.0
    required_sequence_coverage = float(config["required_sequence_coverage"])
    sequence_verifiable = (
        sequence_field_coverage >= required_sequence_coverage
        and sequence_link_coverage is not None
        and sequence_link_coverage >= required_sequence_coverage
    )
    snapshot_gaps_ms = [
        current - previous
        for previous, current in zip(snapshot_timestamps, snapshot_timestamps[1:])
    ]
    go_technical = (
        record_count > 0
        and action_counts["snapshot"] >= 1
        and valid_sample_ratio >= 0.99
        and sequence_verifiable
    )
    decision_reasons = []
    if not sequence_verifiable:
        decision_reasons.append(
            "Sequence continuity is not verifiable at the pre-registered 99% threshold"
        )
    if valid_sample_ratio < 0.99:
        decision_reasons.append("Less than 99% of minute samples contain a valid uncrossed book")

    metric_summaries = {"spread_bps": _summary(samples, "spread_bps")}
    for band in bands:
        for field in (
            f"bid_depth_{band}bps_usdt",
            f"ask_depth_{band}bps_usdt",
            f"imbalance_{band}bps",
        ):
            metric_summaries[field] = _summary(samples, field)

    result: dict[str, Any] = {
        "schema_version": OKX_L2_PILOT_SCHEMA_VERSION,
        "provider": config["provider"],
        "instrument": config["instrument"],
        "date_utc": config["date_utc"],
        "archive": {
            "filename": archive_path.name,
            "compressed_bytes": archive_size,
            "sha256": archive_hash,
            "member_name": member.name,
            "member_bytes": member.size,
        },
        "records": {
            "total": record_count,
            "actions": dict(sorted(action_counts.items())),
            "fields": sorted(record_keys),
            "first_timestamp_ms": first_ts,
            "last_timestamp_ms": last_ts,
            "maximum_timestamp_gap_ms": maximum_timestamp_gap_ms,
        },
        "snapshot_cadence": {
            "count": len(snapshot_timestamps),
            "first_timestamp_ms": snapshot_timestamps[0] if snapshot_timestamps else None,
            "last_timestamp_ms": snapshot_timestamps[-1] if snapshot_timestamps else None,
            "minimum_gap_ms": min(snapshot_gaps_ms) if snapshot_gaps_ms else None,
            "median_gap_ms": _percentile(snapshot_gaps_ms, 0.5),
            "maximum_gap_ms": max(snapshot_gaps_ms) if snapshot_gaps_ms else None,
            "snapshot_only_follow_up_candidate": len(snapshot_timestamps) > 1,
        },
        "sequence_integrity": {
            "records_with_seqId_and_prevSeqId": records_with_sequence_fields,
            "field_coverage": sequence_field_coverage,
            "links_checked": sequence_links_checked,
            "links_valid": sequence_links_valid,
            "link_coverage": sequence_link_coverage,
            "verifiable_at_required_threshold": sequence_verifiable,
        },
        "minute_samples": {
            "total": len(samples),
            "valid_uncrossed": len(valid_samples),
            "valid_ratio": valid_sample_ratio,
            "crossed_or_empty": len(samples) - len(valid_samples),
            "sampling_semantics": "causal as-of UTC grid; no future update used",
        },
        "metric_summaries": metric_summaries,
        "decision": "GO_TECHNICAL" if go_technical else "NO_GO_TECHNICAL",
        "decision_reasons": decision_reasons,
        "limitations": [
            "One instrument and one UTC day only",
            "No predictive model or economic evaluation",
            "No interpolation or silent sequence repair",
            "Minute metrics use deltas whose continuity is not independently verifiable",
        ],
    }
    return result, samples


def _samples_csv(samples: Sequence[Mapping[str, object]]) -> bytes:
    if not samples:
        return b""
    fieldnames = list(samples[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(samples)
    return buffer.getvalue().encode("utf-8")


def write_pilot_artifact(
    result: Mapping[str, Any],
    samples: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    analysis_code_sha256: str,
    output_root: Path,
) -> Path:
    identity = {
        **result,
        "config_sha256": config_sha256,
        "analysis_code_sha256": analysis_code_sha256,
    }
    artifact_id = (
        f"{OKX_L2_PILOT_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Pilot artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "pilot_result.json").write_bytes(_json_bytes(identity, pretty=True))
        (temporary_path / "minute_metrics.csv").write_bytes(_samples_csv(samples))
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "credentials_used": False,
            "orders_placed": False,
            "raw_member_extracted_to_disk": False,
            "model_trained": False,
            "production_touched": False,
            "result_file": "pilot_result.json",
            "result_sha256": file_sha256(temporary_path / "pilot_result.json"),
            "metrics_file": "minute_metrics.csv",
            "metrics_sha256": file_sha256(temporary_path / "minute_metrics.csv"),
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
