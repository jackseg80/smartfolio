"""Causal, independent-snapshot features for the compact OKX L2 corpus."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from services.forecasting.okx_l2_pilot import DAY_MS, OrderBook, file_sha256
from services.forecasting.okx_l2_progressive_extraction import (
    OKX_L2_PROGRESSIVE_EXTRACTION_SCHEMA_VERSION,
)

OKX_L2_FEATURES_SCHEMA_VERSION = "crypto-forecast-okx-l2-features-v1"
META_FIELDS = [
    "date_utc",
    "instrument",
    "grid_timestamp_ms",
    "source_timestamp_ms",
    "signed_offset_ms",
    "absolute_offset_ms",
    "availability_timestamp_ms",
    "available",
    "missing_reason",
]


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


def _day_start_ms(date_utc: str) -> int:
    day = datetime.strptime(date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(day.timestamp() * 1000)


def _safe_source_path(corpus_root: Path, relative_text: str) -> Path:
    if "\\" in relative_text:
        raise ValueError("Source paths must use safe POSIX separators")
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or len(relative.parts) != 3 or ".." in relative.parts:
        raise ValueError("Source path must be data/date/instrument.jsonl.gz")
    resolved_root = corpus_root.resolve()
    resolved_path = resolved_root.joinpath(*relative.parts).resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError("Source path escapes the compact corpus")
    return resolved_path


def _validate_config(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if config.get("schema_version") != OKX_L2_FEATURES_SCHEMA_VERSION:
        raise ValueError(f"Unsupported L2 feature schema: {config.get('schema_version')}")
    dates = [str(value) for value in config["dates_utc"]]
    instruments = [str(value) for value in config["instruments"]]
    if len(dates) != len(set(dates)) or len(instruments) != len(set(instruments)):
        raise ValueError("Frozen dates and instruments must be unique")
    expected_groups = [(date, instrument) for date in dates for instrument in instruments]
    source_files = config.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != len(expected_groups):
        raise ValueError("Source file count does not match the frozen group grid")
    actual_groups: list[tuple[str, str]] = []
    for item in source_files:
        relative = PurePosixPath(str(item["path"]))
        if len(relative.parts) != 3:
            raise ValueError("Source file path does not expose a date and instrument")
        actual_groups.append((relative.parts[1], relative.parts[2].removesuffix(".jsonl.gz")))
    if actual_groups != expected_groups:
        raise ValueError("Source file order does not match the frozen group grid")

    interval_ms = int(config["grid_interval_ms"])
    slots = int(config["expected_grid_slots_per_group"])
    if interval_ms <= 0 or DAY_MS % interval_ms or DAY_MS // interval_ms != slots:
        raise ValueError("Grid interval and frozen slot count are inconsistent")
    expected_total = len(expected_groups) * slots
    if int(config["expected_total_rows"]) != expected_total:
        raise ValueError("Frozen total row count is inconsistent")
    available = int(config["expected_available_rows"])
    missing = int(config["expected_missing_rows"])
    if available + missing != expected_total:
        raise ValueError("Frozen available and missing totals are inconsistent")

    missing_keys = [
        (str(item["date_utc"]), str(item["instrument"]), int(item["grid_timestamp_ms"]))
        for item in config["expected_missing_keys"]
    ]
    if len(missing_keys) != missing or len(missing_keys) != len(set(missing_keys)):
        raise ValueError("Frozen missing keys are inconsistent")
    valid_keys = {
        (date, instrument, _day_start_ms(date) + slot * interval_ms)
        for date, instrument in expected_groups
        for slot in range(slots)
    }
    if any(key not in valid_keys for key in missing_keys):
        raise ValueError("A frozen missing key is outside the expected grid")

    bands = [int(value) for value in config["depth_bands_bps"]]
    if bands != sorted(set(bands)) or any(value <= 0 for value in bands):
        raise ValueError("Depth bands must be positive, unique, and sorted")
    concentration_band = int(config["concentration_band_bps"])
    concentration_levels = int(config["concentration_levels"])
    if concentration_band != 50 or concentration_levels != 5:
        raise ValueError("Concentration settings must match the top5_50bps feature names")
    expected_feature_fields = _feature_field_names(bands)
    if list(config["feature_fields"]) != expected_feature_fields:
        raise ValueError("Feature fields do not match the frozen feature contract")
    if not set(config["reference_fields"]).issubset(expected_feature_fields):
        raise ValueError("Reference fields must be part of the frozen feature contract")
    if (
        config.get("feature_time_semantics")
        != "independent_snapshot_only_at_causal_availability_timestamp"
    ):
        raise ValueError("Feature time semantics must preserve causal snapshot availability")
    if config.get("missing_policy") != "explicit_row_without_feature_values":
        raise ValueError("Missing rows must remain explicit and empty")
    if config.get("target_policy") != "no_target_or_future_return_in_this_lot":
        raise ValueError("Targets and future returns are forbidden in this feature lot")
    tolerances = (
        float(config["reference_relative_tolerance"]),
        float(config["reference_absolute_tolerance"]),
    )
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("Reference tolerances must be finite and non-negative")
    return source_files


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _validate_corpus_artifact(
    corpus_root: Path, source_files: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    result_path = corpus_root / "compact_corpus_result.json"
    manifest_path = corpus_root / "manifest.json"
    if file_sha256(result_path) != str(config["source_result_sha256"]):
        raise ValueError("Compact corpus result SHA-256 does not match the frozen input")
    result = _load_json_object(result_path, "compact corpus result")
    if (
        result.get("schema_version") != "crypto-forecast-okx-l2-compact-corpus-v1"
        or result.get("decision") != "GO_TECHNICAL_COMPACT_CORPUS"
    ):
        raise ValueError("Compact corpus result is not an approved technical input")
    manifest = _load_json_object(manifest_path, "compact corpus manifest")
    if (
        manifest.get("artifact_id") != config["source_artifact_id"]
        or manifest.get("result_sha256") != config["source_result_sha256"]
    ):
        raise ValueError("Compact corpus manifest does not match the frozen artifact")
    expected_files = [
        {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in source_files
    ]
    if manifest.get("data_files") != expected_files:
        raise ValueError("Compact corpus manifest data files differ from the frozen inputs")


def _validate_reference_artifact(reference_path: Path, config: Mapping[str, Any]) -> None:
    result_path = reference_path.parent / "alignment_result.json"
    manifest_path = reference_path.parent / "manifest.json"
    if file_sha256(result_path) != str(config["reference_alignment_result_sha256"]):
        raise ValueError("Reference alignment result SHA-256 does not match the frozen input")
    result = _load_json_object(result_path, "reference alignment result")
    if (
        result.get("schema_version") != "crypto-forecast-okx-l2-symmetric-alignment-v1"
        or result.get("decision") != "GO_TECHNICAL_ALIGNMENT"
    ):
        raise ValueError("Reference alignment result is not an approved technical input")
    manifest = _load_json_object(manifest_path, "reference alignment manifest")
    if (
        manifest.get("artifact_id") != config["reference_alignment_artifact_id"]
        or manifest.get("result_sha256") != config["reference_alignment_result_sha256"]
        or manifest.get("metrics_sha256") != config["reference_metrics_sha256"]
    ):
        raise ValueError("Reference alignment manifest does not match the frozen artifact")


def _feature_field_names(bands: Sequence[int]) -> list[str]:
    fields = [
        "best_bid",
        "best_ask",
        "mid",
        "spread_bps",
        "bid_levels",
        "ask_levels",
        "best_bid_size",
        "best_ask_size",
        "top_level_imbalance",
        "microprice",
        "microprice_deviation_bps",
    ]
    for band in bands:
        fields.extend(
            [
                f"bid_depth_{band}bps_usdt",
                f"ask_depth_{band}bps_usdt",
                f"imbalance_{band}bps",
            ]
        )
    fields.extend(["bid_top5_50bps_share", "ask_top5_50bps_share"])
    return fields


def compute_snapshot_features(
    snapshot: Mapping[str, object], config: Mapping[str, Any]
) -> dict[str, float | int]:
    """Compute features using only one independent snapshot."""

    book = OrderBook()
    book.apply("snapshot", snapshot.get("bids"), snapshot.get("asks"))
    bands = [int(value) for value in config["depth_bands_bps"]]
    metrics = book.metrics(bands)
    if not metrics.get("valid"):
        raise ValueError(f"Snapshot is not a valid uncrossed book: {metrics.get('reason')}")
    best_bid = max(book.bids)
    best_ask = min(book.asks)
    bid_size = book.bids[best_bid][0]
    ask_size = book.asks[best_ask][0]
    size_total = bid_size + ask_size
    if size_total <= 0:
        raise ValueError("Best-level sizes must have a positive total")
    mid = (best_bid + best_ask) / Decimal(2)
    microprice = (best_ask * bid_size + best_bid * ask_size) / size_total
    top_imbalance = (bid_size - ask_size) / size_total

    concentration_band = int(config["concentration_band_bps"])
    fraction = Decimal(concentration_band) / Decimal(10_000)
    bid_floor = mid * (Decimal(1) - fraction)
    ask_ceiling = mid * (Decimal(1) + fraction)
    bid_levels = sorted(
        ((price, size) for price, (size, _count) in book.bids.items() if price >= bid_floor),
        reverse=True,
    )
    ask_levels = sorted(
        ((price, size) for price, (size, _count) in book.asks.items() if price <= ask_ceiling)
    )
    bid_total = sum((price * size for price, size in bid_levels), Decimal(0))
    ask_total = sum((price * size for price, size in ask_levels), Decimal(0))
    if bid_total <= 0 or ask_total <= 0:
        raise ValueError("The concentration band must contain positive depth on both sides")
    top_n = int(config["concentration_levels"])
    bid_top = sum((price * size for price, size in bid_levels[:top_n]), Decimal(0))
    ask_top = sum((price * size for price, size in ask_levels[:top_n]), Decimal(0))

    features: dict[str, float | int] = {
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "mid": float(mid),
        "spread_bps": float(metrics["spread_bps"]),
        "bid_levels": len(book.bids),
        "ask_levels": len(book.asks),
        "best_bid_size": float(bid_size),
        "best_ask_size": float(ask_size),
        "top_level_imbalance": float(top_imbalance),
        "microprice": float(microprice),
        "microprice_deviation_bps": float((microprice - mid) / mid * Decimal(10_000)),
    }
    for band in bands:
        for prefix in ("bid_depth", "ask_depth", "imbalance"):
            field = f"{prefix}_{band}bps" + ("_usdt" if prefix != "imbalance" else "")
            features[field] = float(metrics[field])
    features["bid_top5_50bps_share"] = float(bid_top / bid_total)
    features["ask_top5_50bps_share"] = float(ask_top / ask_total)

    if list(features) != list(config["feature_fields"]):
        raise ValueError("Computed feature order differs from the frozen contract")
    if any(isinstance(value, float) and not math.isfinite(value) for value in features.values()):
        raise ValueError("A computed feature is non-finite")
    if features["spread_bps"] < 0:
        raise ValueError("Spread must not be negative")
    for field, value in features.items():
        if "imbalance" in field and not -1 <= float(value) <= 1:
            raise ValueError(f"Feature must be within [-1, 1]: {field}")
        if field.endswith("_share") and not 0 <= float(value) <= 1:
            raise ValueError(f"Feature must be within [0, 1]: {field}")
    if not features["best_bid"] <= features["microprice"] <= features["best_ask"]:
        raise ValueError("Microprice must remain inside the best quotes")
    return features


def _load_source_rows(
    corpus_root: Path,
    source_files: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, int], dict[str, object]], list[dict[str, Any]]]:
    rows: dict[tuple[str, str, int], dict[str, object]] = {}
    summaries: list[dict[str, Any]] = []
    interval_ms = int(config["grid_interval_ms"])
    maximum_offset_ms = int(config["maximum_absolute_offset_ms"])
    expected_slots = int(config["expected_grid_slots_per_group"])
    for item in source_files:
        relative_path = str(item["path"])
        source_path = _safe_source_path(corpus_root, relative_path)
        actual_size = source_path.stat().st_size
        if actual_size != int(item["bytes"]):
            raise ValueError(f"Source file size mismatch: {relative_path}")
        if actual_size > int(config["max_compressed_bytes_per_file"]):
            raise ValueError(f"Source file exceeds the compressed byte limit: {relative_path}")
        digest = file_sha256(source_path)
        if digest != str(item["sha256"]):
            raise ValueError(f"Source file SHA-256 mismatch: {relative_path}")
        relative = PurePosixPath(relative_path)
        expected_date = relative.parts[1]
        expected_instrument = relative.parts[2].removesuffix(".jsonl.gz")
        day_start = _day_start_ms(expected_date)
        row_count = 0
        uncompressed_bytes = 0
        previous_grid: int | None = None
        with gzip.open(source_path, "rb") as stream:
            for raw_line in stream:
                row_count += 1
                uncompressed_bytes += len(raw_line)
                if row_count > int(config["max_rows_per_file"]):
                    raise ValueError(f"Source row count exceeds the limit: {relative_path}")
                if len(raw_line) > int(config["max_line_bytes"]):
                    raise ValueError(f"Source row exceeds the line-size limit: {relative_path}")
                if uncompressed_bytes > int(config["max_uncompressed_bytes_per_file"]):
                    raise ValueError(
                        f"Source file exceeds the expanded byte limit: {relative_path}"
                    )
                try:
                    snapshot = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid source JSON: {relative_path}:{row_count}") from exc
                if not isinstance(snapshot, dict):
                    raise ValueError(f"Source row is not an object: {relative_path}:{row_count}")
                if (
                    snapshot.get("schema_version") != OKX_L2_PROGRESSIVE_EXTRACTION_SCHEMA_VERSION
                    or snapshot.get("provider") != config["provider"]
                    or snapshot.get("date_utc") != expected_date
                    or snapshot.get("instrument") != expected_instrument
                ):
                    raise ValueError(f"Source row identity mismatch: {relative_path}:{row_count}")
                grid_timestamp = int(snapshot["grid_timestamp_ms"])
                source_timestamp = int(snapshot["source_timestamp_ms"])
                signed_offset = int(snapshot["signed_offset_ms"])
                absolute_offset = int(snapshot["absolute_offset_ms"])
                availability_timestamp = int(snapshot["availability_timestamp_ms"])
                if previous_grid is not None and grid_timestamp <= previous_grid:
                    raise ValueError(f"Source grid timestamps are not increasing: {relative_path}")
                previous_grid = grid_timestamp
                slot_delta = grid_timestamp - day_start
                if (
                    slot_delta < 0
                    or slot_delta % interval_ms
                    or slot_delta // interval_ms >= expected_slots
                ):
                    raise ValueError(f"Source row is outside the frozen grid: {relative_path}")
                if (
                    signed_offset != source_timestamp - grid_timestamp
                    or absolute_offset != abs(signed_offset)
                    or absolute_offset > maximum_offset_ms
                    or availability_timestamp != max(grid_timestamp, source_timestamp)
                ):
                    raise ValueError(
                        f"Source timestamp semantics are inconsistent: {relative_path}"
                    )
                key = (expected_date, expected_instrument, grid_timestamp)
                if key in rows:
                    raise ValueError(f"Duplicate source grid key: {key}")
                rows[key] = {
                    "date_utc": expected_date,
                    "instrument": expected_instrument,
                    "grid_timestamp_ms": grid_timestamp,
                    "source_timestamp_ms": source_timestamp,
                    "signed_offset_ms": signed_offset,
                    "absolute_offset_ms": absolute_offset,
                    "availability_timestamp_ms": availability_timestamp,
                    "available": True,
                    "missing_reason": "",
                    **compute_snapshot_features(snapshot, config),
                }
        summaries.append(
            {
                "path": relative_path,
                "compressed_bytes": actual_size,
                "uncompressed_bytes": uncompressed_bytes,
                "sha256": digest,
                "rows": row_count,
            }
        )
    return rows, summaries


def _ordered_feature_rows(
    source_rows: Mapping[tuple[str, str, int], Mapping[str, object]],
    config: Mapping[str, Any],
) -> list[dict[str, object]]:
    interval_ms = int(config["grid_interval_ms"])
    slots = int(config["expected_grid_slots_per_group"])
    feature_fields = list(config["feature_fields"])
    rows: list[dict[str, object]] = []
    for date_utc in config["dates_utc"]:
        day_start = _day_start_ms(str(date_utc))
        for instrument in config["instruments"]:
            for slot in range(slots):
                grid_timestamp = day_start + slot * interval_ms
                key = (str(date_utc), str(instrument), grid_timestamp)
                available = source_rows.get(key)
                if available is not None:
                    rows.append(dict(available))
                else:
                    rows.append(
                        {
                            "date_utc": date_utc,
                            "instrument": instrument,
                            "grid_timestamp_ms": grid_timestamp,
                            "source_timestamp_ms": "",
                            "signed_offset_ms": "",
                            "absolute_offset_ms": "",
                            "availability_timestamp_ms": "",
                            "available": False,
                            "missing_reason": "no_snapshot_within_symmetric_window",
                            **{field: "" for field in feature_fields},
                        }
                    )
    return rows


def _load_reference(
    path: Path, config: Mapping[str, Any]
) -> dict[tuple[str, str, int], dict[str, str]]:
    if file_sha256(path) != str(config["reference_metrics_sha256"]):
        raise ValueError("Reference metrics SHA-256 does not match the frozen input")
    reference: dict[tuple[str, str, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "date_utc",
            "instrument",
            "grid_timestamp_ms",
            "source_timestamp_ms",
            "availability_timestamp_ms",
            "available",
            *config["reference_fields"],
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Reference metrics are missing frozen fields")
        for row in reader:
            key = (row["date_utc"], row["instrument"], int(row["grid_timestamp_ms"]))
            if key in reference:
                raise ValueError(f"Duplicate reference key: {key}")
            reference[key] = row
    if len(reference) != int(config["expected_total_rows"]):
        raise ValueError("Reference row count does not match the frozen grid")
    return reference


def _compare_reference(
    rows: Sequence[Mapping[str, object]],
    reference: Mapping[tuple[str, str, int], Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, object]] = []
    comparisons = 0
    max_absolute_error = 0.0
    relative_tolerance = float(config["reference_relative_tolerance"])
    absolute_tolerance = float(config["reference_absolute_tolerance"])
    for row in rows:
        key = (str(row["date_utc"]), str(row["instrument"]), int(row["grid_timestamp_ms"]))
        expected = reference.get(key)
        if expected is None:
            mismatches.append({"key": key, "field": "row", "reason": "missing_reference"})
            continue
        availability_text = expected["available"].lower()
        if availability_text not in {"true", "false"}:
            mismatches.append(
                {"key": key, "field": "available", "reason": "invalid_reference_value"}
            )
            continue
        expected_available = availability_text == "true"
        if bool(row["available"]) != expected_available:
            mismatches.append({"key": key, "field": "available", "reason": "different"})
            continue
        if not expected_available:
            continue
        for field in (
            "source_timestamp_ms",
            "signed_offset_ms",
            "absolute_offset_ms",
            "availability_timestamp_ms",
        ):
            if int(row[field]) != int(expected[field]):
                mismatches.append({"key": key, "field": field, "reason": "different"})
        for field in config["reference_fields"]:
            actual_value = float(row[field])
            expected_value = float(expected[field])
            error = abs(actual_value - expected_value)
            max_absolute_error = max(max_absolute_error, error)
            comparisons += 1
            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "actual": actual_value,
                        "expected": expected_value,
                        "absolute_error": error,
                    }
                )
    return {
        "rows": len(reference),
        "numeric_comparisons": comparisons,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "maximum_absolute_error": max_absolute_error,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


def build_l2_feature_table(
    corpus_root: Path, reference_path: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    source_files = _validate_config(config)
    _validate_corpus_artifact(corpus_root, source_files, config)
    _validate_reference_artifact(reference_path, config)
    source_rows, source_summaries = _load_source_rows(corpus_root, source_files, config)
    rows = _ordered_feature_rows(source_rows, config)
    reference = _load_reference(reference_path, config)
    reference_check = _compare_reference(rows, reference, config)
    available_rows = sum(bool(row["available"]) for row in rows)
    missing_rows = len(rows) - available_rows
    actual_missing = [
        {
            "date_utc": row["date_utc"],
            "instrument": row["instrument"],
            "grid_timestamp_ms": row["grid_timestamp_ms"],
        }
        for row in rows
        if not row["available"]
    ]
    go = (
        len(rows) == int(config["expected_total_rows"])
        and available_rows == int(config["expected_available_rows"])
        and missing_rows == int(config["expected_missing_rows"])
        and actual_missing == list(config["expected_missing_keys"])
        and reference_check["mismatch_count"] == 0
    )
    result = {
        "schema_version": OKX_L2_FEATURES_SCHEMA_VERSION,
        "provider": config["provider"],
        "source": {
            "artifact_id": config["source_artifact_id"],
            "result_sha256": config["source_result_sha256"],
            "files": source_summaries,
        },
        "reference": {
            "artifact_id": config["reference_alignment_artifact_id"],
            "result_sha256": config["reference_alignment_result_sha256"],
            "metrics_sha256": config["reference_metrics_sha256"],
            **reference_check,
        },
        "table": {
            "rows": len(rows),
            "available_rows": available_rows,
            "missing_rows": missing_rows,
            "missing_keys": actual_missing,
            "feature_count": len(config["feature_fields"]),
            "feature_fields": list(config["feature_fields"]),
            "time_semantics": config["feature_time_semantics"],
            "missing_policy": config["missing_policy"],
            "target_policy": config["target_policy"],
        },
        "decision": "GO_TECHNICAL_L2_FEATURES" if go else "NO_GO_TECHNICAL_L2_FEATURES",
        "future_values_used": False,
        "temporal_aggregation_used": False,
        "normalization_used": False,
        "targets_created": False,
        "model_trained": False,
        "network_used": False,
        "orders_placed": False,
        "production_touched": False,
        "limitations": [
            "Features cover three isolated UTC days, not a continuous history",
            "No target, predictive evaluation, or economic evaluation",
            "Feature definitions are frozen before any outcome analysis",
        ],
    }
    return result, rows


def _feature_csv(rows: Sequence[Mapping[str, object]], feature_fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[*META_FIELDS, *feature_fields],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_l2_feature_artifact(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    feature_code_sha256: str,
    output_root: Path,
) -> Path:
    csv_payload = _feature_csv(rows, result["table"]["feature_fields"])
    identity = {
        **result,
        "config_sha256": config_sha256,
        "feature_code_sha256": feature_code_sha256,
        "feature_table_sha256": hashlib.sha256(csv_payload).hexdigest(),
    }
    artifact_id = (
        f"{OKX_L2_FEATURES_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"L2 feature artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        result_path = temporary_path / "l2_feature_result.json"
        table_path = temporary_path / "l2_features.csv"
        result_path.write_bytes(_json_bytes(identity, pretty=True))
        table_path.write_bytes(csv_payload)
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_file": result_path.name,
            "result_sha256": file_sha256(result_path),
            "feature_table_file": table_path.name,
            "feature_table_sha256": file_sha256(table_path),
            "targets_created": False,
            "model_trained": False,
            "network_used": False,
            "credentials_used": False,
            "orders_placed": False,
            "production_touched": False,
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
