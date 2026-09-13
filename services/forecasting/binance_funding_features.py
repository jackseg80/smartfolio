"""Causal daily features from checksum-verified Binance funding events."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

FUNDING_FEATURE_SCHEMA_VERSION = "crypto-forecast-binance-funding-features-v1"
SOURCE_SCHEMA_VERSION = "crypto-forecast-binance-funding-history-v1"
EVENT_HEADER = (
    "timestamp_utc",
    "calc_time_ms",
    "utc_date",
    "funding_interval_hours",
    "funding_rate",
)
META_FIELDS = [
    "date_utc",
    "symbol",
    "market_symbol",
    "decision_timestamp_ms",
    "latest_source_timestamp_ms",
    "observations_today",
    "history_days",
    "model_eligible",
]
FEATURE_FIELDS = [
    "funding_daily_sum",
    "funding_daily_mean",
    "funding_daily_last",
    "funding_daily_min",
    "funding_daily_max",
    "funding_daily_std",
    "funding_positive_share",
    "funding_interval_mean_hours",
    "funding_sum_3d",
    "funding_sum_7d",
    "funding_sum_30d",
    "funding_mean_7d",
    "funding_mean_30d",
    "funding_daily_sum_change_1d",
    "funding_daily_sum_change_7d",
    "funding_sum_zscore_30d",
]


@dataclass(frozen=True)
class FundingEvent:
    calc_time_ms: int
    utc_date: str
    funding_interval_hours: int
    funding_rate: str


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


def _date_range(start: date, end: date) -> list[str]:
    return [
        (start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)
    ]


def _decision_timestamp_ms(day: str) -> int:
    next_day = date.fromisoformat(day) + timedelta(days=1)
    timestamp = datetime.combine(next_day, time.min, tzinfo=timezone.utc)
    return int(timestamp.timestamp() * 1000)


def _safe_source_path(source_root: Path, relative_text: str) -> Path:
    if "\\" in relative_text:
        raise ValueError("Source paths must use POSIX separators")
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "events":
        raise ValueError("Source path must be a safe events/*.csv path")
    if ".." in relative.parts or relative.suffix != ".csv":
        raise ValueError("Source path must be a safe events/*.csv path")
    resolved_root = source_root.resolve()
    resolved_path = resolved_root.joinpath(*relative.parts).resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError("Source path escapes the funding artifact")
    return resolved_path


def validate_config(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if config.get("schema_version") != FUNDING_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported funding feature schema: {config.get('schema_version')}")
    if (
        config.get("decision_time_semantics")
        != "end_of_utc_day_with_events_strictly_before_next_midnight"
    ):
        raise ValueError("Decision time semantics must enforce the next UTC midnight cutoff")
    if config.get("missing_policy") != "explicit_empty_value_until_full_window":
        raise ValueError("Incomplete windows must remain explicit and empty")
    if config.get("normalization_policy") != "none_in_this_lot":
        raise ValueError("Normalization is forbidden in this feature lot")
    if config.get("target_policy") != "no_target_or_future_return_in_this_lot":
        raise ValueError("Targets and future returns are forbidden in this feature lot")
    if list(config.get("rolling_windows_days", [])) != [3, 7, 30]:
        raise ValueError("Rolling windows must match the frozen 3/7/30-day contract")
    if list(config.get("feature_fields", [])) != FEATURE_FIELDS:
        raise ValueError("Feature fields do not match the frozen contract")
    if int(config["warmup_days"]) != 30:
        raise ValueError("Warm-up must match the frozen 30-day contract")

    start = date.fromisoformat(str(config["start_date"]))
    end = date.fromisoformat(str(config["end_date"]))
    calendar_days = (end - start).days + 1
    if calendar_days != int(config["expected_calendar_days"]):
        raise ValueError("Expected calendar days do not match the frozen period")
    source_files = config.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != int(
        config["expected_instruments"]
    ):
        raise ValueError("Source files do not match the expected instrument count")
    markets = [str(item["market_symbol"]) for item in source_files]
    symbols = [str(item["symbol"]) for item in source_files]
    if markets != sorted(markets) or len(set(markets)) != len(markets):
        raise ValueError("Source markets must be unique and sorted")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Source symbols must be unique")
    for item in source_files:
        path = PurePosixPath(str(item["path"]))
        if path != PurePosixPath("events") / f"{item['market_symbol']}_funding.csv":
            raise ValueError("Source file name does not match its market")
        if len(str(item["sha256"])) != 64 or int(item["observations"]) <= 0:
            raise ValueError("Source file contract is incomplete")
    expected_rows = calendar_days * len(source_files)
    if int(config["expected_rows"]) != expected_rows:
        raise ValueError("Expected feature rows do not match the frozen grid")
    expected_eligible = (calendar_days - int(config["warmup_days"]) + 1) * len(source_files)
    if int(config["expected_model_eligible_rows"]) != expected_eligible:
        raise ValueError("Expected eligible rows do not match the warm-up contract")
    cutoffs = [date.fromisoformat(str(value)) for value in config["future_mutation_cutoff_dates"]]
    if (
        len(cutoffs) != 3
        or cutoffs != sorted(set(cutoffs))
        or any(not start <= value < end for value in cutoffs)
    ):
        raise ValueError("Future-mutation cutoffs must be unique, sorted, and inside the period")
    return source_files


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _parse_event_file(path: Path, expected: Mapping[str, Any]) -> tuple[FundingEvent, ...]:
    if path.stat().st_size > int(expected["maximum_bytes"]):
        raise ValueError(f"Source event file exceeds the frozen byte limit: {path.name}")
    events: list[FundingEvent] = []
    previous_time: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EVENT_HEADER:
            raise ValueError(f"Unexpected event schema: {path.name}")
        for row_number, row in enumerate(reader, start=2):
            try:
                calc_time_ms = int(row["calc_time_ms"])
                timestamp = datetime.fromisoformat(str(row["timestamp_utc"]).replace("Z", "+00:00"))
                event_date = str(row["utc_date"])
                interval_hours = int(row["funding_interval_hours"])
                funding_rate = str(row["funding_rate"])
                rate = Decimal(funding_rate)
            except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
                raise ValueError(f"Invalid funding event at {path.name}:{row_number}") from exc
            actual = datetime.fromtimestamp(calc_time_ms / 1000, timezone.utc)
            actual_date = actual.date().isoformat()
            if (
                timestamp.tzinfo is None
                or timestamp.utcoffset() != timedelta(0)
                or timestamp != actual
                or event_date != actual_date
                or not rate.is_finite()
                or not 1 <= interval_hours <= 24
            ):
                raise ValueError(f"Invalid funding semantics at {path.name}:{row_number}")
            if previous_time is not None and calc_time_ms <= previous_time:
                raise ValueError(f"Funding events are not strictly increasing: {path.name}")
            events.append(FundingEvent(calc_time_ms, event_date, interval_hours, funding_rate))
            previous_time = calc_time_ms
    if len(events) != int(expected["observations"]):
        raise ValueError(f"Funding event count differs from the frozen input: {path.name}")
    return tuple(events)


def load_funding_events(
    source_root: Path, config: Mapping[str, Any]
) -> dict[str, tuple[FundingEvent, ...]]:
    source_files = validate_config(config)
    manifest_path = source_root / "acquisition_manifest.json"
    if file_sha256(manifest_path) != str(config["source_manifest_sha256"]):
        raise ValueError("Source manifest SHA-256 does not match the frozen input")
    manifest = _load_json_object(manifest_path, "funding acquisition manifest")
    if (
        manifest.get("schema_version") != SOURCE_SCHEMA_VERSION
        or manifest.get("artifact_id") != config["source_artifact_id"]
        or manifest.get("start_date") != config["start_date"]
        or manifest.get("end_date") != config["end_date"]
        or manifest.get("targets_created") is not False
        or manifest.get("predictive_features_created") is not False
    ):
        raise ValueError("Source artifact is not the frozen target-free funding input")
    manifest_inputs = {str(item["market_symbol"]): item for item in manifest.get("inputs", [])}
    if set(manifest_inputs) != {str(item["market_symbol"]) for item in source_files}:
        raise ValueError("Source manifest instruments differ from the frozen contract")

    start = str(config["start_date"])
    end = str(config["end_date"])
    histories: dict[str, tuple[FundingEvent, ...]] = {}
    for item in source_files:
        market = str(item["market_symbol"])
        manifest_input = manifest_inputs[market]
        if (
            manifest_input.get("symbol") != item["symbol"]
            or manifest_input.get("events_file") != item["path"]
            or manifest_input.get("events_file_sha256") != item["sha256"]
            or int(manifest_input.get("observations", -1)) != int(item["observations"])
        ):
            raise ValueError(f"Source manifest contract differs for {market}")
        path = _safe_source_path(source_root, str(item["path"]))
        if file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Source event SHA-256 mismatch: {market}")
        expected = {**item, "maximum_bytes": config["maximum_event_file_bytes"]}
        events = _parse_event_file(path, expected)
        dates = {event.utc_date for event in events}
        if (
            min(dates) != start
            or max(dates) != end
            or len(dates) != int(config["expected_calendar_days"])
        ):
            raise ValueError(f"Source event coverage differs for {market}")
        histories[market] = events
    return histories


def _window(values: Sequence[float], size: int) -> list[float] | None:
    if len(values) < size:
        return None
    return list(values[-size:])


def build_feature_rows(
    histories: Mapping[str, Sequence[FundingEvent]], config: Mapping[str, Any]
) -> list[dict[str, object]]:
    source_files = validate_config(config)
    start = date.fromisoformat(str(config["start_date"]))
    end = date.fromisoformat(str(config["end_date"]))
    dates = _date_range(start, end)
    rows: list[dict[str, object]] = []
    symbols_by_market = {str(item["market_symbol"]): str(item["symbol"]) for item in source_files}
    if set(histories) != set(symbols_by_market):
        raise ValueError("Funding histories differ from the frozen markets")
    for market in symbols_by_market:
        events = sorted(histories[market], key=lambda item: item.calc_time_ms)
        by_date: dict[str, list[FundingEvent]] = defaultdict(list)
        for event in events:
            by_date[event.utc_date].append(event)
        daily_sums: list[float] = []
        for index, day in enumerate(dates):
            day_events = by_date.get(day, [])
            if not day_events:
                raise ValueError(f"No funding events for {market} on {day}")
            decision_timestamp = _decision_timestamp_ms(day)
            latest_source = max(item.calc_time_ms for item in day_events)
            if latest_source >= decision_timestamp:
                raise ValueError(f"Funding event is unavailable at decision time: {market} {day}")
            rates = [float(Decimal(item.funding_rate)) for item in day_events]
            if any(not math.isfinite(value) for value in rates):
                raise ValueError(f"Non-finite funding input for {market} on {day}")
            daily_sum = sum(rates)
            daily_sums.append(daily_sum)
            window_3 = _window(daily_sums, 3)
            window_7 = _window(daily_sums, 7)
            window_30 = _window(daily_sums, 30)
            features: dict[str, object] = {
                "funding_daily_sum": daily_sum,
                "funding_daily_mean": statistics.fmean(rates),
                "funding_daily_last": rates[-1],
                "funding_daily_min": min(rates),
                "funding_daily_max": max(rates),
                "funding_daily_std": statistics.pstdev(rates),
                "funding_positive_share": sum(value > 0 for value in rates) / len(rates),
                "funding_interval_mean_hours": statistics.fmean(
                    item.funding_interval_hours for item in day_events
                ),
                "funding_sum_3d": sum(window_3) if window_3 else "",
                "funding_sum_7d": sum(window_7) if window_7 else "",
                "funding_sum_30d": sum(window_30) if window_30 else "",
                "funding_mean_7d": statistics.fmean(window_7) if window_7 else "",
                "funding_mean_30d": statistics.fmean(window_30) if window_30 else "",
                "funding_daily_sum_change_1d": (
                    daily_sum - daily_sums[index - 1] if index >= 1 else ""
                ),
                "funding_daily_sum_change_7d": (
                    daily_sum - daily_sums[index - 7] if index >= 7 else ""
                ),
                "funding_sum_zscore_30d": "",
            }
            if window_30:
                mean_30 = statistics.fmean(window_30)
                std_30 = statistics.pstdev(window_30)
                features["funding_sum_zscore_30d"] = (
                    (daily_sum - mean_30) / std_30 if std_30 else 0.0
                )
            if list(features) != FEATURE_FIELDS:
                raise ValueError("Computed feature order differs from the frozen contract")
            for value in features.values():
                if value != "" and not math.isfinite(float(value)):
                    raise ValueError(f"A computed feature is non-finite: {market} {day}")
            history_days = index + 1
            model_eligible = history_days >= int(config["warmup_days"])
            if model_eligible and any(value == "" for value in features.values()):
                raise ValueError(f"Eligible row contains an empty feature: {market} {day}")
            rows.append(
                {
                    "date_utc": day,
                    "symbol": symbols_by_market[market],
                    "market_symbol": market,
                    "decision_timestamp_ms": decision_timestamp,
                    "latest_source_timestamp_ms": latest_source,
                    "observations_today": len(day_events),
                    "history_days": history_days,
                    "model_eligible": model_eligible,
                    **features,
                }
            )
    if len(rows) != int(config["expected_rows"]):
        raise ValueError("Feature row count differs from the frozen contract")
    eligible = sum(bool(row["model_eligible"]) for row in rows)
    if eligible != int(config["expected_model_eligible_rows"]):
        raise ValueError("Eligible feature row count differs from the frozen contract")
    return rows


def mutate_future_events(
    histories: Mapping[str, Sequence[FundingEvent]], cutoff_date: str, end_date: str
) -> dict[str, tuple[FundingEvent, ...]]:
    cutoff = date.fromisoformat(cutoff_date)
    end = date.fromisoformat(end_date)
    mutated: dict[str, tuple[FundingEvent, ...]] = {}
    for market, events in histories.items():
        changed = []
        for event in events:
            if date.fromisoformat(event.utc_date) > cutoff:
                changed.append(
                    replace(
                        event,
                        funding_interval_hours=(event.funding_interval_hours % 24) + 1,
                        funding_rate=str(Decimal(event.funding_rate) + Decimal("1.23456789")),
                    )
                )
            else:
                changed.append(event)
        cursor = cutoff + timedelta(days=1)
        while cursor <= end:
            day = cursor.isoformat()
            changed.append(
                FundingEvent(
                    calc_time_ms=_decision_timestamp_ms(day) - 1,
                    utc_date=day,
                    funding_interval_hours=1,
                    funding_rate="9.87654321",
                )
            )
            cursor += timedelta(days=1)
        mutated[market] = tuple(sorted(changed, key=lambda item: item.calc_time_ms))
    return mutated


def run_future_mutation_tests(
    histories: Mapping[str, Sequence[FundingEvent]],
    baseline_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, Any],
) -> list[dict[str, object]]:
    baseline = {
        (str(row["market_symbol"]), str(row["date_utc"])): dict(row) for row in baseline_rows
    }
    results = []
    for cutoff in config["future_mutation_cutoff_dates"]:
        cutoff_text = str(cutoff)
        mutated_histories = mutate_future_events(histories, cutoff_text, str(config["end_date"]))
        mutated_rows = build_feature_rows(mutated_histories, config)
        mutated = {
            (str(row["market_symbol"]), str(row["date_utc"])): dict(row) for row in mutated_rows
        }
        compared_keys = [key for key in baseline if key[1] <= cutoff_text]
        protected_divergences = sum(baseline[key] != mutated[key] for key in compared_keys)
        future_keys = [key for key in baseline if key[1] > cutoff_text]
        future_changed_rows = sum(baseline[key] != mutated[key] for key in future_keys)
        results.append(
            {
                "cutoff_date": cutoff_text,
                "protected_rows_compared": len(compared_keys),
                "protected_divergences": protected_divergences,
                "future_rows_changed": future_changed_rows,
                "future_mutation_effective": future_changed_rows > 0,
            }
        )
    return results


def build_funding_feature_result(
    source_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    histories = load_funding_events(source_root, config)
    rows = build_feature_rows(histories, config)
    mutation_tests = run_future_mutation_tests(histories, rows, config)
    go = all(
        item["protected_divergences"] == 0 and item["future_mutation_effective"]
        for item in mutation_tests
    )
    result = {
        "schema_version": FUNDING_FEATURE_SCHEMA_VERSION,
        "provider": config["provider"],
        "source": {
            "artifact_id": config["source_artifact_id"],
            "manifest_sha256": config["source_manifest_sha256"],
            "files": [
                {
                    "symbol": item["symbol"],
                    "market_symbol": item["market_symbol"],
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "observations": item["observations"],
                }
                for item in config["source_files"]
            ],
        },
        "table": {
            "rows": len(rows),
            "instruments": int(config["expected_instruments"]),
            "calendar_days": int(config["expected_calendar_days"]),
            "feature_count": len(FEATURE_FIELDS),
            "feature_fields": FEATURE_FIELDS,
            "model_eligible_rows": sum(bool(row["model_eligible"]) for row in rows),
            "warmup_days": int(config["warmup_days"]),
            "decision_time_semantics": config["decision_time_semantics"],
            "missing_policy": config["missing_policy"],
        },
        "future_mutation_tests": mutation_tests,
        "decision": "GO_CAUSAL_FUNDING_FEATURES" if go else "NO_GO_CAUSAL_FUNDING_FEATURES",
        "future_values_used": False,
        "normalization_used": False,
        "targets_created": False,
        "model_trained": False,
        "network_used": False,
        "orders_placed": False,
        "production_touched": False,
        "limitations": [
            "Funding features are not predictive evidence",
            "No target, model, or economic evaluation is included",
            "The 29 warm-up days per instrument are not model eligible",
        ],
    }
    return result, rows


def _feature_csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[*META_FIELDS, *FEATURE_FIELDS], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_funding_feature_artifact(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    feature_code_sha256: str,
    output_root: Path,
) -> Path:
    table_payload = _feature_csv_bytes(rows)
    identity = {
        **result,
        "config_sha256": config_sha256,
        "feature_code_sha256": feature_code_sha256,
        "feature_table_sha256": hashlib.sha256(table_payload).hexdigest(),
    }
    artifact_id = (
        f"{FUNDING_FEATURE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"Funding feature artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        result_path = temporary_path / "funding_feature_result.json"
        table_path = temporary_path / "funding_features.csv"
        result_path.write_bytes(_json_bytes(identity, pretty=True))
        table_path.write_bytes(table_payload)
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
