"""Bounded acquisition and snapshot-only analysis of an OKX L2 sample."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from services.forecasting.okx_l2_pilot import file_sha256
from services.forecasting.okx_l2_snapshot_pilot import analyze_snapshot_archive

OKX_L2_SAMPLE_SCHEMA_VERSION = "crypto-forecast-okx-l2-sample-v1"
MEBIBYTE = 1024 * 1024
ProgressCallback = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class ArchiveSpec:
    date_utc: str
    instrument: str
    filename: str
    advertised_size_mib: str
    url: str

    @property
    def advertised_size_bytes(self) -> int:
        return int(
            (Decimal(self.advertised_size_mib) * MEBIBYTE).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )


@dataclass(frozen=True)
class AcquiredArchive:
    spec: ArchiveSpec
    compressed_bytes: int
    sha256: str
    path: Path


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


def _positive_decimal(value: object, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"Invalid {field}: {value!r}")
    return number


def load_candidate_specs(metadata_path: Path, config: Mapping[str, Any]) -> list[ArchiveSpec]:
    if config.get("schema_version") != OKX_L2_SAMPLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported L2 sample schema: {config.get('schema_version')}")
    expected_hash = str(config["coverage_metadata_sha256"])
    if file_sha256(metadata_path) != expected_hash:
        raise ValueError("Coverage metadata SHA-256 does not match the frozen input")

    instruments = [str(value) for value in config["instruments"]]
    dates = [str(value) for value in config["dates_utc"]]
    expected = {(date, instrument) for date in dates for instrument in instruments}
    found: dict[tuple[str, str], ArchiveSpec] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"date_utc", "instrument", "filename", "size_mb", "url"}
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise ValueError("Coverage metadata is missing required columns")
        for row in reader:
            key = (str(row["date_utc"]), str(row["instrument"]))
            if key not in expected:
                continue
            if key in found:
                raise ValueError(f"Duplicate coverage metadata for {key[1]} on {key[0]}")
            expected_filename = f"{key[1]}-L2orderbook-400lv-{key[0]}.tar.gz"
            if row["filename"] != expected_filename:
                raise ValueError(f"Unexpected archive filename for {key[1]} on {key[0]}")
            parsed = urlsplit(str(row["url"]))
            if parsed.scheme != "https" or parsed.hostname != "static.okx.com":
                raise ValueError(f"Unexpected archive URL for {key[1]} on {key[0]}")
            size = _positive_decimal(row["size_mb"], "advertised archive size")
            spec = ArchiveSpec(
                date_utc=key[0],
                instrument=key[1],
                filename=expected_filename,
                advertised_size_mib=str(size),
                url=str(row["url"]),
            )
            if spec.advertised_size_bytes > int(config["maximum_single_file_bytes"]):
                raise ValueError(f"Advertised archive exceeds the frozen limit: {spec.filename}")
            found[key] = spec

    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"Frozen candidate metadata is incomplete: {missing}")
    specs = [found[key] for key in sorted(found)]
    if len(specs) != int(config["expected_archive_count"]):
        raise ValueError("Candidate archive count does not match the frozen contract")
    if sum(item.advertised_size_bytes for item in specs) > int(
        config["total_download_limit_bytes"]
    ):
        raise ValueError("Advertised sample exceeds the frozen total download limit")
    return specs


def validate_actual_size(spec: ArchiveSpec, actual_bytes: int, config: Mapping[str, Any]) -> None:
    if actual_bytes <= 0 or actual_bytes > int(config["maximum_single_file_bytes"]):
        raise ValueError(f"Downloaded archive violates the single-file limit: {spec.filename}")
    difference = abs(actual_bytes - spec.advertised_size_bytes)
    if difference > int(config["advertised_size_tolerance_bytes"]):
        raise ValueError(f"Downloaded size does not match frozen metadata: {spec.filename}")


def _check_free_space(
    raw_root: Path, specs: Sequence[ArchiveSpec], config: Mapping[str, Any]
) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    anticipated = sum(item.advertised_size_bytes for item in specs)
    free = shutil.disk_usage(raw_root).free
    required = anticipated + int(config["minimum_free_bytes_after_download"])
    if free < required:
        raise OSError(f"Insufficient disk space: {free} free bytes, {required} required")


def _download_one(
    client: httpx.Client,
    spec: ArchiveSpec,
    raw_root: Path,
    config: Mapping[str, Any],
    progress: ProgressCallback,
) -> AcquiredArchive:
    destination = raw_root / spec.filename
    if destination.exists():
        actual_bytes = destination.stat().st_size
        validate_actual_size(spec, actual_bytes, config)
        digest = file_sha256(destination)
        progress(
            {
                "event": "archive_reused",
                "filename": spec.filename,
                "bytes": actual_bytes,
                "sha256": digest,
            }
        )
        return AcquiredArchive(spec, actual_bytes, digest, destination)

    temporary = raw_root / f"{spec.filename}.part"
    retries = int(config["download_retries"])
    for attempt in range(1, retries + 1):
        if temporary.exists():
            temporary.unlink()
        progress(
            {
                "event": "download_started",
                "filename": spec.filename,
                "attempt": attempt,
                "advertised_bytes": spec.advertised_size_bytes,
            }
        )
        try:
            digest = hashlib.sha256()
            downloaded = 0
            with client.stream("GET", spec.url) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    validate_actual_size(spec, int(content_length), config)
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(int(config["download_chunk_bytes"])):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > int(config["maximum_single_file_bytes"]):
                            raise ValueError(
                                f"Download crossed the single-file limit: {spec.filename}"
                            )
                        handle.write(chunk)
                        digest.update(chunk)
            validate_actual_size(spec, downloaded, config)
            temporary.replace(destination)
            archive = AcquiredArchive(spec, downloaded, digest.hexdigest(), destination)
            progress(
                {
                    "event": "download_completed",
                    "filename": spec.filename,
                    "bytes": downloaded,
                    "sha256": archive.sha256,
                }
            )
            return archive
        except (httpx.HTTPError, OSError) as exc:
            if temporary.exists():
                temporary.unlink()
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {spec.filename}"
                ) from exc
            time.sleep(attempt)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
    raise RuntimeError(f"Download attempts exhausted: {spec.filename}")


def _acquisition_identity(
    archives: Sequence[AcquiredArchive], config: Mapping[str, Any]
) -> dict[str, Any]:
    files = []
    for archive in sorted(archives, key=lambda item: (item.spec.date_utc, item.spec.instrument)):
        files.append(
            {
                **asdict(archive.spec),
                "compressed_bytes": archive.compressed_bytes,
                "sha256": archive.sha256,
            }
        )
    return {
        "schema_version": OKX_L2_SAMPLE_SCHEMA_VERSION,
        "provider": config["provider"],
        "coverage_artifact_id": config["coverage_artifact_id"],
        "coverage_metadata_sha256": config["coverage_metadata_sha256"],
        "files": files,
        "archive_count": len(files),
        "total_compressed_bytes": sum(item["compressed_bytes"] for item in files),
        "credentials_used": False,
        "orders_placed": False,
    }


def _write_acquisition_manifest(
    archives: Sequence[AcquiredArchive], raw_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    identity = _acquisition_identity(archives, config)
    manifest = {
        **identity,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retention_policy": config["retention_policy"],
        "production_touched": False,
    }
    temporary = raw_root / "acquisition_manifest.json.part"
    destination = raw_root / "acquisition_manifest.json"
    temporary.write_bytes(_json_bytes(manifest, pretty=True))
    temporary.replace(destination)
    return manifest


def download_archives(
    specs: Sequence[ArchiveSpec],
    raw_root: Path,
    config: Mapping[str, Any],
    *,
    progress: ProgressCallback,
) -> tuple[list[AcquiredArchive], dict[str, Any]]:
    _check_free_space(raw_root, specs, config)
    timeout = httpx.Timeout(float(config["download_timeout_seconds"]))
    archives: list[AcquiredArchive] = []
    with httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": "SmartFolio-Offline-Research/1.0"},
    ) as client:
        for spec in specs:
            archives.append(_download_one(client, spec, raw_root, config, progress))
            if sum(item.compressed_bytes for item in archives) > int(
                config["total_download_limit_bytes"]
            ):
                raise ValueError("Downloaded sample crossed the frozen total limit")
    return archives, _write_acquisition_manifest(archives, raw_root, config)


def load_verified_archives(
    specs: Sequence[ArchiveSpec], raw_root: Path, config: Mapping[str, Any]
) -> tuple[list[AcquiredArchive], dict[str, Any]]:
    manifest_path = raw_root / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != OKX_L2_SAMPLE_SCHEMA_VERSION:
        raise ValueError("Unsupported acquisition manifest schema")
    if manifest.get("provider") != config["provider"]:
        raise ValueError("Acquisition manifest provider does not match the frozen contract")
    if manifest.get("coverage_artifact_id") != config["coverage_artifact_id"]:
        raise ValueError("Acquisition manifest references an unexpected coverage artifact")
    if manifest.get("coverage_metadata_sha256") != config["coverage_metadata_sha256"]:
        raise ValueError("Acquisition manifest does not reference the frozen metadata")

    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise ValueError("Acquisition manifest files must be a list")
    entries: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in raw_entries:
        if not isinstance(item, Mapping):
            raise ValueError("Acquisition manifest contains an invalid file entry")
        key = (str(item.get("date_utc")), str(item.get("instrument")))
        if key in entries:
            raise ValueError(f"Duplicate acquisition manifest entry: {key}")
        entries[key] = item

    expected_keys = {(item.date_utc, item.instrument) for item in specs}
    if set(entries) != expected_keys:
        raise ValueError("Acquisition manifest contains an unexpected archive set")
    if int(manifest.get("archive_count", -1)) != len(specs):
        raise ValueError("Acquisition manifest archive count is inconsistent")

    archives: list[AcquiredArchive] = []
    for spec in specs:
        entry = entries.get((spec.date_utc, spec.instrument))
        if (
            entry is None
            or entry.get("filename") != spec.filename
            or entry.get("url") != spec.url
            or entry.get("advertised_size_mib") != spec.advertised_size_mib
        ):
            raise ValueError(f"Acquisition manifest mismatch for {spec.filename}")
        path = raw_root / spec.filename
        actual_bytes = path.stat().st_size
        validate_actual_size(spec, actual_bytes, config)
        digest = file_sha256(path)
        if actual_bytes != int(entry["compressed_bytes"]) or digest != entry["sha256"]:
            raise ValueError(f"Pinned archive verification failed: {spec.filename}")
        archives.append(AcquiredArchive(spec, actual_bytes, digest, path))
    if int(manifest.get("total_compressed_bytes", -1)) != sum(
        item.compressed_bytes for item in archives
    ):
        raise ValueError("Acquisition manifest total size is inconsistent")
    return archives, manifest


def _snapshot_config(archive: AcquiredArchive, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "crypto-forecast-okx-l2-snapshot-pilot-v1",
        "provider": config["provider"],
        "instrument": archive.spec.instrument,
        "date_utc": archive.spec.date_utc,
        "expected_archive_sha256": archive.sha256,
        "download_limit_bytes": config["maximum_single_file_bytes"],
        "uncompressed_limit_bytes": config["maximum_uncompressed_member_bytes"],
        "max_records": config["max_records_per_archive"],
        "max_line_bytes": config["max_line_bytes"],
        "expected_snapshot_count": config["expected_snapshot_count_per_archive"],
        "snapshot_interval_ms": config["snapshot_interval_ms"],
        "snapshot_interval_tolerance_ms": config["snapshot_interval_tolerance_ms"],
        "maximum_levels_per_side": config["maximum_levels_per_side"],
        "depth_bands_bps": config["depth_bands_bps"],
    }


def analyze_sample(
    archives: Sequence[AcquiredArchive],
    config: Mapping[str, Any],
    *,
    progress: ProgressCallback,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    results: list[dict[str, Any]] = []
    metrics: list[dict[str, object]] = []
    for archive in archives:
        progress(
            {
                "event": "analysis_started",
                "filename": archive.spec.filename,
            }
        )
        result, rows = analyze_snapshot_archive(archive.path, _snapshot_config(archive, config))
        results.append(result)
        metrics.extend(
            {
                "date_utc": archive.spec.date_utc,
                "instrument": archive.spec.instrument,
                **row,
            }
            for row in rows
        )
        progress(
            {
                "event": "analysis_completed",
                "filename": archive.spec.filename,
                "decision": result["decision"],
                "snapshots": result["snapshots"]["total"],
            }
        )
    return results, metrics


def build_sample_result(
    archive_results: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, object]],
    acquisition_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_archives = int(config["expected_archive_count"])
    expected_snapshots = int(config["expected_total_snapshot_count"])
    technical_go_count = sum(item.get("decision") == "GO_TECHNICAL" for item in archive_results)
    valid_snapshots = sum(
        int(item.get("snapshots", {}).get("valid_uncrossed", 0)) for item in archive_results
    )
    updates_ignored = sum(
        int(item.get("records", {}).get("updates_ignored_for_features", 0))
        for item in archive_results
    )
    acquisition_identity = {
        key: value
        for key, value in acquisition_manifest.items()
        if key not in {"generated_at_utc", "retention_policy", "production_touched"}
    }
    go = (
        len(archive_results) == expected_archives
        and technical_go_count == expected_archives
        and len(metrics) == expected_snapshots
        and valid_snapshots == expected_snapshots
    )
    return {
        "schema_version": OKX_L2_SAMPLE_SCHEMA_VERSION,
        "provider": config["provider"],
        "sample": {
            "instruments": list(config["instruments"]),
            "dates_utc": list(config["dates_utc"]),
            "expected_archives": expected_archives,
            "analyzed_archives": len(archive_results),
            "technical_go_archives": technical_go_count,
            "expected_snapshots": expected_snapshots,
            "valid_snapshots": valid_snapshots,
            "updates_ignored_for_features": updates_ignored,
        },
        "acquisition": {
            "archive_count": acquisition_manifest["archive_count"],
            "total_compressed_bytes": acquisition_manifest["total_compressed_bytes"],
            "identity_sha256": hashlib.sha256(_json_bytes(acquisition_identity)).hexdigest(),
            "credentials_used": False,
        },
        "archive_results": list(archive_results),
        "decision": "GO_TECHNICAL" if go else "NO_GO_TECHNICAL",
        "updates_used_for_features": False,
        "raw_members_extracted_to_disk": False,
        "model_trained": False,
        "orders_placed": False,
        "production_touched": False,
        "limitations": [
            "Three instruments and three UTC days only",
            "Snapshot features only; all update events are ignored",
            "No predictive model or economic evaluation",
            "No interpolation between snapshots",
        ],
    }


def _metrics_csv(metrics: Sequence[Mapping[str, object]]) -> bytes:
    if not metrics:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(metrics[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(metrics)
    return buffer.getvalue().encode("utf-8")


def _archive_summary_csv(results: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = [
        "date_utc",
        "instrument",
        "filename",
        "compressed_bytes",
        "sha256",
        "records",
        "snapshots",
        "valid_snapshots",
        "updates_ignored",
        "decision",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "date_utc": result["date_utc"],
                "instrument": result["instrument"],
                "filename": result["archive"]["filename"],
                "compressed_bytes": result["archive"]["compressed_bytes"],
                "sha256": result["archive"]["sha256"],
                "records": result["records"]["total"],
                "snapshots": result["snapshots"]["total"],
                "valid_snapshots": result["snapshots"]["valid_uncrossed"],
                "updates_ignored": result["records"]["updates_ignored_for_features"],
                "decision": result["decision"],
            }
        )
    return buffer.getvalue().encode("utf-8")


def write_sample_artifact(
    result: Mapping[str, Any],
    metrics: Sequence[Mapping[str, object]],
    *,
    config_sha256: str,
    sample_code_sha256: str,
    snapshot_code_sha256: str,
    output_root: Path,
) -> Path:
    metrics_payload = _metrics_csv(metrics)
    archive_payload = _archive_summary_csv(result["archive_results"])
    identity = {
        **result,
        "config_sha256": config_sha256,
        "sample_code_sha256": sample_code_sha256,
        "snapshot_code_sha256": snapshot_code_sha256,
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        "archive_summary_sha256": hashlib.sha256(archive_payload).hexdigest(),
    }
    artifact_id = (
        f"{OKX_L2_SAMPLE_SCHEMA_VERSION}-"
        f"{hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_id
    if destination.exists():
        raise FileExistsError(f"L2 sample artifact already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "sample_result.json").write_bytes(_json_bytes(identity, pretty=True))
        (temporary_path / "snapshot_metrics.csv").write_bytes(metrics_payload)
        (temporary_path / "archive_summary.csv").write_bytes(archive_payload)
        manifest = {
            "artifact_id": artifact_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_archive_retention": "retained_until_explicit_cleanup_authorization",
            "credentials_used": False,
            "orders_placed": False,
            "raw_members_extracted_to_disk": False,
            "updates_used_for_features": False,
            "model_trained": False,
            "production_touched": False,
            "result_file": "sample_result.json",
            "result_sha256": file_sha256(temporary_path / "sample_result.json"),
            "metrics_file": "snapshot_metrics.csv",
            "metrics_sha256": file_sha256(temporary_path / "snapshot_metrics.csv"),
            "archive_summary_file": "archive_summary.csv",
            "archive_summary_sha256": file_sha256(temporary_path / "archive_summary.csv"),
        }
        (temporary_path / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        Path(temporary).replace(destination)
    return destination
