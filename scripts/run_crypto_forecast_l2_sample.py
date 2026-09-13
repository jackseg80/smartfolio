"""Acquire or reproduce the frozen multi-asset OKX L2 snapshot sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402
from services.forecasting.okx_l2_sample import (  # noqa: E402
    analyze_sample,
    build_sample_result,
    download_archives,
    load_candidate_specs,
    load_verified_archives,
    write_sample_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_sample.json",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-l2-coverage"
            / "crypto-forecast-okx-l2-coverage-v1-9b584745ea2d87c6"
            / "archive_metadata.csv"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-sample" / "raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-sample" / "analysis",
    )
    parser.add_argument(
        "--reuse-verified-raw",
        action="store_true",
        help="Reanalyze the pinned local archives without network requests.",
    )
    return parser.parse_args()


def _progress(event: dict[str, object]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    specs = load_candidate_specs(args.metadata, config)
    if args.reuse_verified_raw:
        archives, acquisition_manifest = load_verified_archives(specs, args.raw_root, config)
        _progress({"event": "verified_raw_reused", "archives": len(archives)})
    else:
        archives, acquisition_manifest = download_archives(
            specs,
            args.raw_root,
            config,
            progress=_progress,
        )
    archive_results, metrics = analyze_sample(archives, config, progress=_progress)
    result = build_sample_result(archive_results, metrics, acquisition_manifest, config)
    artifact = write_sample_artifact(
        result,
        metrics,
        config_sha256=file_sha256(args.config),
        sample_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_sample.py"
        ),
        snapshot_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_snapshot_pilot.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "archives": len(archives),
                "snapshots": len(metrics),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
