#!/usr/bin/env python3
"""Build the lot-2 causal crypto forecasting dataset from local price caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.dataset import (  # noqa: E402
    UniverseMember,
    build_forecast_dataset,
    load_price_cache,
    write_dataset_artifact,
)


def _load_universe(path: Path | None) -> list[UniverseMember] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Universe manifest must contain a JSON list")
    return [UniverseMember(**item) for item in payload]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible causal dataset without downloading market data."
    )
    parser.add_argument(
        "--price-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "price_history",
        help="Directory containing SYMBOL_1d.json daily-close histories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-lot2",
        help="Root for the content-addressed generated artifact.",
    )
    parser.add_argument(
        "--universe-manifest",
        type=Path,
        help="Optional dated universe JSON. Cache-derived membership is used when omitted.",
    )
    args = parser.parse_args()

    loaded = load_price_cache(args.price_dir)
    build = build_forecast_dataset(
        loaded.histories,
        universe=_load_universe(args.universe_manifest),
        input_metadata=loaded.inputs,
    )
    artifact = write_dataset_artifact(build, args.output_dir)
    summary = {
        "artifact": str(artifact.resolve()),
        "dataset_version": build.manifest["dataset_version"],
        "rows": len(build.frame),
        "rejected_input_files": loaded.rejected_files,
        "coverage": build.coverage,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
