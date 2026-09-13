"""Build the frozen compact OKX L2 corpus from nine local archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.okx_l2_compact_corpus import (  # noqa: E402
    build_compact_corpus,
    write_compact_corpus_artifact,
)
from services.forecasting.okx_l2_pilot import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_okx_l2_compact_corpus.json",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-sample" / "raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-l2-compact-corpus",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result, payloads, performance = build_compact_corpus(args.raw_root, config)
    artifact = write_compact_corpus_artifact(
        result,
        payloads,
        performance,
        config_sha256=file_sha256(args.config),
        corpus_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_compact_corpus.py"
        ),
        extraction_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "okx_l2_progressive_extraction.py"
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "selected_snapshots": result["totals"]["selected_snapshots"],
                "missing_snapshots": result["totals"]["missing_snapshots"],
                "compact_compressed_bytes": result["totals"]["compact_compressed_bytes"],
                "elapsed_seconds": performance["total_elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
