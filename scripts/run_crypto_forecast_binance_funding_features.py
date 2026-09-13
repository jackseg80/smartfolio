"""Build and validate the frozen causal Binance funding feature table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.binance_funding_features import (  # noqa: E402
    build_funding_feature_result,
    file_sha256,
    write_funding_feature_artifact,
)


def code_bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_binance_funding_features.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-binance-funding-history-canonical"
            / "crypto-forecast-binance-funding-history-v1-6288d6a1dca91185"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-binance-funding-features",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result, rows = build_funding_feature_result(args.source_root, config)
    artifact = write_funding_feature_artifact(
        result,
        rows,
        config_sha256=file_sha256(args.config),
        feature_code_sha256=code_bundle_sha256(
            [
                PROJECT_ROOT / "services" / "forecasting" / "binance_funding_features.py",
                PROJECT_ROOT / "scripts" / "run_crypto_forecast_binance_funding_features.py",
            ]
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "rows": result["table"]["rows"],
                "model_eligible_rows": result["table"]["model_eligible_rows"],
                "mutation_tests": result["future_mutation_tests"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
