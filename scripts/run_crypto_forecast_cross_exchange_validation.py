"""Compare verified Binance and OKX daily artifacts without fitting a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.forecasting.cross_exchange_validation import (  # noqa: E402
    file_sha256,
    run_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binance-dir", type=Path, required=True)
    parser.add_argument("--okx-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_cross_exchange.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-cross-exchange",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = run_validation(
        binance_root=args.binance_dir,
        okx_root=args.okx_dir,
        config=config,
        config_sha256=file_sha256(args.config),
        validation_code_sha256=file_sha256(
            PROJECT_ROOT / "services" / "forecasting" / "cross_exchange_validation.py"
        ),
        output_root=args.output_root,
    )
    results = json.loads((artifact / "results.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "all_symbols_passed_price_consistency": results[
                    "all_symbols_passed_price_consistency"
                ],
                "aligned_daily_sha256": results["aligned_daily_sha256"],
                "validation_identity_sha256": results["validation_identity_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
