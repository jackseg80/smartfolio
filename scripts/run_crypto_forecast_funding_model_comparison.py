"""Run the frozen baseline-versus-funding predictive comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy  # noqa: E402
import pandas  # noqa: E402
import sklearn  # noqa: E402

from services.forecasting.funding_model_comparison import (  # noqa: E402
    load_comparison_panel,
    run_funding_model_comparison,
    write_comparison_artifact,
)
from services.forecasting.evaluation import file_sha256  # noqa: E402


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
        default=PROJECT_ROOT / "config" / "crypto_forecast_funding_model_comparison.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-lot2-volume"
            / "crypto-forecast-dataset-v1-6f1490cfee2de334"
        ),
    )
    parser.add_argument(
        "--funding-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "crypto-forecast-binance-funding-features-canonical"
            / "crypto-forecast-binance-funding-features-v1-f32e0489dbb258fd"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-funding-model-comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    panel, baseline_features, funding_features = load_comparison_panel(
        args.dataset_root, args.funding_root, config
    )
    result, predictions = run_funding_model_comparison(
        panel, baseline_features, funding_features, config
    )
    result["runtime_versions"] = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }
    artifact = write_comparison_artifact(
        result,
        predictions,
        config_sha256=file_sha256(args.config),
        comparison_code_sha256=code_bundle_sha256(
            [
                PROJECT_ROOT / "services" / "forecasting" / "funding_model_comparison.py",
                PROJECT_ROOT / "scripts" / "run_crypto_forecast_funding_model_comparison.py",
            ]
        ),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact.resolve()),
                "decision": result["decision"],
                "horizons": [
                    {
                        "horizon_days": item["horizon_days"],
                        **item["assessment"],
                    }
                    for item in result["horizons"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
