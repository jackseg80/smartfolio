"""Run lot-3 crypto forecast comparisons on a frozen lot-2 artifact."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid a noisy Windows hardware probe in joblib. The experiment is deliberately
# single-process and the model implementations remain deterministic.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy  # noqa: E402
import pandas  # noqa: E402
import sklearn  # noqa: E402

from services.forecasting.evaluation import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
    load_verified_dataset,
    run_experiments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "crypto_forecast_experiment.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "crypto-forecast-lot3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset, dataset_manifest = load_verified_dataset(args.dataset_dir)
    code_path = PROJECT_ROOT / "services" / "forecasting" / "evaluation.py"
    runner_path = Path(__file__).resolve()
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }
    identity_payload = {
        "schema_version": config["schema_version"],
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "config_sha256": file_sha256(args.config),
        "evaluation_code_sha256": file_sha256(code_path),
        "runner_code_sha256": file_sha256(runner_path),
        "runtime_versions": runtime_versions,
    }
    experiment_id = f"{config['schema_version']}-{canonical_json_sha256(identity_payload)[:16]}"
    output_directory = args.output_root / experiment_id
    output_directory.mkdir(parents=True, exist_ok=False)

    results, predictions = run_experiments(dataset, config)
    results.update(
        {
            "experiment_id": experiment_id,
            "dataset": {
                "dataset_version": dataset_manifest["dataset_version"],
                "dataset_sha256": dataset_manifest["dataset_sha256"],
                "rows": int(len(dataset)),
                "source_directory": str(args.dataset_dir.resolve()),
            },
        }
    )
    results_path = output_directory / "results.json"
    predictions_path = output_directory / "predictions.csv"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    manifest = {
        **identity_payload,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_file": results_path.name,
        "results_sha256": file_sha256(results_path),
        "predictions_file": predictions_path.name,
        "predictions_sha256": file_sha256(predictions_path),
        "selection_used_final_holdout": False,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output_directory": str(output_directory), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
