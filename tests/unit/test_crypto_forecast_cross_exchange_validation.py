import csv
import io
from pathlib import Path

import pandas as pd

from services.forecasting.cross_exchange_validation import (
    _aligned_csv_bytes,
    compare_asset,
    run_validation,
    stable_manifest_sha256,
)


def _frame(closes, volumes):
    return pd.DataFrame(
        {"close": closes, "quote_volume": volumes},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


def _config(minimum=3):
    return {
        "schema_version": "crypto-forecast-cross-exchange-validation-v1",
        "minimum_common_observations": minimum,
        "minimum_daily_return_correlation": 0.99,
        "maximum_median_absolute_close_difference_bps": 50.0,
        "maximum_p95_absolute_close_difference_bps": 300.0,
        "volume_change_horizons_days": [1, 2],
    }


def test_compare_asset_uses_exact_common_dates_and_reports_pass():
    binance = _frame([100, 101, 103, 102], [1000, 1100, 1300, 1200])
    okx = _frame([100.1, 101.1, 103.1, 102.1], [500, 560, 650, 610]).iloc[1:]

    summary, aligned = compare_asset("BTC", binance, okx, config=_config())

    assert summary["common_observations"] == 3
    assert summary["common_start"] == "2024-01-02"
    assert summary["price_consistency_passed"] is True
    assert aligned["symbol"].eq("BTC").all()


def test_compare_asset_keeps_failed_threshold_visible():
    binance = _frame([100, 101, 102], [1000, 1100, 1200])
    okx = _frame([100, 150, 80], [500, 600, 700])

    summary, _aligned = compare_asset("BTC", binance, okx, config=_config())

    assert summary["price_consistency_passed"] is False
    assert summary["quality_checks"]["maximum_median_absolute_close_difference_bps"] is False


def test_run_validation_rejects_wrong_schema_before_reading_inputs(tmp_path: Path):
    config = {**_config(), "schema_version": "wrong"}

    try:
        run_validation(
            binance_root=tmp_path / "binance",
            okx_root=tmp_path / "okx",
            config=config,
            config_sha256="config",
            validation_code_sha256="code",
            output_root=tmp_path / "out",
        )
    except ValueError as exc:
        assert "Unsupported validation schema" in str(exc)
    else:
        raise AssertionError("Expected invalid schema to be rejected")


def test_stable_manifest_hash_ignores_only_generation_timestamp():
    first = {"artifact_id": "same", "generated_at_utc": "2026-09-13T10:00:00Z"}
    second = {"artifact_id": "same", "generated_at_utc": "2026-09-13T11:00:00Z"}

    assert stable_manifest_sha256(first) == stable_manifest_sha256(second)
    assert stable_manifest_sha256(first) != stable_manifest_sha256(
        {"artifact_id": "changed", "generated_at_utc": first["generated_at_utc"]}
    )


def test_aligned_csv_serializes_initial_returns_as_empty_cells():
    binance = _frame([100, 101], [1000, 1100])
    okx = _frame([100.1, 101.1], [500, 550])
    _summary, aligned = compare_asset("BTC", binance, okx, config=_config(minimum=2))

    payload = _aligned_csv_bytes([aligned]).decode("utf-8")
    rows = list(csv.reader(io.StringIO(payload)))

    assert all(cell.lower() not in {"nan", "inf", "-inf"} for row in rows for cell in row)
    assert rows[1][6:8] == ["", ""]
