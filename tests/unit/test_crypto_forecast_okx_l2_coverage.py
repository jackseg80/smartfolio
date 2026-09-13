import json
from decimal import Decimal
from pathlib import Path

import pytest

from services.forecasting.okx_l2_coverage import (
    ArchiveMetadata,
    build_coverage_result,
    parse_metadata_response,
    write_coverage_artifact,
)


def _file(instrument: str, date: str, size: str = "10.5") -> dict[str, str]:
    compact = date.replace("-", "")
    return {
        "dateTs": str(
            int(
                __import__("datetime").datetime.fromisoformat(f"{date}T00:00:00+00:00").timestamp()
                * 1000
            )
        ),
        "filename": f"{instrument}-L2orderbook-400lv-{date}.tar.gz",
        "sizeMB": size,
        "url": (
            "https://static.okx.com/cdn/okx/match/orderbook/pro/L2/400lv/"
            f"daily/{compact}/{instrument}-L2orderbook-400lv-{date}.tar.gz"
        ),
    }


def _payload(date: str, instruments: list[str]) -> dict[str, object]:
    return {
        "code": "0",
        "msg": "",
        "data": [
            {
                "dateAggrType": "daily",
                "details": [
                    {"instId": instrument, "groupDetails": [_file(instrument, date)]}
                    for instrument in instruments
                ],
            }
        ],
    }


def _config() -> dict[str, object]:
    return {
        "schema_version": "crypto-forecast-okx-l2-coverage-v1",
        "provider": "fixture",
        "endpoint": "/metadata",
        "module": 4,
        "instrument_type": "SPOT",
        "date_aggregation": "daily",
        "instruments": ["BTC-USDT", "ETH-USDT"],
        "dates_utc": ["2026-01-01", "2026-04-01"],
        "candidate_dates_utc": ["2026-01-01"],
        "minimum_total_files": 3,
        "minimum_dates_per_instrument": 1,
        "candidate_download_limit_mb": "100",
    }


def test_parse_metadata_preserves_missing_without_replacing_dates():
    records, missing = parse_metadata_response(
        _payload("2026-01-01", ["BTC-USDT"]),
        date_utc="2026-01-01",
        instruments=["BTC-USDT", "ETH-USDT"],
        maximum_single_file_mb=Decimal("1000"),
    )

    assert [record.instrument for record in records] == ["BTC-USDT"]
    assert missing == ["ETH-USDT"]
    assert records[0].size_mb == "10.5"


def test_parse_metadata_rejects_unofficial_url_and_invalid_size():
    payload = _payload("2026-01-01", ["BTC-USDT"])
    payload["data"][0]["details"][0]["groupDetails"][0]["url"] = "https://example.com/file"
    with pytest.raises(ValueError, match="archive URL"):
        parse_metadata_response(
            payload,
            date_utc="2026-01-01",
            instruments=["BTC-USDT"],
            maximum_single_file_mb=Decimal("1000"),
        )

    payload = _payload("2026-01-01", ["BTC-USDT"])
    payload["data"][0]["details"][0]["groupDetails"][0]["sizeMB"] = "0"
    with pytest.raises(ValueError, match="archive size"):
        parse_metadata_response(
            payload,
            date_utc="2026-01-01",
            instruments=["BTC-USDT"],
            maximum_single_file_mb=Decimal("1000"),
        )


def test_result_and_artifact_are_stable(tmp_path: Path):
    records = []
    for date in ("2026-01-01", "2026-04-01"):
        parsed, _missing = parse_metadata_response(
            _payload(date, ["BTC-USDT", "ETH-USDT"]),
            date_utc=date,
            instruments=["BTC-USDT", "ETH-USDT"],
            maximum_single_file_mb=Decimal("1000"),
        )
        records.extend(parsed)
    result = build_coverage_result(records, [], config=_config())

    first = write_coverage_artifact(
        records=records,
        result=result,
        config_sha256="config-hash",
        acquisition_code_sha256="code-hash",
        output_root=tmp_path / "first",
    )
    second = write_coverage_artifact(
        records=records,
        result=result,
        config_sha256="config-hash",
        acquisition_code_sha256="code-hash",
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert (first / "coverage_result.json").read_bytes() == (
        second / "coverage_result.json"
    ).read_bytes()
    assert result["decision"] == "GO_BOUNDED_SAMPLE_METADATA"
    assert result["archive_downloads_performed"] is False
