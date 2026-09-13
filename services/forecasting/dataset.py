"""Reproducible, causal crypto forecasting dataset builder.

The builder consumes dated daily-close observations only. It never downloads,
forward-fills, or selects a nearest observation. Future labels are kept in
columns that are explicitly separate from information available at decision
time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from shared.asset_groups import get_asset_group

SCHEMA_VERSION = "crypto-forecast-dataset-v1"
HORIZONS = (7, 30)
BASE_FEATURE_COLUMNS = (
    "past_return_7d",
    "past_return_30d",
    "past_return_90d",
    "distance_sma_30d",
    "distance_sma_90d",
    "distance_sma_200d",
    "past_volatility_7d",
    "past_volatility_30d",
    "past_volatility_60d",
    "drawdown_from_90d_peak",
)
VOLUME_FEATURE_COLUMNS = (
    "quote_volume_change_7d",
    "quote_volume_change_30d",
    "quote_volume_to_30d_mean",
    "trade_count_to_30d_mean",
)
RELATIVE_FEATURE_COLUMNS = (
    "relative_btc_return_7d",
    "relative_btc_return_30d",
    "relative_btc_return_90d",
    "relative_group_return_7d",
    "relative_group_return_30d",
    "relative_group_return_90d",
)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS + RELATIVE_FEATURE_COLUMNS


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    group: str
    known_from: str
    known_until: str | None = None
    delisted: bool = False
    membership_provenance: str = "inferred_from_first_cached_observation"
    eligible_for_forecasting: bool = True
    exclusion_reason: str | None = None

    def active_on(self, decision_date: pd.Timestamp) -> bool:
        start = pd.Timestamp(self.known_from)
        end = pd.Timestamp(self.known_until) if self.known_until else None
        return decision_date >= start and (end is None or decision_date <= end)


@dataclass(frozen=True)
class PriceCacheLoad:
    histories: dict[str, pd.Series]
    market_data: dict[str, pd.DataFrame]
    inputs: list[dict[str, Any]]
    rejected_files: list[dict[str, str]]


@dataclass(frozen=True)
class DatasetBuild:
    frame: pd.DataFrame
    universe: list[UniverseMember]
    manifest: dict[str, Any]
    coverage: dict[str, Any]


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_series(raw: pd.Series | Iterable[Sequence[object]]) -> pd.Series:
    if isinstance(raw, pd.Series):
        values = pd.to_numeric(raw, errors="coerce")
        index = pd.to_datetime(raw.index, utc=True, errors="coerce")
    else:
        pairs = list(raw)
        if not pairs:
            return pd.Series(dtype=float)
        index = pd.to_datetime([item[0] for item in pairs], unit="s", utc=True, errors="coerce")
        values = pd.Series(pd.to_numeric([item[1] for item in pairs], errors="coerce"))
    frame = pd.DataFrame({"date": index, "price": np.asarray(values, dtype=float)})
    frame = frame.dropna(subset=["date", "price"])
    frame = frame[np.isfinite(frame["price"]) & (frame["price"] > 0)]
    if frame.empty:
        return pd.Series(dtype=float)
    frame["date"] = pd.DatetimeIndex(frame["date"]).normalize().tz_localize(None)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return pd.Series(frame["price"].to_numpy(dtype=float), index=frame["date"], dtype=float)


def load_price_cache(price_directory: str | Path) -> PriceCacheLoad:
    """Load local daily-close JSON files without network access or mutation."""
    directory = Path(price_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Price history directory does not exist: {directory}")
    acquisition_path = directory / "acquisition_manifest.json"
    if not acquisition_path.is_file():
        acquisition_path = directory.parent / "acquisition_manifest.json"
    provenance_by_file: dict[str, dict[str, Any]] = {}
    acquisition_sha256: str | None = None
    if acquisition_path.is_file():
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        if acquisition.get("schema_version") != "crypto-forecast-history-acquisition-v1":
            raise ValueError(f"Unsupported acquisition manifest: {acquisition_path}")
        acquisition_sha256 = _file_sha256(acquisition_path)
        for record in acquisition.get("inputs", []):
            relative = Path(str(record["price_file"]))
            provenance_by_file[relative.name] = dict(record)
    histories: dict[str, pd.Series] = {}
    market_data: dict[str, pd.DataFrame] = {}
    inputs: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(directory.glob("*_1d.json"), key=lambda item: item.name.upper()):
        symbol = path.name[: -len("_1d.json")].upper()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            series = _normalise_series(raw)
            if series.empty:
                raise ValueError("no finite positive observations")
            provenance = provenance_by_file.get(path.name)
            actual_hash = _file_sha256(path)
            if provenance and provenance.get("price_file_sha256") != actual_hash:
                raise ValueError(f"Acquisition manifest hash mismatch for {path.name}")
            histories[symbol] = series
            metadata = {
                "symbol": symbol,
                "file": path.name,
                "sha256": actual_hash,
                "observations": int(len(series)),
                "first_date": series.index.min().date().isoformat(),
                "last_date": series.index.max().date().isoformat(),
                "provider_provenance": "unavailable_in_legacy_cache_format",
            }
            if provenance:
                ohlcv_path = acquisition_path.parent / str(provenance["ohlcv_file"])
                actual_ohlcv_hash = _file_sha256(ohlcv_path)
                if provenance.get("ohlcv_file_sha256") != actual_ohlcv_hash:
                    raise ValueError(f"Acquisition manifest OHLCV hash mismatch for {path.name}")
                ohlcv = pd.read_csv(
                    ohlcv_path,
                    usecols=["date", "close", "quote_asset_volume", "trades"],
                )
                ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="raise").dt.normalize()
                for column in ("close", "quote_asset_volume", "trades"):
                    ohlcv[column] = pd.to_numeric(ohlcv[column], errors="raise")
                ohlcv = ohlcv.sort_values("date").drop_duplicates("date", keep="last")
                ohlcv = ohlcv.set_index("date")
                aligned_close = ohlcv["close"].reindex(series.index)
                if aligned_close.isna().any() or not np.allclose(
                    aligned_close.to_numpy(dtype=float), series.to_numpy(dtype=float), rtol=1e-12
                ):
                    raise ValueError(f"OHLCV close series does not match {path.name}")
                market_data[symbol] = ohlcv[["quote_asset_volume", "trades"]].copy()
                metadata.update(
                    {
                        "provider_provenance": provenance["provider_provenance"],
                        "market_symbol": provenance["market_symbol"],
                        "quote_asset": provenance["quote_asset"],
                        "exchange_status_at_acquisition": provenance[
                            "exchange_status_at_acquisition"
                        ],
                        "ohlcv_file": provenance["ohlcv_file"],
                        "ohlcv_file_sha256": provenance["ohlcv_file_sha256"],
                        "acquisition_manifest_sha256": acquisition_sha256,
                    }
                )
            inputs.append(metadata)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejected.append({"file": path.name, "reason": str(exc)})
    if "BTC" not in histories:
        raise ValueError("BTC history is required as the common market reference")
    return PriceCacheLoad(
        histories=histories,
        market_data=market_data,
        inputs=inputs,
        rejected_files=rejected,
    )


def _history_inputs(histories: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    result = []
    for symbol, raw in sorted(histories.items()):
        series = _normalise_series(raw)
        canonical = [
            [timestamp.date().isoformat(), format(float(price), ".17g")]
            for timestamp, price in series.items()
        ]
        result.append(
            {
                "symbol": symbol.upper(),
                "sha256": _sha256_bytes(_json_bytes(canonical)),
                "observations": int(len(series)),
                "first_date": series.index.min().date().isoformat() if len(series) else None,
                "last_date": series.index.max().date().isoformat() if len(series) else None,
                "provider_provenance": "caller_supplied",
            }
        )
    return result


def _make_universe(
    histories: Mapping[str, pd.Series],
    group_for_symbol: Callable[[str], str],
    excluded_groups: set[str],
) -> list[UniverseMember]:
    members = []
    for symbol, series in sorted(histories.items()):
        group = str(group_for_symbol(symbol))
        excluded = group in excluded_groups
        members.append(
            UniverseMember(
                symbol=symbol,
                group=group,
                known_from=series.index.min().date().isoformat(),
                eligible_for_forecasting=not excluded,
                exclusion_reason=(
                    "legacy stablecoin histories may be generated; defensive cash is reference-only"
                    if excluded
                    else None
                ),
            )
        )
    return members


def _validate_universe(
    universe: Sequence[UniverseMember], histories: Mapping[str, pd.Series]
) -> list[UniverseMember]:
    seen: set[str] = set()
    validated = []
    for raw in universe:
        member = UniverseMember(**asdict(raw))
        symbol = member.symbol.strip().upper()
        if not symbol or symbol in seen:
            raise ValueError(f"Universe symbols must be unique and non-empty: {symbol!r}")
        seen.add(symbol)
        start = pd.Timestamp(member.known_from).normalize()
        end = pd.Timestamp(member.known_until).normalize() if member.known_until else None
        if end is not None and end < start:
            raise ValueError(f"known_until precedes known_from for {symbol}")
        validated.append(
            UniverseMember(
                **{
                    **asdict(member),
                    "symbol": symbol,
                    "known_from": start.date().isoformat(),
                    "known_until": end.date().isoformat() if end is not None else None,
                }
            )
        )
    return sorted(validated, key=lambda item: (item.group, item.symbol))


def _asset_features(
    series: pd.Series,
    calendar: pd.DatetimeIndex,
    market_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prices = series.reindex(calendar)
    daily_return = prices.pct_change(fill_method=None)
    result = pd.DataFrame(index=calendar)
    for window in (7, 30, 90):
        result[f"past_return_{window}d"] = prices / prices.shift(window) - 1.0
    for window in (30, 90, 200):
        moving_average = prices.rolling(window, min_periods=window).mean()
        result[f"distance_sma_{window}d"] = prices / moving_average - 1.0
    for window in (7, 30, 60):
        result[f"past_volatility_{window}d"] = daily_return.rolling(window, min_periods=window).std(
            ddof=1
        )
    peak = prices.rolling(90, min_periods=90).max()
    result["drawdown_from_90d_peak"] = prices / peak - 1.0
    if market_data is not None:
        quote_volume = pd.to_numeric(
            market_data["quote_asset_volume"].reindex(calendar), errors="coerce"
        )
        trades = pd.to_numeric(market_data["trades"].reindex(calendar), errors="coerce")
        result["quote_volume_change_7d"] = quote_volume / quote_volume.shift(7) - 1.0
        result["quote_volume_change_30d"] = quote_volume / quote_volume.shift(30) - 1.0
        result["quote_volume_to_30d_mean"] = (
            quote_volume / quote_volume.rolling(30, min_periods=30).mean() - 1.0
        )
        result["trade_count_to_30d_mean"] = trades / trades.rolling(30, min_periods=30).mean() - 1.0
    else:
        for column in VOLUME_FEATURE_COLUMNS:
            result[column] = np.nan
    result["decision_price"] = prices
    return result.replace([np.inf, -np.inf], np.nan)


def _mean_complete(values: Iterable[object]) -> float | None:
    parsed = []
    for value in values:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        parsed.append(number)
    return float(np.mean(parsed)) if parsed else None


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _label(value: float | None) -> int | None:
    return int(value > 0.0) if value is not None else None


def _status_for_features(row: Mapping[str, object]) -> tuple[str, str]:
    missing = [column for column in FEATURE_COLUMNS if _float_or_none(row.get(column)) is None]
    return (
        "complete" if not missing else "unavailable",
        json.dumps(missing, separators=(",", ":")),
    )


def _base_row(
    *,
    dataset_version: str,
    decision_date: pd.Timestamp,
    scope: str,
    entity: str,
    group: str,
    member_count: int,
    membership_provenance: str,
) -> dict[str, object]:
    return {
        "dataset_version": dataset_version,
        "decision_date": decision_date.date().isoformat(),
        "information_cutoff": f"{decision_date.date().isoformat()}T23:59:59Z",
        "scope": scope,
        "entity": entity,
        "group": group,
        "universe_member_count": member_count,
        "membership_provenance": membership_provenance,
        "defensive_reference": "USD_cash_zero_total_return_assumption",
    }


def _future_return(series: pd.Series, decision_date: pd.Timestamp, horizon: int) -> float | None:
    exit_date = decision_date + pd.Timedelta(days=horizon)
    if decision_date not in series.index or exit_date not in series.index:
        return None
    start = _float_or_none(series.at[decision_date])
    end = _float_or_none(series.at[exit_date])
    if start is None or end is None or start <= 0:
        return None
    return end / start - 1.0


def _active_members(
    members: Sequence[UniverseMember], decision_date: pd.Timestamp
) -> list[UniverseMember]:
    return [member for member in members if member.active_on(decision_date)]


def _group_future_return(
    members: Sequence[UniverseMember],
    histories: Mapping[str, pd.Series],
    decision_date: pd.Timestamp,
    horizon: int,
) -> tuple[float | None, list[str]]:
    returns: list[float] = []
    missing = []
    for member in members:
        value = _future_return(histories[member.symbol], decision_date, horizon)
        if value is None:
            missing.append(member.symbol)
        else:
            returns.append(value)
    if missing or not returns:
        return None, sorted(missing)
    return float(np.mean(returns)), []


def _set_target_columns(
    row: dict[str, object],
    *,
    horizon: int,
    own_return: float | None,
    btc_return: float | None,
    group_return: float | None,
    required: Sequence[str],
    missing_exit_members: Sequence[str] = (),
) -> None:
    row[f"target_return_{horizon}d"] = own_return
    row[f"target_excess_defensive_{horizon}d"] = own_return
    relative_btc = (
        own_return - btc_return if own_return is not None and btc_return is not None else None
    )
    relative_group = (
        own_return - group_return if own_return is not None and group_return is not None else None
    )
    row[f"target_relative_btc_{horizon}d"] = relative_btc
    row[f"target_relative_group_{horizon}d"] = relative_group
    row[f"label_up_{horizon}d"] = _label(own_return)
    row[f"label_outperform_defensive_{horizon}d"] = _label(own_return)
    row[f"label_outperform_btc_{horizon}d"] = _label(relative_btc)
    row[f"label_outperform_group_{horizon}d"] = _label(relative_group)

    missing_required = [name for name in required if _float_or_none(row.get(name)) is None]
    if missing_required and own_return is not None:
        status = "partial"
    elif missing_required:
        status = "unavailable"
    else:
        status = "complete"
    row[f"target_status_{horizon}d"] = status
    row[f"target_unavailable_fields_{horizon}d"] = json.dumps(
        missing_required, separators=(",", ":")
    )
    row[f"missing_exit_members_{horizon}d"] = json.dumps(
        sorted(missing_exit_members), separators=(",", ":")
    )


def _coverage(frame: pd.DataFrame, universe: Sequence[UniverseMember]) -> dict[str, Any]:
    scopes: dict[str, Any] = {}
    for scope, scoped in frame.groupby("scope", sort=True):
        target_coverage = {}
        for horizon in HORIZONS:
            relevant = [f"target_return_{horizon}d"]
            if scope == "market":
                relevant.append(f"target_excess_defensive_{horizon}d")
            elif scope == "group":
                relevant.append(f"target_relative_btc_{horizon}d")
            else:
                relevant.extend(
                    [
                        f"target_relative_group_{horizon}d",
                        f"target_relative_btc_{horizon}d",
                    ]
                )
            target_coverage[f"{horizon}d"] = {
                column: {
                    "available_rows": int(scoped[column].notna().sum()),
                    "coverage_pct": round(float(scoped[column].notna().mean() * 100.0), 4),
                }
                for column in relevant
            }
        scopes[str(scope)] = {
            "rows": int(len(scoped)),
            "entities": int(scoped["entity"].nunique()),
            "first_decision_date": str(scoped["decision_date"].min()),
            "last_decision_date": str(scoped["decision_date"].max()),
            "complete_feature_rows": int((scoped["feature_status"] == "complete").sum()),
            "complete_feature_coverage_pct": round(
                float((scoped["feature_status"] == "complete").mean() * 100.0), 4
            ),
            "target_coverage": target_coverage,
        }
    return {
        "rows": int(len(frame)),
        "first_decision_date": str(frame["decision_date"].min()),
        "last_decision_date": str(frame["decision_date"].max()),
        "scopes": scopes,
        "universe": {
            "members": len(universe),
            "eligible_members": sum(member.eligible_for_forecasting for member in universe),
            "excluded_members": sum(not member.eligible_for_forecasting for member in universe),
            "explicitly_delisted_members": sum(member.delisted for member in universe),
            "membership_provenance_counts": {
                str(key): int(value)
                for key, value in pd.Series([member.membership_provenance for member in universe])
                .value_counts()
                .sort_index()
                .items()
            },
        },
    }


def build_forecast_dataset(
    histories: Mapping[str, pd.Series | Iterable[Sequence[object]]],
    *,
    universe: Sequence[UniverseMember] | None = None,
    group_for_symbol: Callable[[str], str] = get_asset_group,
    excluded_groups: Iterable[str] = ("Stablecoins",),
    input_metadata: Sequence[Mapping[str, Any]] | None = None,
    market_data: Mapping[str, pd.DataFrame] | None = None,
) -> DatasetBuild:
    """Build market, group, and asset rows on one UTC daily calendar."""
    normalised: dict[str, pd.Series] = {}
    for raw_symbol, history in histories.items():
        symbol = raw_symbol.strip().upper()
        if not symbol:
            continue
        series = _normalise_series(history)
        if not series.empty:
            normalised[symbol] = series
    if "BTC" not in normalised:
        raise ValueError("BTC history is required as the common market reference")
    excluded = set(excluded_groups)
    members = _validate_universe(
        universe or _make_universe(normalised, group_for_symbol, excluded), normalised
    )
    eligible = [member for member in members if member.eligible_for_forecasting]
    if not eligible:
        raise ValueError("The forecasting universe has no eligible members")

    inputs = (
        [dict(item) for item in input_metadata] if input_metadata else _history_inputs(normalised)
    )
    volume_available = bool(market_data) and all(
        member.symbol in market_data for member in eligible
    )
    required_feature_columns = (
        FEATURE_COLUMNS if volume_available else BASE_FEATURE_COLUMNS + RELATIVE_FEATURE_COLUMNS
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "builder_code_sha256": _file_sha256(Path(__file__)),
        "horizons_calendar_days": list(HORIZONS),
        "decision_timezone": "UTC",
        "decision_frequency": "calendar_day",
        "information_cutoff": "daily close through decision_date inclusive",
        "future_label_origin": "decision_date close",
        "execution_convention": "execution is evaluated later at the next daily close",
        "defensive_reference": {
            "name": "USD cash",
            "return": 0.0,
            "kind": "explicit_assumption_not_observed_market_data",
        },
        "group_basket": "equal_weight_members_known_at_decision_weights_held_to_horizon",
        "missing_member_policy": "mark_unavailable_never_drop_and_renormalize",
        "excluded_groups": sorted(excluded),
        "volume_liquidity": (
            "dated_quote_volume_and_trade_count_used_as_causal_features"
            if volume_available
            else "unavailable_not_in_legacy_dated_cache"
        ),
        "probability_columns": "none_binary_event_labels_only_until_calibration",
    }
    universe_payload = [asdict(member) for member in members]
    version_hash = _sha256_bytes(
        _json_bytes({"contract": contract, "inputs": inputs, "universe": universe_payload})
    )
    dataset_version = f"{SCHEMA_VERSION}-{version_hash[:16]}"

    first = min(pd.Timestamp(member.known_from) for member in eligible)
    last = max(series.index.max() for series in normalised.values())
    calendar = pd.date_range(first, last, freq="D")
    for member in eligible:
        normalised.setdefault(member.symbol, pd.Series(dtype=float))
    features = {
        symbol: _asset_features(
            series,
            calendar,
            market_data.get(symbol) if market_data is not None else None,
        )
        for symbol, series in normalised.items()
    }
    forward = {
        symbol: {
            horizon: series.reindex(calendar).shift(-horizon) / series.reindex(calendar) - 1.0
            for horizon in HORIZONS
        }
        for symbol, series in normalised.items()
    }

    by_group: dict[str, list[UniverseMember]] = {}
    for member in eligible:
        by_group.setdefault(member.group, []).append(member)

    group_features: dict[str, pd.DataFrame] = {}
    group_active: dict[str, dict[pd.Timestamp, list[UniverseMember]]] = {}
    for group, group_members in sorted(by_group.items()):
        group_frame = pd.DataFrame(
            index=calendar,
            columns=BASE_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS,
            dtype=float,
        )
        active_by_date: dict[pd.Timestamp, list[UniverseMember]] = {}
        for decision_date in calendar:
            active = _active_members(group_members, decision_date)
            active_by_date[decision_date] = active
            if not active:
                continue
            for column in BASE_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS:
                value = _mean_complete(
                    features[member.symbol].at[decision_date, column] for member in active
                )
                if value is not None:
                    group_frame.at[decision_date, column] = value
        group_features[group] = group_frame
        group_active[group] = active_by_date

    btc_features = features["BTC"]
    rows: list[dict[str, object]] = []

    def add_feature_status(row: dict[str, object]) -> None:
        decision_price = _float_or_none(row.get("decision_price"))
        if decision_price is None:
            row["history_status"] = "missing_decision_price"
        elif any(_float_or_none(row.get(column)) is None for column in BASE_FEATURE_COLUMNS):
            row["history_status"] = "insufficient_history"
        else:
            row["history_status"] = "available"
        missing = [
            column for column in required_feature_columns if _float_or_none(row.get(column)) is None
        ]
        row["feature_status"] = "complete" if not missing else "unavailable"
        row["feature_unavailable_fields"] = json.dumps(missing, separators=(",", ":"))

    btc_member = next(member for member in eligible if member.symbol == "BTC")
    for decision_date in calendar:
        if not btc_member.active_on(decision_date):
            continue
        row = _base_row(
            dataset_version=dataset_version,
            decision_date=decision_date,
            scope="market",
            entity="BTC",
            group="BTC",
            member_count=1,
            membership_provenance=btc_member.membership_provenance,
        )
        for column in BASE_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS:
            row[column] = _float_or_none(btc_features.at[decision_date, column])
        for window in (7, 30, 90):
            base = row[f"past_return_{window}d"]
            row[f"relative_btc_return_{window}d"] = 0.0 if base is not None else None
            row[f"relative_group_return_{window}d"] = 0.0 if base is not None else None
        row["decision_price"] = _float_or_none(btc_features.at[decision_date, "decision_price"])
        add_feature_status(row)
        for horizon in HORIZONS:
            own = _float_or_none(forward["BTC"][horizon].at[decision_date])
            _set_target_columns(
                row,
                horizon=horizon,
                own_return=own,
                btc_return=own,
                group_return=own,
                required=(f"target_return_{horizon}d", f"target_excess_defensive_{horizon}d"),
            )
        rows.append(row)

    for group, group_members in sorted(by_group.items()):
        frame = group_features[group]
        for decision_date in calendar:
            active = group_active[group][decision_date]
            if not active:
                continue
            row = _base_row(
                dataset_version=dataset_version,
                decision_date=decision_date,
                scope="group",
                entity=group,
                group=group,
                member_count=len(active),
                membership_provenance="dated_member_union_no_silent_removal",
            )
            for column in BASE_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS:
                row[column] = _float_or_none(frame.at[decision_date, column])
            for window in (7, 30, 90):
                own = row[f"past_return_{window}d"]
                btc = _float_or_none(btc_features.at[decision_date, f"past_return_{window}d"])
                row[f"relative_btc_return_{window}d"] = (
                    own - btc if own is not None and btc is not None else None
                )
                row[f"relative_group_return_{window}d"] = 0.0 if own is not None else None
            row["decision_price"] = _mean_complete(
                features[member.symbol].at[decision_date, "decision_price"] for member in active
            )
            add_feature_status(row)
            for horizon in HORIZONS:
                own, missing = _group_future_return(active, normalised, decision_date, horizon)
                btc = _float_or_none(forward["BTC"][horizon].at[decision_date])
                _set_target_columns(
                    row,
                    horizon=horizon,
                    own_return=own,
                    btc_return=btc,
                    group_return=own,
                    required=(f"target_return_{horizon}d", f"target_relative_btc_{horizon}d"),
                    missing_exit_members=missing,
                )
            rows.append(row)

    for member in sorted(eligible, key=lambda item: item.symbol):
        symbol_features = features[member.symbol]
        for decision_date in calendar:
            if not member.active_on(decision_date):
                continue
            active = group_active[member.group][decision_date]
            row = _base_row(
                dataset_version=dataset_version,
                decision_date=decision_date,
                scope="asset",
                entity=member.symbol,
                group=member.group,
                member_count=len(active),
                membership_provenance=member.membership_provenance,
            )
            for column in BASE_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS:
                row[column] = _float_or_none(symbol_features.at[decision_date, column])
            for window in (7, 30, 90):
                own = row[f"past_return_{window}d"]
                btc = _float_or_none(btc_features.at[decision_date, f"past_return_{window}d"])
                group_value = _float_or_none(
                    group_features[member.group].at[decision_date, f"past_return_{window}d"]
                )
                row[f"relative_btc_return_{window}d"] = (
                    own - btc if own is not None and btc is not None else None
                )
                row[f"relative_group_return_{window}d"] = (
                    own - group_value if own is not None and group_value is not None else None
                )
            row["decision_price"] = _float_or_none(
                symbol_features.at[decision_date, "decision_price"]
            )
            add_feature_status(row)
            for horizon in HORIZONS:
                own = _float_or_none(forward[member.symbol][horizon].at[decision_date])
                btc = _float_or_none(forward["BTC"][horizon].at[decision_date])
                group_value, missing = _group_future_return(
                    active, normalised, decision_date, horizon
                )
                _set_target_columns(
                    row,
                    horizon=horizon,
                    own_return=own,
                    btc_return=btc,
                    group_return=group_value,
                    required=(
                        f"target_return_{horizon}d",
                        f"target_relative_group_{horizon}d",
                        f"target_relative_btc_{horizon}d",
                    ),
                    missing_exit_members=missing,
                )
            rows.append(row)

    frame = (
        pd.DataFrame(rows)
        .sort_values(["decision_date", "scope", "group", "entity"], kind="stable")
        .reset_index(drop=True)
    )
    coverage = _coverage(frame, members)
    manifest = {
        **contract,
        "dataset_version": dataset_version,
        "input_digest": version_hash,
        "input_files": inputs,
        "universe_scope": "cache_discovered_assets_only_unless_explicit_universe_is_supplied",
        "survivorship_bias": (
            "possible: historical listings and delistings are not recoverable from the legacy cache alone"
        ),
        "provider_provenance": (
            sorted({str(item["provider_provenance"]) for item in inputs}) if inputs else []
        ),
        "feature_aggregation": (
            "group features are equal-weight means and require every member known at the decision date"
        ),
        "feature_columns": list(required_feature_columns),
        "label_columns_are_not_features": True,
    }
    return DatasetBuild(frame=frame, universe=members, manifest=manifest, coverage=coverage)


def causal_feature_digest(frame: pd.DataFrame, cutoff: object) -> str:
    """Hash only information available at or before ``cutoff``."""
    cutoff_date = pd.Timestamp(cutoff).date().isoformat()
    columns = [
        "decision_date",
        "information_cutoff",
        "scope",
        "entity",
        "group",
        "universe_member_count",
        "membership_provenance",
        "history_status",
        "feature_status",
        "feature_unavailable_fields",
        "decision_price",
        *FEATURE_COLUMNS,
    ]
    bounded = frame.loc[frame["decision_date"] <= cutoff_date, columns].copy()
    bounded = bounded.sort_values(["decision_date", "scope", "group", "entity"], kind="stable")
    payload = bounded.to_csv(index=False, lineterminator="\n", float_format="%.12g", na_rep="NA")
    return _sha256_bytes(payload.encode("utf-8"))


def write_dataset_artifact(build: DatasetBuild, output_root: str | Path) -> Path:
    """Write a content-addressed artifact without copying source price files."""
    directory = Path(output_root) / str(build.manifest["dataset_version"])
    directory.mkdir(parents=True, exist_ok=True)
    dataset_path = directory / "forecast_dataset.csv"
    build.frame.to_csv(
        dataset_path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    )
    dataset_hash = _file_sha256(dataset_path)
    (directory / "universe.json").write_text(
        json.dumps([asdict(member) for member in build.universe], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "coverage.json").write_text(
        json.dumps(build.coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        **build.manifest,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file": dataset_path.name,
        "dataset_sha256": dataset_hash,
        "rows": int(len(build.frame)),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory
