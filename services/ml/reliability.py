"""Read-only ML capabilities. No training, governance or synthetic fallback here.

Evaluation and HTTP adapters use the same daily-close calculations and JSON
artifacts. Dataset receipts are required; the generic mixed-provider price cache
does not qualify as a verified forecasting dataset.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import httpx

from api.schemas.ml_contract import (
    Availability, Horizon, ModelType, Provenance, UnifiedPrediction, ValidationStatus,
)

PROTOCOL_ID = "ml-risk-2026-09-30-v1"
FEATURES = ["rv7", "rv30", "rv90", "ewma", "abs_return"]
ROOT = Path(__file__).resolve().parents[2]


def code_version() -> str:
    paths = [Path(__file__), ROOT / "services/ml/risk_evaluation.py", ROOT / "api/schemas/ml_contract.py"]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if not re.fullmatch(r"[A-Z0-9.^=_-]{1,24}", symbol):
        raise ValueError("Invalid asset symbol")
    return symbol


def daily_features(close: pd.Series, market: str) -> pd.DataFrame:
    """Causal, trailing-only features, shared by research and inference."""
    annualization = 365 if market == "crypto" else 252
    returns = np.log(close / close.shift(1))
    frame = pd.DataFrame(index=close.index)
    for window in (7, 30, 90):
        frame[f"rv{window}"] = returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(annualization)
    frame["ewma"] = np.sqrt(returns.pow(2).ewm(alpha=.06, adjust=False, min_periods=30).mean() * annualization)
    frame["abs_return"] = returns.abs()
    return frame


def future_targets(close: pd.Series, market: str, horizon: int) -> pd.DataFrame:
    """Decision close excluded; target close included. Never substitute horizons."""
    if horizon not in (7, 30):
        raise ValueError("Only 7/30 calendar-day forecasts are supported")
    annualization = 365 if market == "crypto" else 252
    returns = np.log(close / close.shift(1)).to_numpy()
    rows = []
    for i, decision in enumerate(close.index):
        target = decision + pd.Timedelta(days=horizon)
        j = close.index.searchsorted(target)
        # Crypto requires every daily close; stocks use the next observed session
        # in retrospective evaluation. Live dates require a verified calendar.
        if j >= len(close) or (market == "crypto" and (j - i != horizon or close.index[j] != target)):
            rows.append((np.nan, pd.NaT))
            continue
        values = returns[i + 1:j + 1]
        variance = float(np.var(values, ddof=0) * annualization) if len(values) > 1 else np.nan
        rows.append((np.sqrt(variance), close.index[j]))
    return pd.DataFrame(rows, columns=["target", "target_end"], index=close.index)


def infer_estimator(model: dict, features: pd.DataFrame, horizon: int) -> np.ndarray:
    if model["method"] == "persistence":
        return features[f"rv{horizon}"].to_numpy()
    if model["method"] == "ewma":
        return features["ewma"].to_numpy()
    if model["method"] == "ridge":
        if model.get("features") != FEATURES:
            raise ValueError("Artifact feature schema is incompatible")
        transformed = (features[FEATURES].to_numpy() - np.array(model["mean"])) / np.array(model["scale"])
        return np.maximum(0, transformed @ np.array(model["coef"]) + model["intercept"])
    if model["method"] == "lstm":
        from services.ml.risk_evaluation import lstm_network, sequences
        import torch
        network = lstm_network()
        network.load_state_dict({key: torch.tensor(value, dtype=torch.float32) for key, value in model["weights"].items()})
        network.eval()
        x = (features[FEATURES].to_numpy()-np.array(model["mean"])) / np.array(model["scale"])
        with torch.no_grad():
            return np.maximum(0, network(torch.tensor(sequences(x), dtype=torch.float32)).numpy().ravel())
    raise ValueError("Unsupported or incompatible estimator")


class CapabilityService:
    def __init__(self, root: Path = ROOT, now=None, refresh_observations=None):
        self.root = Path(root)
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.refresh_observations = (os.getenv("ML_DATA_REFRESH", "0") == "1") if refresh_observations is None else refresh_observations
        # No cross-tenant result cache. Receipts and artifact hashes are verified
        # on each read, so a replaced dataset cannot reuse an old prediction.
        self.loaded: dict[str, str] = {}
        self.successful: dict[str, str] = {}
        self.errors: dict[str, str] = {}

    def registry(self) -> dict:
        path = self.root / "config/ml_capability_registry.json"
        if not path.exists():
            return {}
        registry = json.loads(path.read_text(encoding="utf-8"))
        return registry.get("capabilities", {})

    def history(self, market: str, asset: str) -> tuple[pd.Series, dict]:
        if market not in ("crypto", "stocks"):
            raise ValueError("Unsupported market")
        directory = self.root / "data/ml_verified" / market
        receipt = json.loads((directory / f"{valid_symbol(asset)}.json").read_text(encoding="utf-8"))
        path = (directory / receipt["file"]).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != receipt["sha256"]:
            raise ValueError("Dataset receipt hash mismatch or unsafe path")
        if not receipt.get("provider") or not receipt.get("dataset_id"):
            raise ValueError("Dataset provenance is incomplete")
        if market == "stocks" and not receipt.get("adjustment_policy"):
            raise ValueError("Stock split/dividend adjustment policy is missing")
        frame = pd.read_csv(path)
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True)).normalize()
        values = pd.to_numeric(frame["close"], errors="raise").to_numpy()
        if dates.has_duplicates or not dates.is_monotonic_increasing or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Daily close dataset is invalid")
        # Exclude an incomplete UTC day; stocks receipts carry completed dates.
        complete = dates < pd.Timestamp(self.now()).normalize()
        close = pd.Series(values[complete], index=dates[complete], name="close")
        if market == "crypto" and len(close) > 1 and not ((close.index[1:] - close.index[:-1]).days == 1).all():
            raise ValueError("Daily crypto history contains missing closes")
        if market == "stocks":
            if not receipt.get("exchange_calendar"):
                raise ValueError("Stock exchange session calendar is missing")
            import exchange_calendars as xcals
            expected = xcals.get_calendar(receipt["exchange_calendar"], start=close.index[0].date(), end=close.index[-1].date()).sessions
            expected = pd.DatetimeIndex(expected).tz_localize("UTC") if expected.tz is None else expected.tz_convert("UTC")
            if not close.index.equals(expected):
                raise ValueError("Stock history contains missing or unexpected exchange sessions")
        if market == "crypto" and self.refresh_observations:
            from services.ml.live_observations import extend_crypto
            close, receipt = extend_crypto(self.root, asset, close, receipt, self.now())
        if market == "stocks" and self.refresh_observations:
            from services.ml.live_observations import extend_stocks
            close, receipt = extend_stocks(self.root, asset, close, receipt, self.now())
        return close, receipt

    def artifact(self, market: str, asset: str, horizon: int) -> tuple[dict, Path]:
        path = self.root / "models/validated_risk" / f"{market}_{valid_symbol(asset)}_{horizon}d.json"
        model = json.loads(path.read_text(encoding="utf-8"))
        registry = json.loads((path.parent / "registry.json").read_text(encoding="utf-8"))
        publication = registry.get(path.name, {})
        if publication.get("sha256") != digest(path):
            raise ValueError("Artifact fingerprint mismatch or incompatible publication record")
        required = ("method", "dataset_id", "code_version", "training_start", "training_end", "validation")
        if any(not model.get(key) for key in required) or model.get("schema_version") != 1:
            raise ValueError("Artifact metadata is incomplete or incompatible")
        if model.get("asset") != asset or model.get("market") != market or model.get("horizon") != horizon:
            raise ValueError("Artifact target is incompatible")
        validation = model["validation"]
        if validation.get("state") != "retrospectively_validated" or validation.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("Artifact has no accepted confirmation evaluation")
        if model.get("features") != FEATURES or model.get("annualization") != (365 if market == "crypto" else 252):
            raise ValueError("Artifact features or annualization are incompatible")
        if model.get("code_version") != code_version():
            raise ValueError("Artifact was evaluated against a different calculation version")
        self.loaded[str(path)] = digest(path)
        return model, path

    async def result(self, asset: str, market: str, model_type: ModelType, horizon: Horizon | None = None) -> UnifiedPrediction:
        return await asyncio.to_thread(self._result, asset, market, model_type, horizon)

    def _result(self, asset: str, market: str, model_type: ModelType, horizon: Horizon | None = None) -> UnifiedPrediction:
        asset = valid_symbol(asset)
        diagnostic = model_type == ModelType.REGIME
        result = UnifiedPrediction(asset=asset, market=market, horizon=horizon,
            nature="diagnostic" if diagnostic else "forecast",
            target="economic_rule_regime" if diagnostic else "future_realized_volatility" if model_type == ModelType.VOLATILITY else model_type.value,
            unit="category" if diagnostic else f"annualized fraction ({365 if market == 'crypto' else 252} days/year)" if model_type == ModelType.VOLATILITY else None,
            provenance=Provenance(code_version=code_version()))
        if model_type == ModelType.VOLATILITY and horizon not in (Horizon.D7, Horizon.D30):
            result.reason = "Only validated 7/30 calendar-day volatility forecasts are supported"
            return result
        if model_type not in (ModelType.VOLATILITY, ModelType.REGIME):
            result.reason = "No validated forecast adapter is connected for this target"
            return result
        try:
            close, receipt = self.history(market, asset)
            if len(close) < 200:
                raise ValueError("At least 200 complete daily closes are required")
            result.data_as_of = close.index[-1].to_pydatetime()
            result.provenance = Provenance(provider=receipt["provider"], dataset_id=receipt["dataset_id"], code_version=code_version(), adjustment_policy=receipt.get("adjustment_policy"))
            age = (self.now() - result.data_as_of).total_seconds() / 86400
            result.quality.data_freshness = age * 24
            if model_type == ModelType.REGIME:
                drawdown = float(close.iloc[-1] / close.iloc[-200:].max() - 1)
                trend = float(close.iloc[-1] / close.iloc[-200:].mean() - 1)
                result.value = "Bear Market" if drawdown <= -.2 else "Correction" if drawdown <= -.1 or trend < 0 else "Expansion" if trend > .2 else "Bull Market"
                result.nature = "diagnostic"
                result.target = "economic_rule_regime"
                result.unit = "category"
                result.availability = Availability.PARTIAL if age > 4 else Availability.AVAILABLE
                result.reason = "Descriptive rules: 200-close drawdown <=-20% bear; <=-10% or below MA200 correction; >20% above MA200 expansion; otherwise bull. No future-direction probability." + (" Observations are stale." if age > 4 else "")
                result.provenance.method = "trailing_drawdown_ma200_rules_v1"
                result.validation = ValidationStatus(state="descriptive", reason="Economic thresholds are heuristic diagnostics")
                return result
            days = int(horizon.value[:-1])
            if age > (2 if market == "crypto" else 4):
                raise ValueError("Verified daily closes are stale; refresh their provider receipt before forecasting")
            model, path = self.artifact(market, asset, days)
            if (receipt.get("evaluation_dataset_id", receipt["dataset_id"]) != model["dataset_id"] or
                receipt.get("evaluation_sha256", receipt["sha256"]) != model["dataset_sha256"]):
                raise ValueError("Dataset identity does not match the validated artifact")
            target = close.index[-1] + pd.Timedelta(days=days)
            if market == "stocks":
                calendar = receipt.get("exchange_calendar")
                if not calendar:
                    raise ValueError("Stock target session calendar is not verified")
                import exchange_calendars as xcals
                target = xcals.get_calendar(calendar).date_to_session(target.tz_localize(None), direction="next")
            features = daily_features(close, market).dropna().tail(14)
            value = float(infer_estimator(model, features, days)[-1])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Inference returned an invalid value")
            result.value = value
            result.target = "future_realized_volatility"
            if target.tzinfo is None:
                target = target.tz_localize("UTC")
            result.target_date = target.to_pydatetime()
            result.unit = f"annualized fraction ({model['annualization']} days/year)"
            result.availability = Availability.AVAILABLE
            result.reason = "Retrospectively validated point estimate; future performance is not guaranteed"
            result.validation = ValidationStatus(**model["validation"])
            result.validation.evaluated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            result.provenance = Provenance(provider=receipt["provider"], dataset_id=receipt["dataset_id"], evaluation_dataset_id=model["dataset_id"], observation_sha256=receipt["sha256"], code_version=model["code_version"], artifact_sha256=digest(path), training_start=model["training_start"], training_end=model["training_end"], adjustment_policy=receipt.get("adjustment_policy"), method=model["method"])
            self.successful[str(path)] = digest(path)
            # Intervals are deliberately omitted unless the separate calibration
            # and untouched confirmation record certify their empirical coverage.
            interval = model.get("interval")
            if interval and .85 <= interval.get("confirmation_coverage", -1) <= .95 and interval.get("level") == .9:
                from api.schemas.ml_contract import UncertaintyMeasures
                width = interval["width"]
                result.uncertainty = UncertaintyMeasures(lower_bound=max(0, value-width), upper_bound=value+width, nominal_coverage=.9, confirmation_coverage=interval['confirmation_coverage'])
            else:
                result.reason += "; uncertainty interval is not validated"
        except Exception as exc:
            result.value = None
            result.availability = Availability.UNAVAILABLE
            result.reason = "Verified dataset or artifact is missing" if isinstance(exc, FileNotFoundError) else str(exc)
            self.errors[f"{market}/{asset}/{horizon}"] = result.reason
        return result

    def catalog(self) -> list[dict]:
        """Files present does not imply a loaded model or successful inference."""
        entries = []
        for base in ("models", "cache/ml_pipeline/models"):
            directory = self.root / base
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.name == "registry.json":
                    continue
                if path.suffix not in (".pth", ".pkl", ".keras") and not (path.suffix == ".json" and path.parent.name == "validated_risk"):
                    continue
                entries.append(dict(path=str(path.relative_to(self.root)), sha256=digest(path), present=True, loaded=self.loaded.get(str(path)) == digest(path), inference_succeeded=self.successful.get(str(path)) == digest(path), availability="Available" if self.successful.get(str(path)) == digest(path) else "Experimental", reason="File presence does not certify compatibility, validation or inference"))
        return entries

    def journal(self, user: str, source: str, results: list[UnifiedPrediction]) -> None:
        """Opt-in after deployment: record consultations only, never train or trade."""
        if os.getenv("ML_INFERENCE_JOURNAL", "0") != "1":
            return
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user):
            raise ValueError("Invalid authenticated user identity")
        directory = self.root / "data/users" / user / "ml"
        directory.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(directory / "inferences.sqlite") as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS inferences (identity TEXT PRIMARY KEY, consulted_at TEXT NOT NULL, result_json TEXT NOT NULL)")
            for result in results:
                if result.nature != "forecast" or result.value is None:
                    continue
                payload = dict(user_id=user, source=source, **result.model_dump(mode="json"))
                key = {name: payload[name] for name in ("user_id", "source", "asset", "market", "horizon", "data_as_of", "target_date", "provenance")}
                identity = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
                connection.execute("INSERT OR IGNORE INTO inferences VALUES (?,?,?)", (identity, self.now().isoformat(), json.dumps(payload, allow_nan=False)))

    def historical_correlation(self, market: str, assets: list[str]) -> UnifiedPrediction:
        result = UnifiedPrediction(asset="PORTFOLIO_UNIVERSE", market=market, nature="diagnostic", target="historical_correlation", unit="Pearson correlation [-1,1]")
        histories = {}
        receipts = {}
        for asset in assets:
            try:
                close, receipt = self.history(market, asset)
                histories[asset] = np.log(close / close.shift(1)).tail(90)
                receipts[asset] = receipt
            except (ValueError, FileNotFoundError, KeyError):
                continue
        if len(histories) < 2:
            result.reason = "At least two verified daily histories are required for a descriptive correlation"
            return result
        frame = pd.DataFrame(histories).dropna()
        if len(frame) < 30 or (frame.std(ddof=0) == 0).any():
            result.reason = "Insufficient common observations or constant returns; correlation is undefined"
            return result
        result.value = frame.corr().to_dict()
        result.data_as_of = frame.index[-1].to_pydatetime()
        age = (self.now()-result.data_as_of).total_seconds()/86400
        complete = len(histories) == len(assets) and age <= 4
        result.availability = Availability.AVAILABLE if complete else Availability.PARTIAL
        result.reason = f"Descriptive correlation on {len(frame)} common trailing daily returns; no forecast." + (" Some assets are missing or observations are stale." if not complete else "")
        result.validation = ValidationStatus(state="descriptive", reason="Historical Pearson correlation, not a forecasting validation")
        result.provenance = Provenance(provider="; ".join(sorted({r['provider'] for r in receipts.values()})), dataset_id="; ".join(sorted(r["dataset_id"] for r in receipts.values())), code_version=code_version(), method="trailing_90_observations_Pearson_complete_cases")
        return result

    async def external_sentiment(self) -> UnifiedPrediction:
        result = UnifiedPrediction(asset="CRYPTO_MARKET", market="crypto", nature="diagnostic", target="external_fear_greed", unit="index [0,100]")
        result.provenance = Provenance(provider="Alternative.me", code_version=code_version(), method="external_provider_index_not_ML")
        result.validation = ValidationStatus(state="descriptive", reason="External sentiment index; no predictive validation")
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                response = await client.get("https://api.alternative.me/fng/", params={"limit": 1})
                response.raise_for_status()
                observation = response.json()["data"][0]
            value = float(observation["value"])
            observed = datetime.fromtimestamp(int(observation["timestamp"]), tz=timezone.utc)
            if not 0 <= value <= 100 or observed > self.now():
                raise ValueError("External provider observation is invalid")
            result.value = value
            result.data_as_of = observed
            result.availability = Availability.AVAILABLE if self.now()-observed <= timedelta(days=2) else Availability.PARTIAL
            result.reason = "External Fear & Greed observation; not an ML prediction"
        except Exception:
            result.reason = "External Fear & Greed provider observation is unavailable"
        return result

    async def overview(self, user: str, source: str, assets: list[str], market: str) -> dict[str, Any]:
        results = []
        for asset in assets:
            results.append(await self.result(asset, market, ModelType.REGIME))
            for horizon in (Horizon.D7, Horizon.D30):
                results.append(await self.result(asset, market, ModelType.VOLATILITY, horizon))
        correlation = await asyncio.to_thread(self.historical_correlation, market, assets)
        results.append(correlation)
        sentiment = await self.external_sentiment() if market == "crypto" else None
        if sentiment:
            results.append(sentiment)
        self.journal(user, source, results)
        catalog = await asyncio.to_thread(self.catalog)
        forecasts = [r for r in results if r.nature == "forecast"]
        available_forecasts = sum(r.value is not None for r in forecasts)
        volatility_state = "Available" if forecasts and available_forecasts == len(forecasts) else "Partial" if available_forecasts else "Unavailable"
        capabilities = [
            dict(id="volatility", label="Volatility forecast", availability=volatility_state, reason=f"{available_forecasts}/{len(forecasts)} requested forecasts are available. Only confirmed 7/30-day estimates are published; legacy intraday forecasts are unavailable"),
            dict(id="regime", label="Economic regime rules", availability="Partial" if any(r.value is not None for r in results if r.target == "economic_rule_regime") else "Unavailable", reason="Descriptive trailing-price rules, separate from HMM state probabilities"),
            dict(id="hmm", label="HMM latent states", availability="Experimental", reason="States A–D require documented economic mapping; state probabilities are not probabilities of a future rise"),
            dict(id="correlation", label="Historical correlations", availability=correlation.availability.value, reason=correlation.reason+" Transformer forecasts remain experimental."),
            dict(id="cycle", label="Market cycle", availability="Experimental", reason="Heuristic cycle diagnostic, not a validated forecast"),
            dict(id="sentiment", label="Fear & Greed", availability=sentiment.availability.value if sentiment else "Unavailable", reason=sentiment.reason if sentiment else "External crypto indicator is not applicable to stocks"),
            dict(id="direction", label="Directional prediction", availability="Rejected", reason="Existing causal research failed the economic gates; experiments remain closed"),
            dict(id="alerts", label="ML alert predictor", availability="Rejected", reason="Disabled; reactivation is outside this implementation"),
        ]
        registry = self.registry()
        for capability in capabilities:
            entry = registry.get(capability["id"])
            if entry:
                capability.update({key: entry[key] for key in ("availability", "reason", "validation_state") if key in entry})
        return dict(user_id=user, source=source, market=market, scope="explicit_market_universe", code_version=code_version(), observed_at=self.now().isoformat(), protocol_id=PROTOCOL_ID, results=[r.model_dump(mode="json") for r in results], capabilities=capabilities, artifacts=catalog, counts=dict(files_present=len(catalog), models_loaded=sum(a["loaded"] for a in catalog), successful_inferences=sum(r.value is not None and r.nature == "forecast" for r in results)), governance_integration=False)


capability_service = CapabilityService()
