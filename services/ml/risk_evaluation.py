"""Frozen chronological volatility comparison; explicit actions only."""
from __future__ import annotations

import json
import os
from uuid import uuid4
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from sklearn.linear_model import Ridge

from services.ml.reliability import FEATURES, PROTOCOL_ID, CapabilityService, daily_features, future_targets, infer_estimator, digest, code_version


def sequences(x: np.ndarray) -> np.ndarray:
    padded = np.pad(x, ((13, 0), (0, 0)), mode="edge")
    return np.stack([padded[i:i+14] for i in range(len(x))])


def lstm_network():
    import torch
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(len(FEATURES), 8, batch_first=True)
            self.output = torch.nn.Linear(8, 1)
        def forward(self, x):
            values, _ = self.lstm(x)
            return torch.nn.functional.softplus(self.output(values[:, -1]))
    return Network()


def predict_rows(model, frame, rows, horizon):
    # Include past context at fold boundaries; every sequence ends at its own
    # decision row. Later rows never enter an earlier prediction.
    prediction = pd.Series(infer_estimator(model, frame, horizon), index=frame.index)
    return prediction.loc[rows.index].to_numpy()


def fit_estimator(method: str, train: pd.DataFrame) -> dict:
    if method in ("persistence", "ewma"):
        return {"method": method, "features": FEATURES}
    x = train[FEATURES]
    mean = x.mean().to_numpy()
    scale = x.std(ddof=0).replace(0, 1).to_numpy()
    if method == "lstm":
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(1729)
        torch.use_deterministic_algorithms(True)
        network = lstm_network()
        inputs = torch.tensor(sequences((x.to_numpy()-mean)/scale), dtype=torch.float32)
        target = torch.tensor(train.target.to_numpy()[:, None], dtype=torch.float32)
        optimizer = torch.optim.Adam(network.parameters(), lr=.01)
        for _ in range(12):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(network(inputs), target)
            loss.backward()
            optimizer.step()
        return dict(method="lstm", features=FEATURES, mean=mean.tolist(), scale=scale.tolist(), weights={k:v.detach().tolist() for k,v in network.state_dict().items()}, architecture="lstm_5_8_sequence14", seed=1729, epochs=12)
    fitted = Ridge(alpha=1).fit((x.to_numpy()-mean)/scale, train["target"])
    return dict(method="ridge", features=FEATURES, mean=mean.tolist(), scale=scale.tolist(), coef=fitted.coef_.tolist(), intercept=float(fitted.intercept_))


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    variance = np.maximum(np.square(predicted), 1e-10)
    truth = np.maximum(np.square(actual), 1e-10)
    # QLIKE in the nonnegative, scale-free ratio form.
    ratio = truth / variance
    return dict(qlike=float(np.mean(ratio-np.log(ratio)-1)), mae=float(np.mean(np.abs(actual-predicted))), rows=len(actual))


def partitions(frame: pd.DataFrame, calibration_start, test_start, test_end):
    train = frame[(frame.index < calibration_start) & (frame.target_end < calibration_start)]
    calibration = frame[(frame.index >= calibration_start) & (frame.index < test_start) & (frame.target_end < test_start)]
    test = frame[(frame.index >= test_start) & (frame.index <= test_end) & (frame.target_end <= test_end)]
    if train.empty or calibration.empty or test.empty:
        raise ValueError("Insufficient rows after target-end purge")
    if (train.index[-1]-train.index[0]).days < 730:
        raise ValueError("Training span is shorter than two years after purge")
    return train, calibration, test


def evaluate(service: CapabilityService, market: str, asset: str, horizon: int, *, publish=False) -> dict:
    close, receipt = service.history(market, asset)
    frame = daily_features(close, market).join(future_targets(close, market, horizon)).dropna()
    report = dict(asset=asset, market=market, horizon=horizon, protocol_id=PROTOCOL_ID, dataset_id=receipt["dataset_id"], dataset_sha256=receipt["sha256"], code_version=code_version(), data_end=close.index[-1].isoformat(), annualization=365 if market == "crypto" else 252)
    # Final 12 calendar months are untouched until all development folds and
    # the candidate choice are complete. No confirmation-based switching.
    confirmation_start = close.index[-1] - pd.Timedelta(days=364)
    # A stock boundary can fall on a weekend/holiday. Leave enough slack to
    # guarantee a full 730-day observed training span after target-end purge.
    calibration_start = frame.index[0] + pd.Timedelta(days=740 + horizon)
    folds = []
    methods = ["persistence", "ewma", "ridge"]
    lstm_reason = None
    try:
        import torch
        methods.append("lstm")
    except ImportError:
        lstm_reason = "PyTorch runtime is unavailable for the corrected causal candidate"
    while calibration_start + pd.Timedelta(days=366) < confirmation_start:
        test_start = calibration_start + pd.Timedelta(days=183)
        test_end = test_start + pd.Timedelta(days=182)
        train, calibration, test = partitions(frame, calibration_start, test_start, test_end)
        scores = {}
        for method in methods:
            model = fit_estimator(method, train)
            prediction = predict_rows(model, frame, test, horizon)
            scores[method] = metrics(test.target.to_numpy(), prediction)
        folds.append(dict(train_start=train.index[0].isoformat(), train_end=train.index[-1].isoformat(), calibration_start=calibration.index[0].isoformat(), calibration_end=calibration.index[-1].isoformat(), test_start=test.index[0].isoformat(), test_end=test.index[-1].isoformat(), scores=scores))
        calibration_start += pd.Timedelta(days=183)
    if len(folds) < 3:
        return {**report, "state": "not_evaluable", "reason": "Fewer than three development windows plus reserved confirmation", "folds": folds, "published": False}
    means = {method: float(np.mean([f["scores"][method]["qlike"] for f in folds])) for method in methods}
    reference = min(("persistence", "ewma"), key=lambda m: means[m])
    candidate = min([m for m in methods if m not in ("persistence", "ewma")], key=lambda m: means[m])
    wins = sum(f["scores"][candidate]["qlike"] < f["scores"][reference]["qlike"] for f in folds)
    improvement = 1-means[candidate]/max(means[reference], 1e-10)
    mae_control = np.mean([f["scores"][candidate]["mae"] for f in folds]) <= np.mean([f["scores"][reference]["mae"] for f in folds])
    learned_pass = improvement >= .05 and wins/len(folds) >= 2/3 and mae_control
    chosen = candidate if learned_pass else reference
    # Fit once on pre-calibration rows, then evaluate the chosen frozen estimator.
    train, calibration, confirmation = partitions(frame, confirmation_start-pd.Timedelta(days=183), confirmation_start, close.index[-1])
    models = {method: fit_estimator(method, train) for method in methods}
    confirmation_scores = {method: metrics(confirmation.target.to_numpy(), predict_rows(models[method], frame, confirmation, horizon)) for method in methods}
    confirm_improvement = 1-confirmation_scores[candidate]["qlike"]/max(confirmation_scores[reference]["qlike"], 1e-10)
    confirmed = chosen in ("persistence", "ewma") or (confirm_improvement >= .05 and confirmation_scores[candidate]["mae"] <= confirmation_scores[reference]["mae"])
    # A failed learned confirmation cannot choose another learned candidate.
    if not confirmed:
        chosen = reference
    fitted = models[chosen]
    residuals = np.abs(calibration.target.to_numpy()-predict_rows(fitted, frame, calibration, horizon))
    width = float(np.quantile(residuals, .9, method="higher"))
    prediction = predict_rows(fitted, frame, confirmation, horizon)
    coverage = float(np.mean(np.abs(confirmation.target.to_numpy()-prediction) <= width))
    interval = dict(level=.9, width=width, confirmation_coverage=coverage, calibration_rows=len(calibration), confirmation_rows=len(confirmation)) if .85 <= coverage <= .95 else None
    conclusion = dict(state="retrospectively_validated", reason=f"Frozen walk-forward protocol; selected {chosen}. Learned model gates: QLIKE >=5%, folds >=2/3, MAE control and reserved confirmation.", protocol_id=PROTOCOL_ID, metrics=dict(development_qlike=means, learned_candidate=candidate, learned_improvement=improvement, learned_wins=wins, windows=len(folds), confirmation=confirmation_scores, learned_confirmation_improvement=confirm_improvement, interval_coverage=coverage))
    artifact = {**fitted, **report, "schema_version": 1, "features": FEATURES, "training_start": train.index[0].isoformat(), "training_end": train.index[-1].isoformat(), "validation": conclusion, "interval": interval}
    report.update(state="retrospectively_validated", selected=chosen, reference=reference, folds=folds, validation=conclusion, interval=interval, lstm=dict(state="evaluated" if "lstm" in methods else "not_evaluable", reason=lstm_reason or "Corrected trailing sequences, train-only normalization, fixed seed 1729, 12 epochs; no legacy checkpoint reuse"), published=False)
    if publish:
        directory = service.root / "models/validated_risk"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{market}_{asset}_{horizon}d.json"
        temporary = path.with_suffix(f".{uuid4().hex}.json.tmp")
        temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
        # Verify serialized fields and inference before atomically making visible.
        serialized = json.loads(temporary.read_text(encoding="utf-8"))
        verified = predict_rows(serialized, frame, confirmation, horizon)
        if not np.array_equal(verified, prediction):
            raise ValueError("Artifact inference is not reproducible")
        with FileLock(str(directory / "publication.lock")):
            registry_path = directory / "registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
            registry[path.name] = dict(sha256=digest(temporary), dataset_id=receipt["dataset_id"], code_version=artifact["code_version"], validation_state="retrospectively_validated", governance_eligible=False)
            registry_tmp = registry_path.with_suffix(".json.tmp")
            registry_tmp.write_text(json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
            os.replace(registry_tmp, registry_path)
        service.artifact(market, asset, horizon)
        report.update(published=True, artifact=str(path.relative_to(service.root)), artifact_sha256=digest(path))
    return report
