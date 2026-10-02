"""Explicit CLI evaluation. No credentials, recurrent jobs or production writes.

Run from the isolated project root. Research imports verify the acquisition hashes.
Stocks acquisition requires explicit --stocks and records adjustment/calendar policy.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from services.ml.reliability import CapabilityService, digest
from services.ml.risk_evaluation import evaluate


def save_history(root, market, asset, frame, provenance):
    directory = root / "data/ml_verified" / market
    directory.mkdir(parents=True, exist_ok=True)
    frame = frame[["date", "close"]].sort_values("date").drop_duplicates("date")
    path = directory / f"{asset}.csv"
    frame.to_csv(path, index=False)
    sha = digest(path)
    receipt = {**provenance, "file": path.name, "sha256": sha, "dataset_id": f"verified-daily-{market}-{asset}-{sha[:16]}"}
    (directory / f"{asset}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def import_research(service, acquisition):
    manifest_path = acquisition / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    imported = []
    for row in manifest["inputs"]:
        asset = row["symbol"]
        if asset not in ("BTC", "ETH", "SOL", "ADA", "XRP", "LINK", "LTC", "BCH", "BNB", "DOT"):
            continue
        path = acquisition / row["ohlcv_file"]
        if digest(path) != row["ohlcv_file_sha256"]:
            raise ValueError(f"Research acquisition hash mismatch: {asset}")
        frame = pd.read_csv(path)
        receipt = save_history(service.root, "crypto", asset, frame, dict(provider=row["provider_provenance"], research_artifact_id=manifest["artifact_id"], acquisition_manifest_sha256=digest(manifest_path), original_ohlcv_sha256=row["ohlcv_file_sha256"], adjustment_policy="Unadjusted Binance Spot USDT daily closes"))
        imported.append(receipt)
    return imported


def acquire_stocks(service, symbols):
    import yfinance as yf
    receipts = []
    for symbol in symbols:
        symbol = symbol.upper()
        data = yf.download(symbol, start="2006-01-01", end=pd.Timestamp.utcnow().strftime("%Y-%m-%d"), auto_adjust=True, actions=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if data.empty:
            receipts.append(dict(asset=symbol,state="not_evaluable",reason="Provider returned no verifiable daily history"))
            continue
        frame = pd.DataFrame(dict(date=data.index.strftime("%Y-%m-%d"), close=data["Close"].to_numpy()))
        receipts.append(save_history(service.root, "stocks", symbol, frame, dict(provider="Yahoo Finance via yfinance", yfinance_version=yf.__version__, adjustment_policy="auto_adjust=True: split/dividend-adjusted daily closes; current vendor revisions, not a point-in-time corporate-action archive", exchange_calendar=next((calendar for suffix,calendar in {".SW":"XSWX",".AS":"XAMS",".DE":"XFRA",".L":"XLON",".MI":"XMIL"}.items() if symbol.endswith(suffix)), "XNYS"), fetched_at=pd.Timestamp.utcnow().isoformat())))
    return receipts


def refresh_crypto(service, symbols):
    import httpx
    receipts = []
    for asset in symbols:
        directory = service.root / "data/ml_verified/crypto"
        receipt = json.loads((directory / f"{asset}.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(directory / receipt["file"])
        start = pd.Timestamp(frame.date.iloc[-1], tz="UTC")+pd.Timedelta(days=1)
        end = pd.Timestamp.utcnow().normalize()
        if start >= end:
            continue
        response = httpx.get("https://data-api.binance.vision/api/v3/klines", params=dict(symbol=asset+"USDT", interval="1d", startTime=int(start.timestamp()*1000), endTime=int(end.timestamp()*1000)-1, limit=1000), timeout=20)
        response.raise_for_status()
        additions = pd.DataFrame([dict(date=pd.Timestamp(row[0], unit="ms", tz="UTC").strftime("%Y-%m-%d"), close=float(row[4])) for row in response.json() if row[6] < int(end.timestamp()*1000)])
        if additions.empty:
            raise ValueError(f"No complete refresh observations for {asset}")
        receipts.append(save_history(service.root, "crypto", asset, pd.concat([frame, additions], ignore_index=True), {**receipt, "fetched_at": pd.Timestamp.utcnow().isoformat(), "refresh_raw_response_sha256": hashlib.sha256(response.content).hexdigest()}))
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-acquisition", type=Path)
    parser.add_argument("--stocks", nargs="*", default=[])
    parser.add_argument("--market", choices=["crypto", "stocks"], default="crypto")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--report-name", default=None, help="Distinct aggregate report name")
    parser.add_argument("--publish", action="store_true", help="Publish validated artifacts in this isolated checkout only")
    parser.add_argument("--refresh-crypto", action="store_true", help="Explicitly append complete daily Binance Spot closes before evaluation")
    args = parser.parse_args()
    service = CapabilityService(refresh_observations=False)
    receipts = []
    if args.research_acquisition:
        receipts.extend(import_research(service, args.research_acquisition))
    if args.stocks:
        receipts.extend(acquire_stocks(service, args.stocks))
    if args.refresh_crypto:
        receipts.extend(refresh_crypto(service, args.assets))
    directory = service.root / "outputs/ml-reliability"
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for asset in args.assets:
        for horizon in (7, 30):
            try:
                report = evaluate(service, args.market, asset.upper(), horizon, publish=args.publish)
            except Exception as exc:
                report = dict(asset=asset, market=args.market, horizon=horizon, state="not_evaluable", reason=str(exc), published=False)
            reports.append(report)
            report_dir = directory / "evaluations" / args.market
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"{asset.upper()}-{horizon}d.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({k: report.get(k) for k in ("asset", "horizon", "state", "selected", "reason", "published")}), flush=True)
            (directory / ((args.report_name or f"{args.market}-evaluation")+".json")).write_text(json.dumps(dict(receipts=receipts, reports=reports), indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
