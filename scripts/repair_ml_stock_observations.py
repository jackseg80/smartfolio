"""Explicit repair of recorded provider rows; no filling of missing sessions."""

import argparse
import json
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.ml.reliability import digest, CapabilityService
from scripts.evaluate_ml_risk import save_history


def repair_frame(frame, calendar, today):
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True)).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Provider date order is invalid")
    if len(frame) == 0:
        raise ValueError("Provider returned no rows")
    sessions = xc.get_calendar(calendar, start=dates[0].date(), end=dates[-1].date()).sessions
    sessions = sessions.tz_localize("UTC") if sessions.tz is None else sessions.tz_convert("UTC")
    complete = (dates < pd.Timestamp(today).normalize()) & dates.isin(sessions)
    excluded = dates[~complete].strftime("%Y-%m-%d").tolist()
    clean = frame.loc[complete].copy()
    values = pd.to_numeric(clean.close, errors="coerce").to_numpy()
    good = np.isfinite(values) & (values > 0)
    valid = np.flatnonzero(good)
    if not len(valid):
        raise ValueError("No finite positive closes")
    last = valid[-1]
    excluded += clean.date.iloc[last + 1 :].tolist()
    # Only incomplete trailing provider rows are dropped. Interior gaps stay visible.
    clean = clean.iloc[: last + 1]
    return clean, {
        "excluded_dates": excluded,
        "policy": "Exclude non-session/incomplete dates and trailing invalid provider rows; never fill an interior missing close",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", required=True)
    args = parser.parse_args()
    service = CapabilityService()
    directory = service.root / "data/ml_verified/stocks"
    for asset in args.assets:
        receipt = json.loads((directory / (asset + ".json")).read_text())
        path = directory / receipt["file"]
        if digest(path) != receipt["sha256"]:
            raise ValueError("Acquisition hash mismatch")
        raw = directory / "raw" / receipt["sha256"]
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, raw / path.name)
        shutil.copy2(directory / (asset + ".json"), raw / (asset + ".json"))
        frame, audit = repair_frame(
            pd.read_csv(path), receipt["exchange_calendar"], pd.Timestamp.utcnow()
        )
        receipt = save_history(
            service.root,
            "stocks",
            asset,
            frame,
            {
                **receipt,
                "original_provider_sha256": receipt["sha256"],
                "raw_archive": str((raw / path.name).relative_to(directory)),
                "observation_filter": audit,
            },
        )
        try:
            close, _ = service.history("stocks", asset)
            status = "Verified"
            reason = None
        except ValueError as error:
            status = "Unavailable"
            reason = str(error)
        print(
            json.dumps(
                {
                    "asset": asset,
                    "status": status,
                    "reason": reason,
                    "excluded_rows": len(audit["excluded_dates"]),
                    "data_as_of": frame.date.iloc[-1],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
