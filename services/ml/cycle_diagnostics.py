"""Descriptive Bitcoin cycle statistics using verified closes only."""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path

HALVINGS = ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")


def describe_cycles(close, receipt):
    cycles = []
    for number, halving in enumerate(HALVINGS, 1):
        start = pd.Timestamp(halving, tz="UTC")
        end = (
            pd.Timestamp(HALVINGS[number], tz="UTC")
            if number < len(HALVINGS)
            else close.index[-1] + pd.Timedelta(days=1)
        )
        if start > close.index[-1]:
            continue
        values = close[(close.index >= start) & (close.index < end)]
        if values.empty:
            cycles.append({"cycle": number, "halving": halving, "anchor_available": False,
                           "start": None, "end": None, "peak_date": None,
                           "days_to_observed_peak": None, "observed_peak": None,
                           "maximum_drawdown": None, "return_since_halving": None,
                           "complete_cycle": False, "coverage": "Unavailable", "points": []})
            continue
        complete_anchor = values.index[0] == start
        running_peak = values.cummax()
        drawdowns = values / running_peak - 1
        peak_date = values.idxmax()
        cycles.append(
            {
                "cycle": number,
                "halving": halving,
                "anchor_available": bool(complete_anchor),
                "start": values.index[0].date().isoformat(),
                "end": values.index[-1].date().isoformat(),
                "peak_date": peak_date.date().isoformat(),
                "days_to_observed_peak": int((peak_date - start).days),
                "observed_peak": float(values.max()),
                "halving_close": float(values.iloc[0]) if complete_anchor else None,
                "latest_close": float(values.iloc[-1]),
                "drawdown_from_peak": float(values.iloc[-1]/values.max()-1),
                "lowest_after_peak": float(values.loc[peak_date:].min()),
                "lowest_after_peak_date": values.loc[peak_date:].idxmin().date().isoformat(),
                "days_since_observed_peak": int((values.index[-1]-peak_date).days),
                "provider": receipt['provider'],
                "dataset_id": receipt['dataset_id'],
                "quote_unit": receipt.get('quote_unit','USDT'),
                "maximum_drawdown": float(drawdowns.min()),
                "return_since_halving": float(values.iloc[-1] / values.iloc[0] - 1)
                if complete_anchor
                else None,
                "complete_cycle": bool(complete_anchor and number < len(HALVINGS) and close.index[-1] >= end),
                "coverage": "Complete" if complete_anchor and number < len(HALVINGS) and close.index[-1] >= end else "Ongoing" if number == len(HALVINGS) else "Partial",
                "points": [
                    {
                        "day": int((date - start).days),
                        "normalized": float(value / values.iloc[0]) if complete_anchor else None,
                        "drawdown": float(drawdowns.loc[date]),
                    }
                    for date, value in values.items()
                ],
            }
        )
    as_of = close.index[-1]
    return {
        "availability": "Available" if cycles else "Unavailable",
        "nature": "diagnostic",
        "reason": "Observed daily closes; retrospective cycle peaks are not real-time forecasts. Earlier cycles may lack a halving close.",
        "data_as_of": as_of.date().isoformat(),
        "provider": receipt["provider"],
        "dataset_id": receipt["dataset_id"],
        "days_since_halving": int((as_of - pd.Timestamp(HALVINGS[-1], tz="UTC")).days),
        "cycles": cycles,
    }


def historical_cycle_comparison(root, current_close, current_receipt):
    """Use one price basis per cycle. Never splice different vendors within a cycle."""
    current = describe_cycles(current_close,current_receipt)
    directory = Path(root)/'data/ml_verified/cycles'
    metadata = directory/'BTC.json'
    if not metadata.exists():
        return current
    receipt = json.loads(metadata.read_text(encoding='utf-8'))
    path = (directory/receipt['file']).resolve()
    if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']:
        raise ValueError('Historical cycle reference receipt is invalid')
    frame = pd.read_csv(path)
    reference = pd.Series(frame.close.to_numpy(),index=pd.DatetimeIndex(pd.to_datetime(frame.date,utc=True)))
    if reference.index.has_duplicates or not reference.index.is_monotonic_increasing or not np.isfinite(reference).all() or (reference<=0).any() or not ((reference.index[1:]-reference.index[:-1]).days==1).all():
        raise ValueError('Historical cycle reference series is invalid')
    receipt['quote_unit']='USD reference rate'
    older=describe_cycles(reference,receipt)
    current['cycles']=[c for c in older['cycles'] if c['cycle']<4]+[c for c in current['cycles'] if c['cycle']==4]
    current['provider']='Coin Metrics Community (completed cycles); Binance Spot (current cycle)'
    current['dataset_id']=receipt['dataset_id']+'; '+current_receipt['dataset_id']
    current['reason']='Each cycle uses one observed price series: completed cycles use Coin Metrics USD reference rates; cycle 4 uses Binance USDT closes. No mid-cycle vendor splice. Historical peaks and lows are retrospective; current extrema are only observed so far. No future peak or bottom forecast.'
    current['attribution']=receipt.get('attribution')
    return current
