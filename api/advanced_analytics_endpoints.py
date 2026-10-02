"""Advanced diagnostics for the authenticated selected source, without simulated fallback."""
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.deps import get_required_user
from services.ml.portfolio_context import read_context
from services.price_history import get_cached_history
from services.portfolio_metrics import portfolio_metrics_service

router = APIRouter(prefix='/analytics/advanced', tags=['advanced-analytics'])

class AdvancedMetrics(BaseModel):
    total_return_pct: float
    annualized_return_pct: float
    volatility_pct: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown_duration_days: float
    drawdown_periods: list[dict]
    calmar_ratio: float | None
    sortino_ratio: float | None
    omega_ratio: float | None
    skewness: float | None
    kurtosis: float | None
    var_95: float
    cvar_95: float
    best_month_pct: float
    worst_month_pct: float
    positive_months_pct: float
    win_loss_ratio: float | None
    provenance: dict

class TimeSeriesData(BaseModel):
    dates: list[str]
    portfolio_values: list[float]
    returns: list[float]
    cumulative_returns: list[float]
    drawdowns: list[float]
    rolling_sharpe: list[float | None]
    rolling_volatility: list[float | None]
    provenance: dict

async def _performance(days, user, source):
    try:
        context = await read_context(user, source, 'crypto')
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(503, detail='Selected portfolio source is unavailable') from exc
    balances = context['items']
    if not balances:
        raise HTTPException(404, detail='No holdings in the selected source')
    prices = {}
    minimum=max(32,min(days+1,182))
    excluded_stale=0
    today=pd.Timestamp.now(tz='UTC').normalize()
    for balance in balances:
        symbol = str(balance.get('symbol') or '').upper()
        if not symbol or float(balance.get('value_usd') or 0) <= 0:
            continue
        observations = get_cached_history(symbol, days=days+3)
        if observations and len(observations) >= minimum:
            series = pd.Series([p[1] for p in observations], index=pd.to_datetime([p[0] for p in observations], unit='s', utc=True))
            series = series.groupby(series.index.normalize()).last()
            # Do not fabricate missing closes or include today's incomplete close.
            series = series[series.index < today]
            if series.empty or series.index[-1] < today-pd.Timedelta(days=2):
                excluded_stale+=1
                continue
            gaps=np.flatnonzero((series.index[1:]-series.index[:-1]).days!=1)
            if len(gaps):
                series=series.iloc[gaps[-1]+1:]
            if len(series)>=minimum and (series > 0).all() and np.isfinite(series).all():
                prices[symbol] = series
    if not prices:
        raise HTTPException(503, detail='Historical observations are unavailable for the selected holdings')
    frame = pd.DataFrame(prices).dropna().tail(days+1)
    if len(frame) < 30 or not ((frame.index[1:]-frame.index[:-1]).days==1).all():
        raise HTTPException(503, detail='Insufficient consecutive daily observations; no simulated fallback')
    returns = portfolio_metrics_service._calculate_weighted_portfolio_returns(frame, balances).dropna()
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise HTTPException(503, detail='Invalid historical portfolio returns')
    total = sum(float(b.get('value_usd') or 0) for b in balances)
    covered = sum(float(b.get('value_usd') or 0) for b in balances if str(b.get('symbol') or '').upper() in prices)
    provenance = {'source':source, 'source_used':context['source_used'], 'data_as_of':returns.index[-1].isoformat(),
        'coverage_by_current_value':covered/total if total > 0 else None, 'annualization':365,
        'data_start':frame.index[0].isoformat(),'observed_returns':len(returns),'requested_days':days,
        'minimum_asset_closes':minimum,'excluded_stale_assets':excluded_stale,
        'availability':'Partial' if covered < total else 'Available',
        'method':'current_holdings_weighted_historical_returns', 'value_unit':'normalized index, initial value 100',
        'reason':'Retrospective reconstruction using current holdings and cached daily observations, not actual account NAV or a strategy backtest. Missing, stale or insufficient histories are excluded; covered weights are normalized. Only the latest consecutive common window is used, without filling gaps. Historical cache provider provenance is not certified for forecasting. No future forecast.'}
    return returns, provenance

def _equity(returns):
    values=(1+returns).cumprod()*100
    return pd.concat([pd.Series([100.],index=[returns.index[0]-pd.Timedelta(days=1)]),values])

def _drawdowns(values, minimum=1):
    drawdown = values/values.cummax()-1
    periods = []
    start = None
    peak_date = values.index[0]
    for date, dd in drawdown.items():
        if dd == 0:
            if start is not None:
                duration = (date-start).days
                if duration >= minimum:
                    period = values.loc[start:date]
                    periods.append({'start_date':start.isoformat(), 'end_date':date.isoformat(), 'peak_value':float(values.loc[start]),
                        'trough_value':float(period.min()), 'drawdown_pct':float(drawdown.loc[start:date].min()*100),
                        'duration_days':duration, 'recovery_days':(date-period.idxmin()).days, 'is_recovered':True})
                start = None
            peak_date = date
        elif start is None:
            start = peak_date
    if start is not None and (values.index[-1]-start).days >= minimum:
        period = values.loc[start:]
        periods.append({'start_date':start.isoformat(), 'end_date':None, 'peak_value':float(values.loc[start]),
            'trough_value':float(period.min()), 'drawdown_pct':float(drawdown.loc[start:].min()*100),
            'duration_days':(values.index[-1]-start).days, 'recovery_days':None, 'is_recovered':False})
    return drawdown, periods

def _calculate(returns, provenance):
    values = _equity(returns)
    dd, periods = _drawdowns(values)
    total = values.iloc[-1]/100-1
    span = max(1, (returns.index[-1]-returns.index[0]).days+1)
    annual = (1+total)**(365/span)-1
    vol = returns.std(ddof=0)*np.sqrt(365)
    downside = np.sqrt(np.minimum(returns,0).pow(2).mean()*365)
    monthly = (1+returns).resample('ME').prod()-1
    losses = abs(returns[returns<0].sum())
    gains = returns[returns>0].sum()
    quantile = float(returns.quantile(.05))
    tail = returns[returns <= quantile]
    return AdvancedMetrics(total_return_pct=total*100, annualized_return_pct=annual*100,
        volatility_pct=vol*100, sharpe_ratio=float((returns.mean()*365-.02)/vol) if vol>0 else None,
        max_drawdown_pct=float(dd.min()*100), avg_drawdown_pct=float(dd[dd<0].mean()*100) if (dd<0).any() else 0,
        max_drawdown_duration_days=max((p['duration_days'] for p in periods),default=0),
        avg_drawdown_duration_days=float(np.mean([p['duration_days'] for p in periods])) if periods else 0,
        drawdown_periods=periods, calmar_ratio=float(annual/abs(dd.min())) if dd.min()<0 else None,
        sortino_ratio=float((returns.mean()*365-.02)/downside) if downside>0 else None,
        omega_ratio=float(gains/losses) if losses>0 else None, skewness=float(returns.skew()) if returns.std()>0 else None,
        kurtosis=float(returns.kurtosis()+3) if returns.std()>0 else None, var_95=quantile*100,
        cvar_95=float(tail.mean()*100), best_month_pct=float(monthly.max()*100), worst_month_pct=float(monthly.min()*100),
        positive_months_pct=float((monthly>0).mean()*100), win_loss_ratio=float(gains/losses) if losses>0 else None, provenance=provenance)

@router.get('/metrics', response_model=AdvancedMetrics)
async def get_advanced_metrics(user: str = Depends(get_required_user), days: int = Query(365,ge=30,le=3650), benchmark: Optional[str] = None, source: str = Query('cointracking',min_length=1,max_length=100)):
    if benchmark:
        raise HTTPException(422,detail='Benchmark comparison is not implemented; no substitute is shown')
    returns, provenance = await _performance(days,user,source)
    return _calculate(returns,provenance)

@router.get('/timeseries', response_model=TimeSeriesData)
async def get_timeseries_data(user: str = Depends(get_required_user), days: int = Query(365,ge=30,le=3650), granularity: str = Query('daily',pattern='^(daily|weekly|monthly)$'), source: str = Query('cointracking',min_length=1,max_length=100)):
    returns, provenance = await _performance(days,user,source)
    if granularity != 'daily':
        returns = (1+returns).resample('W' if granularity=='weekly' else 'ME').prod()-1
    values = _equity(returns)
    dd,_ = _drawdowns(values)
    factor = {'daily':365,'weekly':52,'monthly':12}[granularity]
    rolling_vol = returns.rolling(30).std(ddof=0)*np.sqrt(factor)
    rolling_sharpe = (returns.rolling(30).mean()*factor-.02)/rolling_vol.replace(0,np.nan)
    def clean(series):
        return [float(x) if pd.notna(x) and np.isfinite(x) else None for x in series]
    return TimeSeriesData(dates=values.index.strftime('%Y-%m-%d').tolist(),portfolio_values=values.tolist(),returns=[0.]+(returns*100).tolist(),
        cumulative_returns=((values/100-1)*100).tolist(),drawdowns=(dd*100).tolist(),rolling_sharpe=[None]+clean(rolling_sharpe),rolling_volatility=[None]+clean(rolling_vol*100),provenance=provenance)

@router.get('/drawdown-analysis')
async def analyze_drawdowns(user: str = Depends(get_required_user),days: int = Query(365,ge=30,le=3650),min_duration: int = Query(5,ge=1),source: str = Query('cointracking')):
    returns,provenance = await _performance(days,user,source)
    _,periods = _drawdowns(_equity(returns),min_duration)
    return {'summary':{'total_drawdown_periods':len(periods),'avg_drawdown_pct':float(np.mean([p['drawdown_pct'] for p in periods])) if periods else 0,
        'avg_duration_days':float(np.mean([p['duration_days'] for p in periods])) if periods else 0,'max_duration_days':max((p['duration_days'] for p in periods),default=0),
        'recovery_rate':sum(p['is_recovered'] for p in periods)/len(periods) if periods else None},'periods':periods,'provenance':provenance}

@router.get('/strategy-comparison')
async def compare_strategies(user: str = Depends(get_required_user), strategies: list[str] = Query(['rebalancing','buy_hold','momentum']),days: int = Query(365)):
    raise HTTPException(503,detail='Validated strategy return series are unavailable; simulated comparisons are not published')

@router.get('/risk-metrics')
async def get_risk_metrics(user: str = Depends(get_required_user),days: int = Query(365,ge=30,le=3650),confidence_level: float = Query(.95,gt=0,lt=1),source: str = Query('cointracking')):
    returns,provenance = await _performance(days,user,source)
    dd,_ = _drawdowns(_equity(returns))
    var = float(returns.quantile(1-confidence_level)*100)
    return {'value_at_risk':{f'var_{int(confidence_level*100)}':var,f'cvar_{int(confidence_level*100)}':float(returns[returns*100<=var].mean()*100)},
        'distribution_metrics':{'skewness':float(returns.skew()) if returns.std()>0 else None,'kurtosis':float(returns.kurtosis()+3) if returns.std()>0 else None},
        'drawdown_metrics':{'max_drawdown_pct':float(dd.min()*100),'avg_drawdown_pct':float(dd[dd<0].mean()*100) if (dd<0).any() else 0,'time_in_drawdown_pct':float((dd<0).mean()*100)},'provenance':provenance}
