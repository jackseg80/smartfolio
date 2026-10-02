import hashlib,json
from datetime import datetime,timezone

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI,Header,HTTPException
from fastapi.testclient import TestClient

from api.deps import get_required_user
from services.ml.live_observations import extend_crypto
from services.ml.portfolio_context import stock_symbol

NOW=datetime(2026,10,1,tzinfo=timezone.utc)

def base():
 dates=pd.date_range('2026-09-27',periods=3,tz='UTC')
 close=pd.Series([100.1234567890123,102.,101.],index=dates,name='close')
 return close,{'provider':'binance_spot_public_market_data','sha256':'frozen-base','dataset_id':'frozen-dataset','file':'BTC.csv'}

def response(start='2026-09-30',price=105.):
 day=pd.Timestamp(start,tz='UTC');rows=[[int(day.timestamp()*1000),0,0,0,str(price),0,int((day+pd.Timedelta(days=1)).timestamp()*1000)-1]]
 return httpx.Response(200,json=rows,request=httpx.Request('GET','https://data-api.binance.vision/api/v3/klines'))

def test_append_preserves_evaluation_identity_and_no_repeat_http(tmp_path,monkeypatch):
 close,receipt=base();calls=[]
 monkeypatch.setattr(httpx,'get',lambda *a,**kw:(calls.append(kw) or response()))
 extended,meta=extend_crypto(tmp_path,'BTC',close,receipt,NOW)
 pd.testing.assert_series_equal(extended.iloc[:len(close)],close,check_freq=False,check_names=False)
 assert meta['dataset_id']!=receipt['dataset_id'] and meta['evaluation_dataset_id']==receipt['dataset_id'] and meta['evaluation_sha256']==receipt['sha256']
 replay,again=extend_crypto(tmp_path,'BTC',close,receipt,NOW)
 pd.testing.assert_series_equal(extended,replay)
 assert len(calls)==1 and again==meta and replay.index[-1].strftime('%Y-%m-%d')=='2026-09-30'

@pytest.mark.parametrize('price',[0.,-1.,float('nan')])
def test_invalid_new_closes_rejected(tmp_path,monkeypatch,price):
 close,receipt=base();monkeypatch.setattr(httpx,'get',lambda *a,**kw:response(price=price))
 with pytest.raises(ValueError):extend_crypto(tmp_path,'BTC',close,receipt,NOW)
 assert not (tmp_path/'cache/ml_observations/crypto/BTC.json').exists()

def test_missing_day_and_historical_mutation_rejected(tmp_path,monkeypatch):
 close,receipt=base();monkeypatch.setattr(httpx,'get',lambda *a,**kw:response())
 extend_crypto(tmp_path,'BTC',close,receipt,NOW)
 changed=close.copy();changed.iloc[0]+=1
 with pytest.raises(ValueError,match='historical prefix'):extend_crypto(tmp_path,'BTC',changed,receipt,NOW)
 metadata=tmp_path/'cache/ml_observations/crypto/BTC.json';metadata.unlink()
 from services.ml.live_observations import _last_attempt
 _last_attempt.clear()
 monkeypatch.setattr(httpx,'get',lambda *a,**kw:response(start='2026-09-29'))
 with pytest.raises(ValueError,match='incomplete'):extend_crypto(tmp_path,'BTC',close,receipt,NOW)

def test_upstream_failure_preserves_observation_date(tmp_path,monkeypatch):
 close,receipt=base()
 def broken(*a,**kw):raise httpx.ConnectError('offline')
 monkeypatch.setattr(httpx,'get',broken)
 unchanged,meta=extend_crypto(tmp_path,'BTC',close,receipt,NOW)
 pd.testing.assert_series_equal(close,unchanged,check_freq=False,check_names=False)
 assert meta==receipt

@pytest.mark.parametrize('symbol,expected',[('SLHN','SLHN.SW'),('SLHN:xswx','SLHN.SW'),('WRDUSW_CHF:xswx','WRDUSW.SW'),('WRDUSW_CHF.SW','WRDUSW.SW'),('SMH:xlon','SMH.L'),('BRKb:xnys','BRK-B')])
def test_same_instrument_vendor_mapping(symbol,expected):assert stock_symbol(symbol)==expected

def test_advanced_uses_two_users_two_sources_and_no_mock(monkeypatch):
 from api import advanced_analytics_endpoints as endpoints
 calls=[]
 async def read(user,source,market):
  calls.append((user,source,market));return {'items':[{'symbol':'BTC','value_usd':1}],'source_used':source}
 monkeypatch.setattr(endpoints,'read_context',read)
 monkeypatch.setattr(endpoints,'get_cached_history',lambda *a,**kw:[])
 app=FastAPI();app.include_router(endpoints.router)
 def identity(x_user: str=Header(None,alias='X-User')):
  if not x_user:raise HTTPException(401,'Session required')
  return x_user
 app.dependency_overrides[get_required_user]=identity
 client=TestClient(app)
 for user in ['alice','bob']:
  for source in ['a','b']:
   result=client.get('/analytics/advanced/metrics?source='+source,headers={'X-User':user})
   assert result.status_code==503 and 'total_return_pct' not in result.json()
   assert calls[-1]==(user,source,'crypto')
 assert client.get('/analytics/advanced/metrics').status_code==401
 assert client.get('/analytics/advanced/strategy-comparison',headers={'X-User':'alice'}).status_code==503

def test_advanced_monthly_returns_and_unrecovered_drawdown():
 from api.advanced_analytics_endpoints import _calculate,_drawdowns,_equity
 returns=pd.Series([-.1,.05,-.2,0.],index=pd.date_range('2026-01-30',periods=4,tz='UTC'))
 metrics=_calculate(returns,{'method':'test'})
 assert metrics.max_drawdown_pct==pytest.approx(-24.4)
 assert metrics.best_month_pct==pytest.approx(-5.5) and metrics.worst_month_pct==pytest.approx(-20)
 assert metrics.omega_ratio==pytest.approx(1/6) and metrics.positive_months_pct==0
 assert metrics.drawdown_periods[-1]['is_recovered'] is False
 assert metrics.max_drawdown_duration_days==4
 prefix,_=_drawdowns(_equity(returns.iloc[:2]))
 changed=returns.copy();changed.iloc[2]=50
 after,_=_drawdowns(_equity(changed))
 pd.testing.assert_series_equal(prefix,after.loc[prefix.index])

def test_cycle_reference_has_all_three_completed_cycles():
 from services.ml.reliability import CapabilityService
 from services.ml.cycle_diagnostics import historical_cycle_comparison
 service=CapabilityService(refresh_observations=False)
 close,receipt=service.history('crypto','BTC')
 data=historical_cycle_comparison(service.root,close,receipt)
 assert [c['coverage'] for c in data['cycles']]==['Complete','Complete','Complete','Ongoing']
 assert all(len(c['points'])>800 for c in data['cycles'])
 assert all(c['provider'].startswith('Coin Metrics') for c in data['cycles'][:3])
 assert data['cycles'][-1]['provider']=='binance_spot_public_market_data'

def test_stock_extension_rejects_adjustment_revision_and_missing_sessions(tmp_path,monkeypatch):
 from services.ml.live_observations import extend_stocks,_last_attempt
 import yfinance as yf
 close=pd.Series([100.,101.,102.],index=pd.DatetimeIndex(pd.to_datetime(['2026-09-25','2026-09-28','2026-09-29'],utc=True)),name='close')
 receipt={'provider':'Yahoo Finance via yfinance','exchange_calendar':'XNYS','dataset_id':'stock-frozen','sha256':'stock-sha','adjustment_policy':'auto_adjust=True'}
 dates=pd.to_datetime(['2026-09-25','2026-09-28','2026-09-29','2026-09-30'])
 original=pd.DataFrame({'Close':[100.,101.,102.,104.]},index=dates)
 monkeypatch.setattr(yf,'download',lambda *a,**kw:original)
 extended,meta=extend_stocks(tmp_path,'AAPL',close,receipt,NOW)
 np.testing.assert_array_equal(extended.iloc[:3],close)
 assert len(extended)==4 and meta['evaluation_sha256']=='stock-sha' and meta['dataset_id']!='stock-frozen'
 assert meta['raw_response_file'] and meta['yfinance_version']==yf.__version__
 # A newer action that adjusts a historical overlapping close must be rejected.
 next_day=datetime(2026,10,2,tzinfo=timezone.utc)
 revised=original.copy();revised.loc[dates[2],'Close']=51
 revised.loc[pd.Timestamp('2026-10-01'),'Close']=106
 monkeypatch.setattr(yf,'download',lambda *a,**kw:revised)
 with pytest.raises(ValueError,match='revised'):extend_stocks(tmp_path,'AAPL',close,receipt,next_day)
 _last_attempt.clear()
 monkeypatch.setattr(yf,'download',lambda *a,**kw:original)
 with pytest.raises(ValueError,match='missing'):extend_stocks(tmp_path,'AAPL',close,receipt,next_day)


def test_unverified_currency_listing_cannot_use_the_chf_model():
 with pytest.raises(ValueError):stock_symbol('WRDUSW_USD:xswx')
 with pytest.raises(ValueError):stock_symbol('WRDUSW_CHF:xlon')

@pytest.mark.asyncio
async def test_stale_holdings_cannot_move_advanced_metrics_to_an_old_window(monkeypatch):
 from api import advanced_analytics_endpoints as endpoints
 async def read(*a,**kw):return {'items':[{'symbol':'BTC','value_usd':1},{'symbol':'OLD','value_usd':1}],'source_used':'selected'}
 monkeypatch.setattr(endpoints,'read_context',read)
 dates=pd.date_range(end=pd.Timestamp.now(tz='UTC').normalize()-pd.Timedelta(days=1),periods=400)
 fresh=[(int(d.timestamp()),float(100+i)) for i,d in enumerate(dates)]
 stale=[(int((d-pd.Timedelta(days=150)).timestamp()),v) for d,(_,v) in zip(dates,fresh)]
 monkeypatch.setattr(endpoints,'get_cached_history',lambda symbol,**kw:fresh if symbol=='BTC' else stale)
 returns,meta=await endpoints._performance(365,'alice','selected')
 assert meta['source']=='selected' and meta['coverage_by_current_value']==.5 and meta['excluded_stale_assets']==1 and meta['availability']=='Partial'
 assert returns.index[-1]==dates[-1] and len(returns)==365
 assert 'without filling gaps' in meta['reason']
