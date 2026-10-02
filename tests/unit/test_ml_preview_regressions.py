"""Regression gates for issues uncovered by authenticated robot2 preview."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from api.risk_endpoints import risk_dashboard_cache_key

@pytest.mark.parametrize('field,value', [('user','bob'),('source','source-b'),('price_history_days',365),('lookback_days',90),('use_dual_window',False),('min_history_days',365),('min_coverage_pct',.9),('min_asset_count',6),('risk_version','legacy'),('csv_hint','new.csv'),('min_usd',10)])
def test_risk_cache_does_not_reuse_other_scope_or_window(field,value):
    context=dict(user='alice',source='source-a',min_usd=1,risk_version='v2_active',csv_hint='',price_history_days=30,lookback_days=30,use_dual_window=True,min_history_days=180,min_coverage_pct=.8,min_asset_count=5)
    assert risk_dashboard_cache_key(**context) != risk_dashboard_cache_key(**{**context,field:value})

@pytest.mark.asyncio
async def test_hmm_history_never_maps_state_zero_to_bear(monkeypatch):
    import api.ml_crypto_endpoints as module
    index=pd.date_range('2026-01-01',periods=30)
    history=[(int(d.timestamp()),100+i) for i,d in enumerate(index)]
    features=pd.DataFrame(dict(drawdown_from_peak=[-.5]*10+[0.]*20,days_since_peak=30,trend_30d=0.,market_volatility=.3),index=index)
    class Detector:
        num_regimes=4
        regime_names=['Unverified']*4
        feature_columns=list(features.columns)
        scaler=SimpleNamespace(transform=lambda frame:frame)
        hmm_model=SimpleNamespace(predict=lambda frame:np.zeros(len(frame),dtype=int))
        async def prepare_regime_features(self,**kwargs): return features.copy()
        def load_model(self,*args): return True
    monkeypatch.setattr(module,'BTCRegimeDetector',Detector)
    monkeypatch.setattr(module.price_history,'get_cached_history',lambda *args,**kwargs:history)
    monkeypatch.setattr(module,'_detect_regime_rule_based_optimized',lambda dd,*args:{'regime_id':0,'regime_name':'Bear Market'} if dd<-.3 else None)
    monkeypatch.setattr(module,'_regime_history_cache',{})
    response=await module.get_crypto_regime_history(symbol='BTC',lookback_days=365)
    data=json.loads(response.body)['data']
    assert data['regimes']==['Bear Market']*10+['State A']*20
    assert data['regime_ids']==[0]*10+[4]*20
    assert data['regime_id_mapping']['4']=='State A'
    assert data['economic_mapping_verified'] is False and data['retrospective'] is True


