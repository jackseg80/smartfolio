"""Numerical and isolation checks for the user's second preview review."""
import pandas as pd
import pytest
from services.ml.cycle_diagnostics import describe_cycles
from services.ml.bourse.opportunity_scanner import OpportunityScanner
from services.ml.bourse.portfolio_gap_detector import PortfolioGapDetector
from scripts.repair_ml_stock_observations import repair_frame

@pytest.mark.asyncio
async def test_scan_and_impact_share_classification_without_mutating_holdings(monkeypatch):
    scanner=OpportunityScanner()
    monkeypatch.setattr(scanner,'_enrich_position_with_sector',lambda _: 'Technology')
    positions=[{'symbol':'A','market_value':40},{'symbol':'B','market_value':60,'sector':'Other'}]
    classified=scanner._classify_positions(positions)
    before=scanner._extract_sector_allocation(classified)
    impact=await PortfolioGapDetector().calculate_reallocation_impact(classified,[],[])
    assert before==impact['before']==impact['after']=={'Technology':40,'Other':60}
    assert 'sector' not in positions[0]
    sales=await PortfolioGapDetector().calculate_reallocation_impact(classified,[{'symbol':'A','sale_value':10}],[])
    assert sales['after']=={'Technology':30,'Other':60,'Cash':10}
    assert classified[0]['market_value']==40

def test_indeterminate_gap_preserves_known_allocation_and_uncertainty():
    scanner=OpportunityScanner()
    rows=scanner._describe_sector_bounds({'Technology':40,'Other':60},{'Technology':50,'Healthcare':50,**{s:0 for s in ['Financials','Consumer Discretionary','Communication Services','Industrials','Consumer Staples','Energy','Utilities','Real Estate','Materials']}},60)
    technology=next(r for r in rows if r['sector']=='Technology')
    assert technology['known_pct']==40 and technology['minimum_gap_pct']==0 and technology['possible_gap_pct']==10
    assert technology['status']=='Indeterminate'
    assert not scanner._detect_gaps({'Technology':40,'Other':60},5,unclassified_pct=60)

def test_missing_early_cycle_is_visible_and_partial_drawdown_is_preserved():
    close=pd.Series(100.,index=pd.date_range('2017-08-18','2026-09-29',tz='UTC'))
    data=describe_cycles(close,{'provider':'fixture','dataset_id':'fixture'})
    assert [c['cycle'] for c in data['cycles']]==[1,2,3,4]
    first,second,third,fourth=data['cycles']
    assert first['coverage']=='Unavailable' and first['points']==[]
    assert second['coverage']=='Partial' and not second['complete_cycle'] and not second['anchor_available']
    assert second['points'][0]['drawdown']==0 and second['points'][0]['normalized'] is None
    assert third['complete_cycle'] and fourth['coverage']=='Ongoing'

def test_stock_repair_drops_only_trailing_invalid_rows_and_non_sessions():
    frame=pd.DataFrame({'date':['2026-09-25','2026-09-26','2026-09-28','2026-09-29'],'close':[100,100,102,float('nan')]})
    repaired,audit=repair_frame(frame,'XSWX',pd.Timestamp('2026-09-30',tz='UTC'))
    assert repaired.date.tolist()==['2026-09-25','2026-09-28']
    assert repaired.close.tolist()==[100,102]
    assert set(audit['excluded_dates'])=={'2026-09-26','2026-09-29'}
    frame.loc[0,'close']=float('nan')
    repaired,_=repair_frame(frame,'XSWX',pd.Timestamp('2026-09-30',tz='UTC'))
    assert pd.isna(repaired.close.iloc[0])  # Interior invalid values remain rejected by the service.
