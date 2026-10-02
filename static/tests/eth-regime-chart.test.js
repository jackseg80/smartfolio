import {jest, test, beforeEach, expect} from '@jest/globals';
import {initializeETHRegimeChart, refreshETHRegimeChart} from '../modules/eth-regime-chart.js';

const history = {dates:['2026-09-29','2026-09-30','2026-10-01'], prices:[100,90,110],
    regimes:['Bear Market','Unknown','Bull Market'], unknown_days:1,
    note:'Partial rule-based history. No compatible HMM artifact.', history_limitation:'Not point-in-time decision evidence.'};
const ok = () => ({ok:true,status:200,json:async()=>({ok:true,data:history})});
beforeEach(()=>{
    document.body.innerHTML='<div class="eth-regime-timeframe-selector"><button class="active" data-days="365">1Y</button></div><div id="eth-regime-chart-container"><canvas id="eth-regime-timeline-chart"></canvas></div><div id="eth-regime-error-message"></div><p id="eth-regime-history-note"></p>';
    jest.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue({});
    global.Chart = jest.fn(function(){this.destroy=jest.fn();});
    global.fetch = jest.fn();
});
test('partial history renders actual Unknown periods and the missing-HMM limitation',async()=>{
    fetch.mockResolvedValue(ok());await initializeETHRegimeChart();
    const [,config] = Chart.mock.calls.at(-1);
    expect(config.data.datasets[0].data).toEqual(history.prices);
    expect(document.getElementById('eth-regime-history-note').textContent).toContain('No compatible HMM artifact');
    expect(document.getElementById('eth-regime-history-note').textContent).toContain('1 Unknown days');
    expect(config.options.plugins.tooltip.callbacks.label({parsed:{y:90},dataIndex:1})).toContain('Regime: Unknown');
    expect(document.getElementById('eth-regime-error-message').style.display).toBe('none');
});
test('API reason is visible, stale chart is destroyed, and a successful retry clears the error',async()=>{
    fetch.mockResolvedValueOnce(ok()).mockResolvedValueOnce({ok:false,status:503,json:async()=>({ok:false,error:'Actual provider unavailable'})}).mockResolvedValueOnce(ok());
    await initializeETHRegimeChart();const chart=Chart.mock.instances.at(-1);
    await refreshETHRegimeChart();
    expect(chart.destroy).toHaveBeenCalled();
    expect(document.getElementById('eth-regime-error-message').textContent).toContain('Actual provider unavailable');
    await refreshETHRegimeChart();
    expect(document.getElementById('eth-regime-error-message').style.display).toBe('none');
    expect(document.getElementById('eth-regime-chart-container').classList.contains('loading')).toBe(false);
});
