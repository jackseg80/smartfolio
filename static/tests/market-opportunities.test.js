import { jest } from '@jest/globals';
jest.unstable_mockModule('../core/auth-guard.js', () => ({ getAuthHeaders: () => ({ 'X-User': 'test' }) }));
const { safeFetch: realSafeFetch } = await import('../modules/http.js');
const { matchesContext, renderAnalysis, mountMarketOpportunities } = await import('../components/market-opportunities.js');

function page() {
    document.body.innerHTML = '<section><p id="opportunityMethod"></p></section>' +
        ['portfolioGaps', 'opportunitiesTable', 'suggestedSales', 'impactSimulator', 'gapsCount', 'opportunitiesCount', 'salesCount']
            .map(id => `<div id="${id}"></div>`).join('') +
        '<button id="btnScanOpportunities"></button><button id="btnExportTextOpps"></button>';
    window.getApiBase = () => '';
}
function result() {
    return { version: 2, context: { user_id: 'test', source: 'saxobank_csv', file_key: 'selected.csv', reporting_currency: 'USD', valuation_currency: 'EUR', warnings: [] },
        generated_at: '2026-10-01', snapshot_id: 'scope', horizon_details: { label: '6-12 Months', signal_sessions: 189 },
        target_note: 'Generic reference', cash: { value_usd: null, reason: 'Selected cash unavailable' },
        exposure: { coverage: .4, unclassified_pct: 60, reason: 'Retain uncertainty', sector_assessment: [] },
        positions: [], candidates: [], holding_reviews: [], automatic_sales: [], automatic_sale_reason: 'Acquisition dates unavailable',
        candidate_summary: { total: 11, screened: 0, held: 1, unavailable: 10, ranking_comparable: true },
        conclusion: 'Zero results does not establish that this portfolio is balanced.' };
}
beforeEach(() => { page(); window.getCurrentUser = () => 'test'; });
test('response matching checks account, source and explicit CSV', () => {
    const context = { user: 'test', source: 'saxobank_csv', fileKey: 'selected.csv' };
    expect(matchesContext(result(), context)).toBe(true);
    expect(matchesContext(result(), { ...context, user: 'other' })).toBe(false);
    expect(matchesContext(result(), { ...context, fileKey: 'other.csv' })).toBe(false);
    expect(matchesContext(result(), { ...context, source: 'manual_bourse' })).toBe(false);
});
test('zero results retain uncertainty and missing cash blocks funding form', () => {
    renderAnalysis(result());
    expect(document.querySelector('.mo-metrics').textContent).toContain('Unclassified exposure60.00%');
    expect(document.body.textContent).toContain('Zero results does not establish');
    expect(document.querySelector('#opportunitiesTable .mo-metrics').textContent).toContain('Unavailable / excluded10');
});
test('provider text and source URLs cannot inject markup or javascript', () => {
    const data = result(); data.conclusion = '<img src=x onerror=alert(1)>';
    data.positions = [{ name: '<script>bad()</script>', symbol: 'TEST', isin: null, quote_currency: 'USD',
        classification: { weights: {}, known_fraction: 0, source_kind: 'unavailable', reason: '<b>unsafe</b>', source_url: 'javascript:alert(1)' },
        geography: { reason: 'Unavailable' }, mapping_reason: 'Unavailable' }];
    renderAnalysis(data);
    expect(document.querySelector('img,script')).toBeNull();
    expect(document.querySelector('a')).toBeNull();
    expect(document.body.textContent).toContain('<script>bad()</script>');
});
test('a late scan response cannot cross a source or account change', async () => {
    let context = { user: 'test', source: 'saxobank_csv', fileKey: 'selected.csv', horizon: 'medium' };
    let resolve;
    window.safeFetch = jest.fn(() => new Promise(r => { resolve = r; }));
    const controller = mountMarketOpportunities({ getContext: () => context, readTargets: () => null, invalidatePage: () => {} });
    const pending = controller.scan();
    context = { ...context, user: 'other' }; controller.invalidate();
    resolve({ ok: true, data: { data: result() } });
    await pending;
    expect(document.querySelector('#portfolioGaps .mo-metrics')).toBeNull();
    expect(document.getElementById('btnScanOpportunities').disabled).toBe(false);
});
test('a settings change during asynchronous JSON parsing also rejects the response', async () => {
    let context = { user: 'test', source: 'saxobank_csv', fileKey: 'selected.csv', horizon: 'medium' };
    let resolveBody;
    window.safeFetch = realSafeFetch;
    global.fetch = jest.fn(async () => ({ ok: true, status: 200, headers: { get: () => null }, json: () => new Promise(r => { resolveBody = r; }) }));
    const controller = mountMarketOpportunities({ getContext: () => context, readTargets: () => null, invalidatePage: () => {} });
    const pending = controller.scan();
    await new Promise(resolve => setTimeout(resolve, 0));
    context.horizon = 'long'; controller.invalidate();
    resolveBody({ data: result() }); await pending;
    expect(document.querySelector('#portfolioGaps .mo-metrics')).toBeNull();
});

function nativeResponse(data, status = 200) {
    return { ok: status === 200, status, headers: { get: () => null }, json: async () => data };
}
function controller() {
    return mountMarketOpportunities({ getContext: () => ({ user: 'test', source: 'saxobank_csv', fileKey: 'selected.csv', horizon: 'medium' }),
        readTargets: () => null, invalidatePage: () => {} });
}
test('scan uses the real safeFetch decoded-data contract', async () => {
    window.safeFetch = realSafeFetch;
    global.fetch = jest.fn(async () => nativeResponse({ data: result() }));
    await controller().scan();
    expect(document.querySelector('#portfolioGaps .mo-metrics').textContent).toContain('Industry classification coverage40.00%');
    expect(document.getElementById('impactSimulator').textContent).toContain('Selected cash unavailable');
    expect(document.body.textContent).not.toContain('response.json is not a function');
});
test('scan displays the real safeFetch validation error', async () => {
    window.safeFetch = realSafeFetch;
    global.fetch = jest.fn(async () => nativeResponse({ detail: 'Selected source changed. Scan again.' }, 422));
    await controller().scan();
    expect(document.getElementById('portfolioGaps').textContent).toBe('Selected source changed. Scan again.');
    expect(document.getElementById('btnScanOpportunities').disabled).toBe(false);
});
test('manual fixture scenario also uses the real safeFetch decoded-data contract', async () => {
    const data = result(); data.cash = { value_usd: 100, reason: 'Synthetic fixture cash' };
    data.candidates = [{ id: 'C', symbol: 'C', name: 'Fixture candidate', kind: 'ETF', status: 'screened', intended_sector: 'Technology' }];
    const computed = { context: data.context, snapshot_id: data.snapshot_id, scenario: {
        cash_before_usd: 100, cash_after_usd: 75, before_total_usd: 100, after_total_usd: 100, costs_and_slippage_usd: 0,
        exposure_before: { sector_assessment: [] }, exposure_after: { sector_assessment: [] }, limits: ['Synthetic arithmetic fixture'] } };
    window.safeFetch = realSafeFetch;
    global.fetch = jest.fn().mockResolvedValueOnce(nativeResponse({ data })).mockResolvedValueOnce(nativeResponse({ data: computed }));
    await controller().scan();
    const form = document.querySelector('#impactSimulator form');
    form.querySelector('[aria-label="buy C estimated amount in USD"]').value = '25';
    form.querySelector('input[type=checkbox]').checked = true;
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(global.fetch).toHaveBeenCalledTimes(2);
    const request = JSON.parse(global.fetch.mock.calls[1][1].body);
    expect(request.scenario.trades).toEqual([{ side: 'buy', id: 'C', amount_usd: 25 }]);
    expect(document.querySelector('#impactSimulator [role=status]').textContent).toContain('Estimated cash: 100.00 → 75.00 USD');
    expect(form.querySelector('button').disabled).toBe(false);
});

test('critical uncertainty stays visible while full evidence remains in closed disclosures', () => {
    const data = result(); data.exposure.unvalued_positions = 2;
    data.exposure.coverage_scope = 'valued_subset_only';
    renderAnalysis(data);
    const root = document.getElementById('portfolioGaps');
    expect(root.querySelector('.mo-notice-warning').textContent).toContain('positions with known values only');
    expect(root.querySelector('.mo-notice-warning').closest('details')).toBeNull();
    const provenance = [...root.querySelectorAll('details')].find(d => d.textContent.includes('selected.csv'));
    expect(provenance.open).toBe(false);
    expect(provenance.textContent).toContain('Generic reference');
    expect(document.querySelector('#opportunitiesTable details').textContent).toContain('personal trading eligibility are unassessed');
    expect(document.getElementById('opportunityMethod').textContent).toContain('No validated forward-return forecast');
});
