import { jest, describe, test, beforeEach, expect } from '@jest/globals';

const apiCall = jest.fn();
let user = 'alice';
let source = 'source-a';
jest.unstable_mockModule('../core/fetcher.js', () => ({ apiCall }));
jest.unstable_mockModule('../core/storage-service.js', () => ({ StorageService: { getActiveUser: () => user, getDataSource: () => source } }));
const { refresh } = await import('../components/ml-overview.js');

function snapshot() {
    return { user_id: user, source, observed_at: '2026-09-30T10:00:00Z',
        counts: { files_present: 1, models_loaded: 1, successful_inferences: 1 },
        capabilities: [{ id: 'volatility', label: 'Volatility', availability: 'Available', reason: 'Confirmed point estimate' }],
        results: [{ asset: 'BTC', target: 'future_realized_volatility', horizon: '7d', value: 0, nature: 'forecast', availability: 'Available', reason: 'Real zero', unit: 'annualized fraction', validation: { state: 'retrospectively_validated' }, provenance: { provider: 'Test source' }, data_as_of: '2026-09-29', target_date: '2026-10-06' }] };
}

beforeEach(() => {
    user = 'alice'; source = 'source-a'; apiCall.mockReset();
    document.body.innerHTML = '<div id="models-overview"></div><div id="models-detailed"></div><div id="live-predictions"></div><span id="active-models"></span><span id="ml-confidence"></span><span id="last-update"></span>';
});

describe('Verified ML snapshot rendering', () => {
    test('zero stays zero and coverage counters never become confidence percentages', async () => {
        apiCall.mockResolvedValue({ ok: true, data: snapshot() });
        await refresh(true);
        expect(document.getElementById('live-predictions').textContent).toContain('0.00%');
        expect(document.getElementById('ml-confidence').textContent).toBe('1/1');
        expect(document.getElementById('active-models').textContent).toBe('1/1');
    });
    test('expired session clears previous values and shows a failure reason', async () => {
        apiCall.mockResolvedValueOnce({ ok: true, data: snapshot() });
        await refresh(true);
        apiCall.mockResolvedValueOnce({ ok: false, status: 401, data: null });
        await refresh(true);
        expect(document.getElementById('live-predictions').textContent).not.toContain('0.00%');
        expect(document.getElementById('live-predictions').textContent).toContain('Unavailable');
        expect(document.getElementById('active-models').textContent).toBe('Unavailable');
    });
    test('switching user or source cannot reuse the previous snapshot', async () => {
        apiCall.mockResolvedValue({ ok: true, data: snapshot() });
        await refresh(true);
        user = 'bob'; source = 'source-b';
        apiCall.mockResolvedValue({ ok: true, data: snapshot() });
        await refresh();
        expect(apiCall).toHaveBeenLastCalledWith(expect.stringContaining('source-b'));
        expect(apiCall).toHaveBeenCalledTimes(2);
    });
    test('matrix and external index units are rendered without false percentages', async () => {
        const data = snapshot();
        data.results[0].value = 45; data.results[0].unit = 'index [0,100]'; data.results[0].target = 'external_fear_greed'; data.results[0].nature = 'diagnostic';
        apiCall.mockResolvedValue({ ok: true, data });
        await refresh(true);
        expect(document.getElementById('live-predictions').textContent).toContain('45/100');
        expect(document.getElementById('live-predictions').textContent).not.toContain('4500');
    });
    test('provider text is escaped as text, including failure reasons', async () => {
        const data = snapshot(); data.capabilities[0].reason = '<img src=x onerror=alert(1)>';
        apiCall.mockResolvedValue({ ok: true, data });
        await refresh(true);
        expect(document.getElementById('models-overview').querySelector('img')).toBeNull();
        expect(document.getElementById('models-overview').textContent).toContain('<img');
    });
});


test('personal mode is the default and mismatched source cannot be displayed', async () => {
    const data=snapshot();data.source='wrong-source';
    apiCall.mockResolvedValue({ok:true,data});
    await refresh(true);
    expect(apiCall).toHaveBeenLastCalledWith(expect.stringContaining('mode=portfolio'));
    expect(document.getElementById('active-models').textContent).toBe('Unavailable');
});
