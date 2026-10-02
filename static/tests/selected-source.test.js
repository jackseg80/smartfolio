import { jest, beforeEach, describe, test, expect } from '@jest/globals';
jest.unstable_mockModule('../core/auth-guard.js', () => ({ getAuthHeaders: () => ({ 'X-User': localStorage.getItem('activeUser'), Authorization: 'Bearer test-only' }) }));
const { ensureSelectedSource } = await import('../core/selected-source.js');
let selected;
beforeEach(() => {
    selected = null; localStorage.clear(); localStorage.setItem('activeUser', 'alice');
    window.globalConfig = { get: () => selected, set: jest.fn((key, value) => { selected = value; }) };
    global.fetch = jest.fn();
});
function response(source = 'cointracking_api', user = 'alice') {
    return { ok: true, json: async () => ({ user, current_source: source, sources: [{ key: 'cointracking_api' }, { key: 'cointracking_csv' }] }) };
}
describe('Authenticated source restoration', () => {
    test('fresh browser restores only the configured source, without server writes', async () => {
        fetch.mockResolvedValue(response());
        expect(await ensureSelectedSource()).toBe('cointracking_api');
        expect(fetch).toHaveBeenCalledWith('/api/users/sources', { headers: { 'X-User': 'alice', Authorization: 'Bearer test-only' } });
        expect(window.globalConfig.set).toHaveBeenCalledWith('data_source', 'cointracking_api');
    });
    test('explicit selection remains selected', async () => {
        selected = 'manual_crypto'; expect(await ensureSelectedSource()).toBe('manual_crypto'); expect(fetch).not.toHaveBeenCalled();
    });
    test.each(['unknown', 'stub_conservative', null])('missing or unlisted configured source %s stays unavailable', async source => {
        fetch.mockResolvedValue(response(source)); expect(await ensureSelectedSource()).toBeNull(); expect(window.globalConfig.set).not.toHaveBeenCalled();
    });
    test('another identity cannot supply the source', async () => {
        fetch.mockResolvedValue(response('cointracking_api', 'bob')); expect(await ensureSelectedSource()).toBeNull();
    });
    test('expired session supplies no source', async () => {
        fetch.mockResolvedValue({ ok: false }); expect(await ensureSelectedSource()).toBeNull();
    });
    test('concurrent readers share one read', async () => {
        fetch.mockResolvedValue(response()); expect(await Promise.all([ensureSelectedSource(), ensureSelectedSource()])).toEqual(['cointracking_api', 'cointracking_api']); expect(fetch).toHaveBeenCalledTimes(1);
    });
    test('an identity change during the read cannot persist old preferences', async () => {
        fetch.mockImplementation(async () => { localStorage.setItem('activeUser', 'bob'); return response(); }); expect(await ensureSelectedSource()).toBeNull(); expect(window.globalConfig.set).not.toHaveBeenCalled();
    });
});
