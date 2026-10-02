import { getAuthHeaders } from './auth-guard.js';

let pending = null;
let pendingUser = null;

/** Restore only the authenticated user's configured source; never pick a fallback. */
export async function ensureSelectedSource() {
    const config = window.globalConfig;
    const selected = config?.get?.('data_source');
    if (selected) return selected;
    const user = localStorage.getItem('activeUser');
    if (!user || !config) return null;
    if (pending && pendingUser === user) return pending;
    pendingUser = user;
    const request = (async () => {
        const response = await fetch('/api/users/sources', { headers: getAuthHeaders() });
        if (!response.ok) return null;
        const data = await response.json();
        if (data.user !== user || localStorage.getItem('activeUser') !== user) return null;
        const current = config.get('data_source');
        if (current) return current;
        const candidate = data.current_source;
        const listed = new Set((data.sources || []).map(item => item.key));
        const equivalent = candidate === 'cointracking' ? 'cointracking_csv' : candidate;
        if (!candidate || !listed.has(equivalent) || candidate.startsWith('stub')) return null;
        config.set('data_source', candidate);
        return candidate;
    })();
    pending = request;
    try { return await request; }
    finally { if (pending === request) { pending = null; pendingUser = null; } }
}
