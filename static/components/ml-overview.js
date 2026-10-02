import { summarizeResults, renderAssetResults } from './ml-result-table.js';
import { apiCall } from '../core/fetcher.js';
import { ensureSelectedSource } from '../core/selected-source.js';
import { StorageService } from '../core/storage-service.js';

let snapshot = null;
let pending = null;
let identity = null;
let sourceRevision = 0;
const text = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value ?? 'Unavailable'; };
const date = value => value ? new Date(value).toLocaleString() : 'Unavailable';
const dailyDate = value => value ? new Date(value).toISOString().slice(0, 10) : 'Unavailable';
const targetLabel = value => ({economic_rule_regime:'Economic regime rules',future_realized_volatility:'Future realized volatility',historical_correlation:'Historical correlation',external_fear_greed:'External Fear & Greed'}[value] || value);
const market = () => document.getElementById('ml-market')?.value || 'crypto';
const mode = () => document.getElementById('ml-universe-mode')?.value || 'portfolio';
const limit = () => document.getElementById('ml-limit')?.value || '25';
const stockSelection = () => window.wealthContextBar?.getContext?.()?.bourse;
const fileKey = () => stockSelection()?.endsWith('.csv') ? stockSelection().slice(5) : window.userSettings?.saxo_selected_file || null;
const source = () => {
    if (market() !== 'stocks') return window.globalConfig?.get?.('data_source') || StorageService.getDataSource() || null;
    const selected = stockSelection();
    if (selected === 'manual_bourse') return 'manual_bourse';
    if (selected === 'saxobank_api' || selected?.startsWith('api:')) return 'saxobank_api';
    if (selected?.startsWith('saxo:')) return 'saxobank';
    return window.globalConfig?.get?.('saxo_source') || 'saxobank';
};
const assets = () => document.getElementById('ml-assets')?.value || (market() === 'stocks' ? 'SPY,QQQ' : 'BTC,ETH,SOL');

function element(tag, content, className) {
    const el = document.createElement(tag);
    if (content != null) el.textContent = content;
    if (className) el.className = className;
    return el;
}

function render() {
    const context = snapshot?.portfolio_context;
    text('ml-scope-summary', mode() === 'portfolio' ? context ? 'Account: '+snapshot.user_id+' · Source: '+snapshot.source+' · Source entries: '+(context.held_positions ?? 'Unavailable')+' · Positions date: '+(context.data_as_of || 'Unavailable')+' · Selected assets: '+context.selected_assets+' · Omitted: '+(context.omitted_assets ?? 0)+' · '+(context.observation_mode === 'dated_read_only_production_copy' ? 'Dated read-only production copy: '+context.captured_at : 'Selected account source')+' · '+context.reason : 'Portfolio context unavailable' : 'Explicit market benchmarks; not your portfolio');
    const coverage = summarizeResults(snapshot?.results);
    text('active-models', snapshot ? `${coverage.available}/${coverage.requested}` : null);
    text('total-predictions', snapshot?.counts?.successful_inferences);
    text('avg-accuracy', null);
    text('ml-confidence', snapshot ? `${coverage.covered}/${coverage.assets}` : null);
    text('decision-score', snapshot ? coverage.diagnostics : null);
    text('last-update', snapshot ? date(snapshot.observed_at) : null);
    for (const id of ['models-overview', 'models-detailed']) {
        const root = document.getElementById(id);
        if (!root) continue;
        root.replaceChildren();
        for (const capability of snapshot?.capabilities || [{ label: 'ML overview', availability: 'Unavailable', reason: 'Snapshot could not be loaded. Check your session and API connection.' }]) {
            const card = element('section', null, 'ml-card');
            card.append(element('h3', capability.label, 'card-title'));
            card.append(element('p', capability.availability, 'status-value'));
            card.append(element('p', capability.reason));
            if (capability.id) {
                const detail = element('button', 'Details', 'btn secondary');
                detail.addEventListener('click', () => details(capability.id));
                card.append(detail);
            }
            root.append(card);
        }
    }
    const root = document.getElementById('live-predictions');
    if (root) {
        root.replaceChildren();
        root.append(element('p', '7/30 calendar-day forecasts and descriptive diagnostics. No allocation integration.'));
        renderAssetResults(root, snapshot?.results || []);
        for (const result of snapshot?.results || []) {
            if (['future_realized_volatility','economic_rule_regime'].includes(result.target)) continue;
            const card=element('details',null,'ml-card');card.append(element('summary',targetLabel(result.target)+' · '+result.availability));
            const value=result.value==null?'Unavailable':typeof result.value==='object'?'Historical correlation matrix (descriptive)':result.unit?.startsWith('index') ? `${result.value}/100` : String(result.value);
            card.append(element('p',value));card.append(element('p',result.reason));
            card.append(element('small','Data: '+dailyDate(result.data_as_of)+' · Provider: '+(result.provenance.provider||'Unavailable')));
            root.append(card);
        }
        if (!snapshot) root.append(element('p', 'Unavailable — snapshot failed; no prediction is substituted.'));

    }
    text('admin-total-models', snapshot?.counts?.files_present);
    text('admin-loaded-models', snapshot?.counts?.models_loaded);
    text('ml-predictor-status', 'Rejected · disabled');
    text('risk-engine-status', 'Unavailable · no verified integration');
    text('alert-engine-status', window.smartfolioPreviewReadOnly ? 'Standby · preview scheduler disabled' : 'Checking alert storage…');
    text('active-alerts-count', null);
    if (document.getElementById('active-alerts-count')) apiCall('/api/alerts/active', {maxRetries:0}).then(response => {
        if (response.ok && Array.isArray(response.data)) {
            text('active-alerts-count', response.data.length);
            text('alert-engine-status', window.smartfolioPreviewReadOnly ? 'Standby · isolated preview storage' : 'Available · alert storage');
        } else text('alert-engine-status', 'Unavailable · '+(response.error || 'alert API failed'));
    }).catch(() => text('alert-engine-status', 'Unavailable · alert API failed'));
}

export async function refresh(force = false) {
    if (mode() === 'portfolio' && market() === 'crypto' && !source()) await ensureSelectedSource();
    if (mode() === 'portfolio' && !source()) { snapshot = null; pending = null; render(); return; }
    const requestedSource = source();
    const requestedFile = market() === 'stocks' ? fileKey() : null;
    const key = `${StorageService.getActiveUser()}|${source()}|${market()}|${assets()}|${mode()}|${limit()}|${requestedFile || ''}|${sourceRevision}`;
    if (key !== identity) { snapshot = null; pending = null; identity = key; }
    if (pending) return pending;
    pending = (async () => {
        try {
            const response = await apiCall(`/api/ml/overview?source=${encodeURIComponent(requestedSource)}&market=${market()}&mode=${mode()}&limit=${limit()}&assets=${encodeURIComponent(assets())}${requestedFile ? '&file_key='+encodeURIComponent(requestedFile) : ''}`);
            const data = response?.data?.capabilities ? response.data : response;
            if (!data?.capabilities || data.user_id !== StorageService.getActiveUser() || data.source !== requestedSource) throw new Error('Invalid snapshot identity');
            if (identity === key) { snapshot = data; }
        } catch (error) {
            if (identity === key) snapshot = null;
            console.warn('ML overview unavailable:', error.message);
        } finally {
            if (identity === key) { pending = null; render(); }
        }
        return snapshot;
    })();
    return pending;
}

export function details(id) {
    const dialog = element('dialog');
    const capability = snapshot?.capabilities.find(c => c.id === id);
    dialog.append(element('h2', capability?.label || 'ML details'));
    dialog.append(element('p', capability?.reason || 'Unavailable — no snapshot'));
    const targets = {regime:'economic_rule_regime',volatility:'future_realized_volatility',sentiment:'external_fear_greed',correlation:'historical_correlation'};
    const payload = { user_id: snapshot?.user_id, source: snapshot?.source, scope: snapshot?.scope, portfolio_context: snapshot?.portfolio_context, capability, results: snapshot?.results.filter(r => r.target === targets[id]), artifacts: snapshot?.artifacts };
    const pre = element('pre', JSON.stringify(payload, null, 2));
    pre.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;max-height:65vh;overflow:auto';
    dialog.append(pre);
    const close = element('button', 'Close', 'btn secondary');
    close.addEventListener('click', () => dialog.remove());
    dialog.append(close);
    document.body.append(dialog);
    dialog.showModal();
}

window.sfML = { refresh, details };
window.addEventListener('dataSourceChanged', () => { sourceRevision++; refresh(true); });
window.addEventListener('bourseSourceChanged', () => { sourceRevision++; refresh(true); });
document.addEventListener('DOMContentLoaded', () => {
    const controls = element('div', null, 'ml-universe-controls');
    const select = element('select');
    select.id = 'ml-market'; select.setAttribute('aria-label', 'Market');
    for (const [value, label] of [['crypto', 'Crypto'], ['stocks', 'Stocks']]) {
        const option = element('option', label); option.value = value; select.append(option);
    }
    const input = element('input'); input.id = 'ml-assets'; input.value = 'BTC,ETH,SOL'; input.setAttribute('aria-label', 'Explicit asset universe');
    const button = element('button', 'Apply universe', 'btn primary');
    button.addEventListener('click', () => refresh(true));
    select.addEventListener('change', () => { input.value = select.value === 'stocks' ? 'SPY,QQQ' : 'BTC,ETH,SOL'; refresh(true); });
    const universe = element('select'); universe.id='ml-universe-mode'; universe.setAttribute('aria-label','Universe');
    for (const [value,label] of [['portfolio','Selected account holdings'],['benchmarks','Explicit benchmarks']]) { const option=element('option',label);option.value=value;universe.append(option); }
    const count = element('select');count.id='ml-limit';count.setAttribute('aria-label','Portfolio asset limit');
    for (const [value,label] of [['25','Top 25 by source value'],['50','Top 50 by source value'],['250','All holdings (up to 250)']]) {const option=element('option',label);option.value=value;count.append(option);}
    input.disabled=true; input.hidden=true;
    universe.addEventListener('change',()=>{input.disabled=universe.value==='portfolio';input.hidden=input.disabled;count.disabled=universe.value==='benchmarks';refresh(true);});
    count.addEventListener('change',()=>refresh(true));
    controls.append(element('span', 'Manual decision support'), select, universe, count, input, button);
    const summary=element('p');summary.id='ml-scope-summary';controls.append(summary);

    document.querySelector('.overview-stats')?.before(controls);
});
