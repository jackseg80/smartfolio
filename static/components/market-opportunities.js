import { getAuthHeaders } from '../core/auth-guard.js';

export const sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
    'Communication Services', 'Industrials', 'Consumer Staples', 'Energy', 'Utilities', 'Real Estate', 'Materials'];
const show = (v, digits = 2) => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : 'Unavailable';
const elem = (tag, value, parent) => {
    const node = document.createElement(tag);
    if (value !== undefined) node.textContent = String(value);
    parent?.append(node);
    return node;
};
const paragraph = (parent, text) => elem('p', text, parent);
function stack(primary, secondary) {
    const node = elem('div'); node.className = 'mo-stack';
    elem('strong', primary, node);
    if (secondary) elem('small', secondary, node);
    return node;
}
function badge(text, tone = 'neutral') {
    const node = elem('span', text); node.className = `mo-badge mo-badge-${tone}`;
    return node;
}
function note(parent, text, warning = false) {
    const node = paragraph(parent, text); node.className = warning ? 'mo-notice mo-notice-warning' : 'mo-notice';
    return node;
}
function drawer(parent, title) {
    const node = elem('details', undefined, parent); node.className = 'mo-details';
    elem('summary', title, node);
    const content = elem('div', undefined, node); content.className = 'mo-details-body';
    return content;
}
function metrics(parent, entries) {
    const grid = elem('div', undefined, parent); grid.className = 'mo-metrics';
    entries.forEach(([label, value, detail]) => {
        const card = elem('div', undefined, grid); card.className = 'mo-metric';
        elem('span', label, card).className = 'mo-metric-label';
        elem('strong', value, card).className = 'mo-metric-value';
        if (detail) elem('small', detail, card);
    });
}
function table(parent, labels, rows, { numeric = [], privateRows = false, caption } = {}) {
    const scroll = elem('div', undefined, parent); scroll.className = 'mo-table-scroll';
    scroll.tabIndex = 0; scroll.setAttribute('role', 'region'); scroll.setAttribute('aria-label', caption || labels.join(', '));
    const t = elem('table', undefined, scroll); t.className = 'mo-table';
    t.style.minWidth = `${Math.max(600, labels.length * 110)}px`;
    if (caption) elem('caption', caption, t);
    const head = elem('tr', undefined, elem('thead', undefined, t));
    labels.forEach((label, i) => {
        const cell = elem('th', label, head); cell.scope = 'col';
        if (numeric.includes(i)) cell.className = 'mo-number';
    });
    const body = elem('tbody', undefined, t);
    rows.forEach(values => {
        const tr = elem('tr', undefined, body);
        values.forEach((value, i) => {
            const cell = elem('td', undefined, tr);
            if (numeric.includes(i)) cell.classList.add('mo-number');
            if (privateRows) cell.classList.add('mo-private');
            if (value instanceof Node) cell.append(value);
            else cell.textContent = String(value ?? 'Unavailable');
        });
    });
    return t;
}
function sourceLink(url) {
    const node = elem('span', 'Unavailable');
    if (typeof url !== 'string') return node;
    try {
        const parsed = new URL(url);
        if (parsed.protocol !== 'https:' || !['www.ishares.com', 'finance.yahoo.com'].includes(parsed.hostname)) return node;
        const a = elem('a', 'View source'); a.className = 'mo-source-link';
        a.href = parsed.href; a.target = '_blank'; a.rel = 'noopener noreferrer';
        return a;
    } catch { return node; }
}
export function matchesContext(data, context) {
    return data?.version === 2 && data.context?.user_id === context.user && data.context.source === context.source &&
        (!context.fileKey || data.context.file_key === context.fileKey);
}
export function renderAnalysis(data) {
    const exposure = data.exposure;
    document.getElementById('opportunityMethod').textContent =
        `${data.horizon_details.label} holding horizon · ${data.horizon_details.signal_sessions} historical sessions. No validated forward-return forecast.`;
    const root = document.getElementById('portfolioGaps'); root.replaceChildren();
    const certain = exposure.sector_assessment.filter(s => ['Underweight', 'Overweight'].includes(s.status)).length;
    metrics(root, [
        ['Industry classification coverage', exposure.coverage === null ? 'Unavailable' : `${show(exposure.coverage * 100)}%`, 'Verified sector evidence'],
        ['Unclassified exposure', `${show(exposure.unclassified_pct)}%`, 'Composition remains unknown'],
        ['Unvalued positions', exposure.unvalued_positions || 0, 'Missing source valuation'],
        ['Certain deviations', certain, 'Supported by exposure bounds']
    ]);
    note(root, `Scope: ${exposure.coverage_scope === 'valued_subset_only' ? 'positions with known values only' : 'all valued securities, excluding cash'}. Unvalued positions: ${exposure.unvalued_positions || 0}. ${exposure.reason}`, true);
    note(root, data.conclusion);
    table(root, ['Industry', 'Verified range (%)', 'Target (%)', 'Assessment'],
        exposure.sector_assessment.map(s => [s.sector,
            s.lower_pct === null || s.upper_pct === null ? 'Unavailable' : `${show(s.lower_pct)} – ${show(s.upper_pct)}`,
            show(s.target_pct), badge(s.status, s.status === 'Indeterminate' ? 'warning' : 'neutral')]),
        { numeric: [1, 2], caption: 'Economic sectors · lower and conservative upper bounds' });
    const provenance = drawer(root, 'Data provenance, allocation reference and limits');
    const privateInfo = elem('div', undefined, provenance); privateInfo.className = 'mo-private';
    paragraph(privateInfo, `Source: ${data.context.source} · selected CSV: ${data.context.file_key || 'Not applicable'} · imported: ${data.context.imported_at || 'Unavailable'} · valuation date: ${data.context.valuation_as_of || 'Unconfirmed'} · reporting currency: USD.`);
    paragraph(privateInfo, `Value currency: ${data.context.valuation_currency}. FX observation: ${data.context.fx_as_of || 'USD identity / unavailable'}. Saved cash: ${show(data.cash.value_usd)} USD · cash date: ${data.cash.as_of || 'Unavailable'}. ${data.cash.reason}`);
    (data.context.warnings || []).forEach(w => paragraph(provenance, w).classList.add('mo-private'));
    paragraph(provenance, data.target_note);
    const detail = drawer(root, `Instrument evidence · ${data.positions.length} positions`);
    paragraph(detail, 'Company profile dates are retrieval dates; effective sector dates are unavailable. Fund observations are dated issuer Fund tables. Geography and asset class are separate dimensions. Non-industry weights remain outside coverage.');
    const sourceLabels = { unavailable: 'Unavailable', issuer_fund_breakdown: 'Issuer fund breakdown', secondary_company_profile: 'Company profile' };
    table(detail, ['Instrument', 'ISIN / listing', 'Quote currency', 'Sector evidence', 'Observed / retrieved', 'Known weight (%)', 'Source', 'Limit'],
        data.positions.map(p => [stack(p.name, p.symbol), stack(p.isin || 'ISIN unavailable', p.symbol), p.quote_currency,
            sourceLabels[p.classification.source_kind] || p.classification.source_kind?.replaceAll('_', ' '),
            stack(p.classification.as_of || 'Effective date unavailable', p.classification.retrieved_at || 'Retrieval unavailable'),
            show(p.classification.known_fraction * 100), sourceLink(p.classification.source_url),
            `${p.valuation_reason || ''} ${p.classification.reason}. ${p.mapping_reason}. ${p.geography.reason}`]),
        { numeric: [5], privateRows: true });
    document.getElementById('gapsCount').textContent = `${certain} certain deviations`;

    const candidates = document.getElementById('opportunitiesTable'); candidates.replaceChildren();
    const counts = data.candidate_summary;
    metrics(candidates, [['Examined', counts.total], ['Screened', counts.screened], ['Already held', counts.held], ['Unavailable / excluded', counts.unavailable]]);
    note(candidates, 'Historical screening supports manual review. A screened candidate is not a buy recommendation. Exploration runs even when sector gaps are indeterminate.');
    if (!counts.ranking_comparable) note(candidates, 'Ranking unavailable: candidates have different completed observation dates.', true);
    table(candidates, ['Rank', 'Candidate', 'Sector mandate', 'Screening status', 'Historical score', 'Return (%)', 'Volatility (%/year)', 'USD history', 'Evidence'],
        [...data.candidates].sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity)).map(c => {
            const status = elem('div'); status.className = 'mo-stack';
            status.append(badge(({ screened: 'Screened', held: 'Already held', unavailable: 'Unavailable' })[c.status] || c.status?.replaceAll('_', ' '), c.status === 'screened' ? 'info' : 'neutral'));
            if (c.exclusions?.length) elem('small', c.exclusions.join('; '), status);
            return [c.rank ?? '—', stack(c.symbol, c.name), stack(c.intended_sector, `Reported: ${c.reported_sector || 'Unconfirmed'}`), status,
                show(c.metrics?.score), show(c.metrics?.return_pct), show(c.metrics?.annualized_volatility_pct),
                c.metrics ? stack(`${c.metrics.start} – ${c.metrics.end}`, `${c.metrics.sessions} sessions`) : 'Unavailable', sourceLink(c.price_source)];
        }), { numeric: [0, 4, 5, 6], privateRows: true });
    const method = drawer(candidates, 'Screening method and limitations');
    paragraph(method, 'Curated candidates are for manual review. ETF names describe a mandate, not verified composition. Constituent overlap and personal trading eligibility are unassessed. ISIN deduplication is incomplete when the provider omits ISIN.');
    paragraph(method, 'Score = clip(50 + 15 × mean(daily USD returns) / sample standard deviation × √historical sessions, 0, 100). Scores describe past price behavior. No valuation or diversification composite, forecast or probability is added.');
    document.getElementById('opportunitiesCount').textContent = `${counts.screened} screened candidates`;

    const reviews = document.getElementById('suggestedSales'); reviews.replaceChildren();
    note(reviews, data.automatic_sale_reason, true);
    const reviewDetail = drawer(reviews, `Review ${data.holding_reviews.length} holdings · reasons and sale limits`);
    table(reviewDetail, ['Holding', 'Weight (%)', 'Review evidence', 'Sale eligibility'], data.holding_reviews.map(r => {
        const evidence = elem('div');
        if (r.review_reasons.length) {
            const reasons = drawer(evidence, `${r.review_reasons.length} review flags`);
            r.review_reasons.forEach(reason => paragraph(reasons, reason));
        } else elem('span', 'No generic rule triggered', evidence);
        const sale = elem('div'); sale.append(badge('Unassessed'));
        const limits = drawer(sale, 'Required evidence'); paragraph(limits, r.sale_reason);
        return [stack(r.symbol, r.name), show(r.weight_pct), evidence, sale];
    }), { numeric: [1], privateRows: true });
    document.getElementById('salesCount').textContent = '0 automatic sales';
}

export function renderScenario(result, root) {
    root.replaceChildren();
    note(root, `Estimated cash: ${show(result.cash_before_usd)} → ${show(result.cash_after_usd)} USD. Total: ${show(result.before_total_usd)} → ${show(result.after_total_usd)} USD. Costs and slippage: ${show(result.costs_and_slippage_usd)} USD.`).classList.add('mo-private');
    table(root, ['Industry', 'Before lower / upper (%)', 'After lower / upper (%)', 'After assessment'],
        result.exposure_after.sector_assessment.map(s => {
            const before = result.exposure_before.sector_assessment.find(b => b.sector === s.sector);
            return [s.sector, `${show(before?.lower_pct)} / ${show(before?.upper_pct)}`, `${show(s.lower_pct)} / ${show(s.upper_pct)}`, badge(s.status)];
        }), { numeric: [1, 2], privateRows: true });
    const risk = result.historical_risk;
    note(root, 'Risk Score: unavailable. No validated robustness score is implemented here.');
    if (risk?.status === 'calculated') {
        paragraph(root, `Historical annualized volatility: ${show(risk.volatility_before_pct)}% → ${show(risk.volatility_after_pct)}%. ${risk.start} – ${risk.end}: ${risk.sessions} common daily intervals. ${risk.reason}`).classList.add('mo-private');
        risk.correlation_changes.forEach(c => paragraph(root, `${c.symbol}: historical correlation with the before portfolio = ${show(c.correlation_to_before, 3)}.`).classList.add('mo-private'));
    } else note(root, `Historical volatility unavailable. ${risk?.reason || 'History calculation was not requested.'} ${risk ? `${risk.histories_available ?? 0}/${risk.positions_required ?? 0} exact histories available.` : ''}`, true);
    result.limits.forEach(l => paragraph(root, l));
}

export function mountMarketOpportunities({ getContext, readTargets, invalidatePage }) {
    let revision = 0;
    let analysis = null;
    const config = document.getElementById('opportunityMethod').parentElement;
    const label = elem('label', 'Explore candidates: ', config); label.className = 'mo-selector';
    const selector = elem('select', undefined, label);
    selector.id = 'candidateSector'; selector.className = 'br-filter-select';
    ['all', ...sectors].forEach(s => { const o = elem('option', s === 'all' ? 'All industry ETFs' : s, selector); o.value = s; });
    selector.addEventListener('change', invalidatePage);
    const contextKey = () => JSON.stringify({ ...getContext(), targets: readTargets(), candidate_sector: selector.value });
    const params = () => {
        const c = getContext();
        if (!c.user || !c.source) throw new Error('Select an authenticated stock source before scanning.');
        return { source: c.source, ...(c.fileKey ? { file_key: c.fileKey } : {}), horizon: c.horizon,
            candidate_sector: selector.value, ...(readTargets() ? { sector_targets: readTargets() } : {}) };
    };
    function invalidate() { revision++; analysis = null; document.getElementById('btnScanOpportunities').disabled = false; }
    function clear(message) {
        for (const id of ['portfolioGaps', 'opportunitiesTable', 'suggestedSales', 'impactSimulator']) document.getElementById(id).textContent = message;
        for (const id of ['gapsCount', 'opportunitiesCount', 'salesCount']) document.getElementById(id).textContent = '—';
    }
    function scenarioForm(data) {
        const root = document.getElementById('impactSimulator'); root.replaceChildren();
        paragraph(root, 'Choose manual estimated USD amounts. Cash, positions and FX have different dates; this is decision support, not executable orders.');
        if (data.cash.value_usd === null) { note(root, `Funding simulation unavailable: ${data.cash.reason}`, true); return; }
        if (data.positions.some(p => p.value_usd === null)) { note(root, 'Scenario unavailable: some selected positions have no source valuation. The full portfolio cannot be valued; missing amounts are not assumed to be zero.', true); return; }
        const form = elem('form', undefined, root);
        const inputs = [];
        const rows = [];
        function input(side, id, name, maximum) {
            const amount = elem('input'); amount.type = 'number'; amount.min = '0'; amount.step = '0.01'; amount.value = '0';
            amount.setAttribute('aria-label', `${side} ${name} estimated amount in USD`);
            if (maximum !== undefined) amount.max = String(maximum);
            inputs.push({ side, id, amount });
            rows.push([side === 'sell' ? 'Reduce holding' : 'Add candidate', name, maximum === undefined ? 'Cash constrained' : show(maximum), amount]);
        }
        data.positions.filter(p => p.value_usd > 0).forEach(p => input('sell', p.id, p.symbol, p.value_usd));
        data.candidates.filter(c => c.status === 'screened').forEach(c => input('buy', c.id, c.symbol));
        table(form, ['Manual change', 'Instrument', 'Maximum snapshot value (USD)', 'Estimated amount (USD)'], rows);
        function setting(text, type, defaultValue) {
            const l = elem('label', text + ' ', form); l.style.display = 'block';
            const n = elem('input', undefined, l); n.type = type;
            if (type === 'number') { n.value = defaultValue; n.min = '0'; n.step = '0.01'; n.required = true; }
            return n;
        }
        const costs = setting('Total estimated fees and taxes (USD)', 'number', '0');
        const slippage = setting('Estimated slippage (%)', 'number', '0'); slippage.max = '10';
        const acknowledge = setting('I understand that saved cash and position values are dated estimates', 'checkbox'); acknowledge.required = true;
        const include = setting('Calculate full-portfolio historical volatility and candidate correlations (requires all exact histories)', 'checkbox');
        const button = elem('button', 'Calculate manual scenario', form); button.type = 'submit'; button.className = 'btn primary small';
        const output = elem('div', undefined, root); output.setAttribute('role', 'status');
        const formKey = contextKey(); let formRevision = revision;
        form.addEventListener('submit', async event => {
            event.preventDefault();
            if (formRevision !== revision || formKey !== contextKey() || !analysis) return;
            const trades = inputs.filter(i => i.amount.valueAsNumber > 0).map(i => ({ side: i.side, id: i.id, amount_usd: i.amount.valueAsNumber }));
            if (!trades.length) { output.textContent = 'Choose at least one manual change.'; return; }
            button.disabled = true; output.textContent = 'Checking the selected source and scenario…';
            const requestRevision = ++revision;
            const controls = [...form.querySelectorAll('input')];
            controls.forEach(n => { n.disabled = true; });
            try {
                const response = await window.safeFetch(window.getApiBase() + '/api/bourse/opportunities/scenario', {
                    method: 'POST', timeout: 160000, maxRetries: 0, headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...params(), scenario: { snapshot_id: data.snapshot_id, trades,
                        costs_usd: costs.valueAsNumber, slippage_pct: slippage.valueAsNumber,
                        acknowledge_dated_values: acknowledge.checked, include_history: include.checked } })
                });
                if (requestRevision !== revision || formKey !== contextKey()) return;
                const body = response.data;
                if (requestRevision !== revision || formKey !== contextKey()) return;
                if (!response.ok) throw new Error(response.error || body?.detail || 'Scenario calculation is unavailable.');
                const value = body.data || body;
                if (value.context?.user_id !== getContext().user || value.context?.source !== getContext().source || value.snapshot_id !== data.snapshot_id) throw new Error('Scenario context changed. Scan again.');
                renderScenario(value.scenario, output);
            } catch (error) {
                if (requestRevision === revision) output.textContent = error.message;
            } finally {
                controls.forEach(n => { n.disabled = false; });
                if (requestRevision === revision) { button.disabled = false; formRevision = revision; }
            }
        });
    }
    async function scan() {
        const requestRevision = ++revision; analysis = null;
        const btn = document.getElementById('btnScanOpportunities'); btn.disabled = true;
        clear('Checking source, instrument evidence and completed price histories…');
        let key;
        try {
            key = contextKey();
            const parameters = params(); const query = new URLSearchParams();
            Object.entries(parameters).forEach(([k, v]) => query.set(k, typeof v === 'object' ? JSON.stringify(v) : v));
            const response = await window.safeFetch(`${window.getApiBase()}/api/bourse/opportunities?${query}`, {
                timeout: 115000, maxRetries: 0, headers: getAuthHeaders()
            });
            if (requestRevision !== revision || key !== contextKey()) return;
            const body = response.data;
            if (requestRevision !== revision || key !== contextKey()) return;
            if (!response.ok) throw new Error(response.error || body?.detail || 'Selected-source analysis is unavailable.');
            const data = body.data || body;
            if (!matchesContext(data, getContext())) throw new Error('The response does not match the selected account and source. Scan again.');
            analysis = data; renderAnalysis(data); scenarioForm(data);
        } catch (error) {
            if (requestRevision === revision) clear(error.message);
        } finally {
            if (requestRevision === revision) btn.disabled = false;
        }
    }
    function exportResult() {
        if (!analysis || !matchesContext(analysis, getContext())) return;
        // Only the authorized aggregate audit controls; never private holdings or money.
        const report = { version: 2, generated_at: analysis.generated_at,
            source_matches_selection: true, position_count: analysis.positions.length,
            positions_with_isin: analysis.positions.filter(p => p.isin).length,
            positions_with_acquisition_date: analysis.positions.filter(p => p.acquisition_date).length,
            unvalued_positions: analysis.exposure.unvalued_positions,
            reporting_currency: analysis.context.reporting_currency, valuation_currency: analysis.context.valuation_currency,
            imported_at: analysis.context.imported_at, valuation_as_of: analysis.context.valuation_as_of,
            cash_as_of: analysis.cash.as_of, cash_record_available: analysis.cash.value_usd !== null,
            classification_coverage: analysis.exposure.coverage, unclassified_pct: analysis.exposure.unclassified_pct,
            candidate_counts: analysis.candidate_summary, automatic_sale_count: analysis.automatic_sales.length,
            conclusion: analysis.conclusion, method: 'Historical descriptive screening; no validated forward-return forecast' };
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob); const a = elem('a'); a.href = url;
        a.download = `market-opportunities-${analysis.generated_at.slice(0, 10)}.json`; a.click(); URL.revokeObjectURL(url);
    }
    document.getElementById('btnScanOpportunities').addEventListener('click', scan);
    document.getElementById('btnExportTextOpps').addEventListener('click', exportResult);
    return { invalidate, scan };
}
