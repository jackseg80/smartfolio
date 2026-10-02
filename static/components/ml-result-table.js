// Affichage compact, provenance détaillée à la demande; zéro reste une valeur.
export const isAvailable = result => result?.availability === 'Available' && result.value != null;
export function summarizeResults(results = []) {
    const forecasts = results.filter(r => r.nature === 'forecast');
    const available = forecasts.filter(isAvailable);
    const assets = [...new Set(forecasts.map(r => r.asset))];
    return {requested: forecasts.length, available: available.length, assets: assets.length,
        covered: new Set(available.map(r => r.asset)).size,
        diagnostics: results.filter(r => r.nature === 'diagnostic' && isAvailable(r)).length};
}
const el = (tag, text) => {const node=document.createElement(tag); if(text!=null)node.textContent=text;return node;};
export function renderAssetResults(root, results = []) {
    const grouped = new Map();
    for (const result of results) {
        if (!['future_realized_volatility','economic_rule_regime'].includes(result.target)) continue;
        if(!grouped.has(result.asset))grouped.set(result.asset,[]);
        grouped.get(result.asset).push(result);
    }
    const wrap=el('div');wrap.style.overflowX='auto';
    const table=el('table');table.className='data-table ml-asset-table';table.style.width='100%';
    const head=el('thead');const header=el('tr');
    for(const title of ['Asset','Regime diagnostic','7-day volatility','30-day volatility','Details'])header.append(el('th',title));
    head.append(header);table.append(head);
    const body=el('tbody');
    for(const [asset, rows] of grouped) {
        const row=el('tr');row.append(el('td',asset));
        const rule=rows.find(r=>r.target==='economic_rule_regime');
        row.append(el('td',isAvailable(rule)?String(rule.value):'Unavailable'));
        for(const days of [7,30]) {
            const r=rows.find(r=>r.nature==='forecast' && r.horizon===`${days}d`);
            row.append(el('td',isAvailable(r) && typeof r.value==='number' ? (r.value*100).toFixed(2)+'% / year' : 'Unavailable'));
        }
        const detailCell=el('td');const details=el('details');details.append(el('summary','Method, dates & limits'));
        for(const r of rows) {
            const section=el('div');section.style.cssText='min-width:15rem;max-width:30rem;padding:.5rem;overflow-wrap:anywhere';
            section.append(el('strong',(r.horizon || 'Regime')+' · '+r.availability));
            section.append(el('p',r.reason));
            section.append(el('p','Data: '+(r.data_as_of?.slice(0,10)||'Unavailable')+' · Target session: '+(r.target_date?.slice(0,10)||'Not applicable')));
            section.append(el('p','Method: '+(r.provenance?.method||'Unavailable')+' · Provider: '+(r.provenance?.provider||'Unavailable')+' · Validation: '+(r.validation?.state||'Unavailable')));
            if(r.uncertainty?.nominal_coverage===.9)section.append(el('p','Validated 90% interval: '+(r.uncertainty.lower_bound*100).toFixed(2)+'%–'+(r.uncertainty.upper_bound*100).toFixed(2)+'%'));
            else if(r.nature==='forecast' && isAvailable(r))section.append(el('p','Uncertainty interval not validated.'));
            const provenance=el('details');provenance.append(el('summary','Full provenance'));provenance.append(el('pre',JSON.stringify(r.provenance,null,2)));provenance.style.whiteSpace='pre-wrap';section.append(provenance);
            details.append(section);
        }
        detailCell.append(details);row.append(detailCell);body.append(row);
    }
    if(!grouped.size) {const row=el('tr');const cell=el('td','No asset results available.');cell.colSpan=5;row.append(cell);body.append(row);}
    table.append(body);wrap.append(table);root.append(wrap);
    return summarizeResults(results);
}
