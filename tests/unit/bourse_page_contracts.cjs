const {JSDOM} = require('jsdom');
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

(async () => {
    const html = fs.readFileSync('static/bourse-recommendations.html', 'utf8');
    const dom = new JSDOM(html, {url:'http://localhost/static/bourse-recommendations.html', runScripts:'outside-only'});
    const w = dom.window;
    const context = dom.getInternalVMContext();
    const auth = new vm.SyntheticModule(['getAuthHeaders'], function () {
        this.setExport('getAuthHeaders', () => ({'X-User': 'alice'}));
    }, {context});
    await auth.link(() => {throw new Error('Unexpected authentication import');});
    await auth.evaluate();
    const component = new vm.SourceTextModule(fs.readFileSync('static/components/market-opportunities.js', 'utf8'), {context});
    await component.link(specifier => {
        assert.equal(specifier, '../core/auth-guard.js');
        return auth;
    });
    await component.evaluate();
    const run = code => vm.runInContext(code, context, {importModuleDynamically: async specifier => {
        assert.equal(specifier, './components/market-opportunities.js');
        return component;
    }});
    if (w.document.readyState !== 'complete') await new Promise(resolve => w.addEventListener('load', resolve));
    w.debugLogger = {debug(){}, warn(){}, error(){}};
    w.getApiBase = () => 'http://localhost';
    w.Toast = {warning(){}, error(){}};
    w.localStorage.setItem('activeUser', 'alice');
    for (const script of w.document.querySelectorAll('script:not([src]):not([type="module"])')) run(script.textContent);
    w.saxoSourceType = 'csv';
    run("currentFileKey = 'selected.csv'");
    w.setupRecommendationsListeners();
    await w.setupOpportunitiesListeners();
    const get = id => w.document.getElementById(id);
    const custom = get('useCustomSectorTargets');
    custom.checked = true;
    custom.dispatchEvent(new w.Event('change'));
    assert.equal(get('sectorTargetInputs').disabled, false);
    assert.equal(Object.values(w.readSectorTargets()).reduce((a,b) => a+b, 0).toFixed(2), '100.00');
    let requestURL, release;
    // Synthetic fixture: exercises the real component and page integration, never real positions.
    const fixture = {version:2, context:{user_id:'alice', source:'saxobank_csv', file_key:'selected.csv', valuation_currency:'USD', warnings:[]},
        horizon_details:{label:'6-12 Months',signal_sessions:189}, target_note:'Personal targets',
        cash:{value_usd:null,reason:'Fixture cash unavailable'},
        exposure:{coverage:.7,unclassified_pct:30,unvalued_positions:0,reason:'Retain unknown composition',sector_assessment:[]},
        positions:[], holding_reviews:[], automatic_sales:[], automatic_sale_reason:'Verified sale evidence unavailable',
        conclusion:'Zero results does not establish that this portfolio is balanced.',
        candidate_summary:{total:1,screened:1,held:0,unavailable:0,ranking_comparable:true},
        candidates:[{symbol:'TEST',name:'<img src=x onerror=bad()>',intended_sector:'Technology',status:'screened',exclusions:[],
            metrics:{score:70,return_pct:5,annualized_volatility_pct:18,start:'2025-01-01',end:'2025-09-30',sessions:189}}]};
    w.safeFetch = async url => {requestURL=url; return {ok:true,data:{data:fixture}};};
    await w.marketOpportunities.scan();
    const url = new URL(requestURL);
    assert.equal(url.searchParams.get('file_key'), 'selected.csv');
    assert.equal(url.searchParams.get('source'), 'saxobank_csv');
    assert.equal(JSON.parse(url.searchParams.get('sector_targets')).Technology > 0, true);
    assert.match(get('opportunityMethod').textContent, /6-12 Months.*189/);
    assert.match(get('opportunityMethod').textContent, /No validated forward-return forecast/);
    assert.match(get('portfolioGaps').textContent, /70\.00%/);
    assert.equal(get('opportunitiesTable').querySelectorAll('table tbody tr').length, 1);
    assert.match(get('opportunitiesTable').textContent, /Screened/);
    assert.match(get('opportunitiesTable').textContent, /past price behavior/);
    assert.equal(get('opportunitiesTable').querySelector('img'), null);
    // Switching the holding horizon must clear previously computed results.
    w.document.querySelector('[data-opp-horizon="short"]').click();
    assert.match(get('opportunitiesTable').textContent, /Run a new scan/);
    assert.equal(get('opportunitiesTable').querySelector('table'), null);
    // An old response cannot overwrite the new selection.
    w.safeFetch = () => new Promise(resolve => {release=resolve;});
    const pending = w.marketOpportunities.scan();
    w.document.querySelector('[data-opp-horizon="long"]').click();
    release({ok:true, data:{data:{...fixture,candidates:[{symbol:'STALE'}]}}});
    await pending;
    assert.doesNotMatch(get('opportunitiesTable').textContent, /STALE/);
    assert.equal(get('btnScanOpportunities').disabled, false);
    // Reject invalid percentages without querying the API.
    let calls = 0;
    w.safeFetch = async () => {calls++; throw new Error('must not query');};
    w.document.querySelector('[data-sector-target]').value = '101';
    await w.marketOpportunities.scan();
    assert.equal(calls, 0);
    assert.match(get('portfolioGaps').textContent, /between 0 and 100/);
    assert.equal(get('btnScanOpportunities').disabled, false);
    dom.window.close();
    console.log('Recommendations DOM contracts passed');
})().catch(error => {console.error(error);process.exitCode=1;});
