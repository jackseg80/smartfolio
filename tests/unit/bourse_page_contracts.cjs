const {JSDOM} = require('jsdom');
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

(async () => {
    const html = fs.readFileSync('static/bourse-recommendations.html', 'utf8');
    const dom = new JSDOM(html, {url:'http://localhost/static/bourse-recommendations.html', runScripts:'outside-only'});
    const w = dom.window;
    const run = code => vm.runInContext(code, dom.getInternalVMContext());
    await new Promise(resolve => w.addEventListener('load', resolve));
    w.debugLogger = {debug(){}, warn(){}, error(){}};
    w.getApiBase = () => 'http://localhost';
    w.Toast = {warning(){}, error(){}};
    w.localStorage.setItem('activeUser', 'alice');
    for (const script of w.document.querySelectorAll('script:not([src]):not([type="module"])')) run(script.textContent);
    w.saxoSourceType = 'csv';
    run("currentFileKey = 'selected.csv'");
    w.setupRecommendationsListeners();
    w.setupOpportunitiesListeners();
    const get = id => w.document.getElementById(id);
    const custom = get('useCustomSectorTargets');
    custom.checked = true;
    custom.dispatchEvent(new w.Event('change'));
    assert.equal(get('sectorTargetInputs').disabled, false);
    assert.equal(Object.values(w.readSectorTargets()).reduce((a,b) => a+b, 0).toFixed(2), '100.00');
    let requestURL, release;
    w.safeFetch = async url => {
        requestURL = url;
        return {ok:true, data:{gaps:[], opportunities:[{symbol:'TEST', name:'<img src=x onerror=bad()>', sector:'Technology', score:70, type:'ETF', confidence:.8, capital_needed:1000, score_components_available:['momentum'], diversification_score:null}], suggested_sales:[], impact:{}, target_source:'user', classification_coverage:.7, horizon_details:{label:'6-12 Months', signal_sessions:189, history_calendar_days:365}}};
    };
    await w.loadMarketOpportunities();
    const url = new URL(requestURL);
    assert.equal(url.searchParams.get('file_key'), 'selected.csv');
    assert.equal(JSON.parse(url.searchParams.get('sector_targets')).Technology > 0, true);
    assert.match(get('opportunityMethod').textContent, /6-12 Months.*189.*365/);
    assert.match(get('opportunityMethod').textContent, /70%/);
    assert.match(get('opportunitiesTable').textContent, /REVIEW/);
    assert.match(get('opportunitiesTable').textContent, /80%/);
    assert.match(get('opportunitiesTable').textContent, /not cumulative/);
    assert.equal(get('opportunitiesTable').querySelector('img'), null);
    // Switching the holding horizon must clear previously computed results.
    w.document.querySelector('[data-opp-horizon="short"]').click();
    assert.match(get('opportunitiesTable').textContent, /Run a new scan/);
    assert.equal(run('lastOpportunitiesData'), null);
    // An old response cannot overwrite the new selection.
    w.safeFetch = () => new Promise(resolve => {release=resolve;});
    const pending = w.loadMarketOpportunities();
    w.document.querySelector('[data-opp-horizon="long"]').click();
    release({ok:true, data:{opportunities:[{symbol:'STALE'}]}});
    await pending;
    assert.doesNotMatch(get('opportunitiesTable').textContent, /STALE/);
    assert.equal(get('btnScanOpportunities').disabled, false);
    // Reject invalid percentages without querying the API.
    let calls = 0;
    w.safeFetch = async () => {calls++; throw new Error('must not query');};
    w.document.querySelector('[data-sector-target]').value = '101';
    await w.loadMarketOpportunities();
    assert.equal(calls, 0);
    assert.match(get('portfolioGaps').textContent, /between 0 and 100/);
    assert.equal(get('btnScanOpportunities').disabled, false);
    dom.window.close();
    console.log('Recommendations DOM contracts passed');
})().catch(error => {console.error(error);process.exitCode=1;});
