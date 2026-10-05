// Contrat navigateur isolé : aucun serveur ni compte réel, ressources locales et API de test.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('@playwright/test');
const root = path.resolve(__dirname, '../..');
const output = path.join(root, 'outputs/saxo-valuation');
fs.mkdirSync(output, { recursive: true });

function fixture(mode, partial = false) {
  const amount = mode === 'export' ? 100 : 220;
  const position = { symbol: 'AAPL:xnas', name: 'Apple', instrument: 'Apple', asset_class: 'Stock',
    quantity: 2, currency: 'USD', market_value_display: partial ? null : amount,
    market_value_usd: mode === 'current' && !partial ? amount : null,
    market_value: 100, quote_at: '2026-10-05T15:30:00Z', valuation_status: partial ? 'export_fallback' : mode };
  return { portfolio_id: 'test', file_key: 'old.csv', source: 'csv', mode, currency: mode === 'export' ? 'EUR' : 'USD',
    export_date: '2021-01-18', oldest_quote_at: '2026-10-05T15:30:00Z', warnings: partial ? ['Valuation is partial.'] : [],
    comparison: null, positions: [position], totalValueIncludesCash: true,
    cash: { value_display: 0, included: false }, coverage: { positions: 1, updated: partial ? 0 : 1, partial },
    summary: { total_value: partial ? 0 : amount, total_value_usd: mode === 'current' ? amount : null,
      total_positions: 1, asset_allocation: { Stock: 100 }, currency_exposure: { USD: 100 }, top_holdings: [position] } };
}

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
    page.setDefaultTimeout(15000);
    const errors = [];
    page.on('console', msg => { if (msg.type() === 'error') console.error('Browser console:', msg.text()); });
    page.on('pageerror', err => { errors.push(err.message); console.error('Page error:', err.message); });
    let partial = false;
    const requests = [];
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/saxo-dashboard.html') {
        let html = fs.readFileSync(path.join(root, 'static/saxo-dashboard.html'), 'utf8');
        html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, script => script.includes('let currentPortfolioData') ? script : '');
        html = html.replace('<head>', `<head><style>view-toggle { display: inline-block; width: 160px; }</style><script>
          localStorage.setItem('activeUser', 'alice'); localStorage.setItem('bourseSource', 'saxo:old.csv');
          window.getApiBase = () => ''; window.globalConfig = { getApiUrl: path => path };
          window.debugLogger = { debug(){}, info(){}, warn(){}, error(...args){ console.error(...args); } };
          window.Chart = class { constructor(){ this.data = {}; } destroy(){} update(){} };
          window.wealthContextBar = { setContext(){}, getContext: () => ({ bourse: 'saxo:old.csv' }) };
        </script>`);
        return route.fulfill({ contentType: 'text/html', body: html });
      }
      if (url.pathname === '/core/auth-guard.js') {
        return route.fulfill({ contentType: 'text/javascript', body: `export function getAuthHeaders(){ return {'X-User':'alice'}; }` });
      }
      if (url.pathname === '/modules/export-button.js') {
        return route.fulfill({ contentType: 'text/javascript', body: 'export function renderExportButton(){}' });
      }
      if (url.pathname === '/api/users/sources') {
        return route.fulfill({ json: { sources: [{ key: 'old.csv', file_path: 'old.csv' }] } });
      }
      if (url.pathname === '/api/saxo/valuation') {
        requests.push(Object.fromEntries(url.searchParams));
        return route.fulfill({ json: { ok: true, data: fixture(url.searchParams.get('mode'), partial) } });
      }
      const target = path.resolve(root, 'static', url.pathname.replace(/^\/(?:static\/)?/, ''));
      if (target.startsWith(path.join(root, 'static') + path.sep) && fs.existsSync(target) && fs.statSync(target).isFile()) {
        const ext = path.extname(target);
        return route.fulfill({ contentType: ext === '.js' ? 'text/javascript' : ext === '.css' ? 'text/css' : 'image/svg+xml', body: fs.readFileSync(target) });
      }
      return route.fulfill({ status: 404, body: '' });
    });
    await page.goto('http://smartfolio.test/saxo-dashboard.html');
    await page.waitForFunction(() => document.getElementById('portfolioSummary').textContent.includes('$220.00'));
    await page.waitForFunction(() => document.getElementById('dashboardContent').style.display === 'block');
    assert.equal(await page.locator('#portfolioSummary').isVisible(), true);
    await page.locator('#valuationMode').selectOption('export');
    await page.locator('#portfolioSummary').getByText('€100.00', { exact: true }).waitFor();
    assert.match(await page.locator('#valuationStatus').innerText(), /Original currency: EUR/);
    await page.locator('#positionsTab').count(); // La navigation se fait par les boutons de la page.
    await page.evaluate(() => switchTab('positions'));
    assert.match(await page.locator('#allPositions').innerText(), /€100\.00/);
    assert.doesNotMatch(await page.locator('#allPositions').innerText(), /\$100\.00/);
    await page.screenshot({ path: path.join(output, 'historical-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.locator('#valuationMode').isVisible(), true);
    await page.screenshot({ path: path.join(output, 'historical-mobile.png'), fullPage: true });
    partial = true;
    await page.locator('#valuationMode').selectOption('current');
    await page.locator('#valuationStatus').getByText('Valuation is partial.', { exact: false }).waitFor();
    assert.match(await page.locator('#allPositions').innerText(), /Unavailable/);
    assert.equal(await page.locator('#valuationStatus').getAttribute('data-partial'), 'true');
    await page.locator('#btnForceRefresh').click();
    await page.waitForFunction(() => document.getElementById('dashboardContent').style.display === 'block');
    assert.equal(requests.at(-1).force, 'true');
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser-contract.json'), JSON.stringify({ requests, errors, checks: 'current/export, original currency, missing value, refresh, desktop/mobile' }, null, 2));
    console.log('PASS: browser valuation contract (current, export, partial, refresh, desktop/mobile)');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
