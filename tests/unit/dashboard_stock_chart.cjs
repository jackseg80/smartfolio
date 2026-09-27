const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');

const summaryCode = fs.readFileSync('static/modules/wealth-saxo-summary.js', 'utf8');
const controller = fs.readFileSync('static/modules/dashboard-main-controller.js', 'utf8');
const html = fs.readFileSync('static/dashboard.html', 'utf8');

async function setup(source = 'saxo:selected') {
    const dom = new JSDOM(html, { url: 'http://localhost/static/dashboard.html', runScripts: 'outside-only' });
    const w = dom.window;
    const context = dom.getInternalVMContext();
    w.console = { debug() {}, log() {}, warn() {}, error() {}, assert: assert.ok };
    w.debugLogger = w.console;
    w.localStorage.setItem('activeUser', 'alice');
    w.wealthContextBar = { getContext: () => ({ bourse: source }) };
    w.availableSources = [{ key: 'selected', file_path: 'data/users/alice/saxobank/data/selected file.csv' }];
    w._availableSourcesUser = 'alice';
    const calls = [];
    let payload = { positions: [
        { tags: ['asset_class:EQUITY'], market_value: 60 },
        { tags: ['asset_class:ETF'], market_value: 30 },
    ], asof: '2026-09-27T09:00:00Z' };
    let cash = 10;
    const safeFetch = async (url, options) => {
        calls.push({ url, options });
        return { ok: true, data: url.includes('/cash') ? { cash_amount: cash } : payload };
    };
    w.fetch = async (url, options) => {
        calls.push({ url, options });
        assert.equal(url, '/api/sources/v2/bourse/balances');
        return { ok: true, json: async () => payload };
    };
    w.HTMLCanvasElement.prototype.getContext = function () { return { canvas: this }; };
    w.Chart = class {
        constructor(ctx, config) { this.canvas = ctx.canvas; this.config = config; }
        destroy() { this.destroyed = true; }
    };
    const dependency = new vm.SyntheticModule(['safeFetch', 'formatUSD'], function () {
        this.setExport('safeFetch', safeFetch);
        this.setExport('formatUSD', value => `$${value.toFixed(2)}`);
    }, { context });
    const summary = new vm.SourceTextModule(summaryCode, { context });
    await summary.link(() => dependency);
    await summary.evaluate();
    const chartCode = controller.slice(controller.indexOf('async function updateSaxoChart('), controller.indexOf('// Create or update Wealth chart'));
    const refreshCode = controller.slice(controller.indexOf('async function refreshSaxoTile('), controller.indexOf('async function refreshPatrimoineTile('));
    const tile = new vm.SourceTextModule(`let isRefreshingSaxo = false; const PORTFOLIO_COLORS = ['red','blue','green']; const formatUSD = v => '$' + v; ${chartCode}\n${refreshCode}\nexport { updateSaxoChart, refreshSaxoTile };`, {
        context, importModuleDynamically: async () => summary,
    });
    await tile.link(() => dependency);
    await tile.evaluate();
    return { w, calls, summary: summary.namespace, tile: tile.namespace,
        setPayload: value => { payload = value; }, setCash: value => { cash = value; },
        close: () => w.close() };
}

function categories(chart) {
    return Object.fromEntries(chart.config.data.labels.map((label, i) => [label, chart.config.data.datasets[0].data[i]]));
}

test('selected CSV: tile total and chart share positions and cash, with no duplicate request', async () => {
    const env = await setup();
    try {
        await env.tile.refreshSaxoTile();
        assert.equal(env.w.document.getElementById('saxo-total-value').textContent, '$100.00');
        assert.deepEqual(categories(env.w.saxoChart), { EQUITY: 60, ETF: 30, Cash: 10 });
        assert.equal(env.calls.length, 2);
        for (const call of env.calls) {
            assert.equal(new URL(call.url, 'http://localhost').searchParams.get('file_key'), 'selected file.csv');
            assert.equal(call.options.headers['X-User'], 'alice');
        }
        await env.tile.refreshSaxoTile();
        assert.equal(env.calls.length, 2, 'cached summary still contains chart data');
    } finally { env.close(); }
});

test('API positions use USD valuation and direct asset classes', async () => {
    const env = await setup('api:saxobank_api');
    try {
        env.setPayload({ ok: true, data: { positions: [
            { asset_class: 'equity', market_value: 999, market_value_usd: 60 },
            { asset_class: 'etf', market_value_usd: 30 },
        ], total_value: 100, cash_balance: 10 } });
        await env.tile.refreshSaxoTile();
        assert.deepEqual(categories(env.w.saxoChart), { EQUITY: 60, ETF: 30, Cash: 10 });
        assert.equal(env.calls.length, 1);
    } finally { env.close(); }
});

test('manual source keeps asset classes and balance values', async () => {
    const env = await setup('manual_bourse');
    try {
        env.setPayload({ ok: true, data: { items: [
            { symbol: 'A', asset_class: 'equity', value_usd: 60 },
            { symbol: 'B', asset_class: 'etf', value_usd: 30 },
            { symbol: 'USD', asset_class: 'cash', value_usd: 10 },
        ] } });
        await env.tile.refreshSaxoTile();
        assert.deepEqual(categories(env.w.saxoChart), { EQUITY: 60, ETF: 30, CASH: 10 });
        assert.equal(env.calls.length, 1);
    } finally { env.close(); }
});

test('cash-only CSV remains visible', async () => {
    const env = await setup();
    try {
        env.setPayload({ positions: [] });
        await env.tile.refreshSaxoTile();
        assert.deepEqual(categories(env.w.saxoChart), { Cash: 10 });
        assert.equal(env.w.document.getElementById('saxo-total-value').textContent, '$10.00');
    } finally { env.close(); }
});

test('empty chart releases old chart and recovers its canvas and accessible description', async () => {
    const env = await setup();
    try {
        await env.tile.updateSaxoChart([{ market_value: 20, asset_class: 'ETF' }], 0);
        const previous = env.w.saxoChart;
        await env.tile.updateSaxoChart([], 0);
        assert.equal(previous.destroyed, true);
        assert.equal(env.w.saxoChart, null);
        assert.match(env.w.document.getElementById('saxo-chart').textContent, /No data/);
        await env.tile.updateSaxoChart([], 25);
        assert.deepEqual(categories(env.w.saxoChart), { Cash: 25 });
        assert.match(env.w.document.getElementById('saxo-chart-desc').textContent, /Cash: 100.0%/);
    } finally { env.close(); }
});

test('old summary cache without positions is ignored', async () => {
    const env = await setup();
    try {
        env.w.localStorage.setItem('saxo_summary_alice_saxo:selected', JSON.stringify({
            source: 'saxo:selected', timestamp: Date.now(), summary: { total_value: 999, positions_count: 2, isEmpty: false },
        }));
        await env.tile.refreshSaxoTile();
        assert.equal(env.calls.length, 2);
        assert.deepEqual(categories(env.w.saxoChart), { EQUITY: 60, ETF: 30, Cash: 10 });
    } finally { env.close(); }
});

test('dashboard no longer loads or displays Morning Brief', () => {
    assert.doesNotMatch(html, /morning-brief/i);
});
