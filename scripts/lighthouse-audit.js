/**
 * Lighthouse Audit Script with Authentication
 * Runs Lighthouse on authenticated SmartFolio pages
 *
 * Usage: node scripts/lighthouse-audit.js
 * Requires: npm install puppeteer lighthouse
 */

import puppeteer from 'puppeteer';
import lighthouse from 'lighthouse';
import fs from 'fs';

const BASE_URL = process.env.BASE_URL || 'http://localhost:8080';
const LOGIN_CREDENTIALS = {
    username: process.env.LH_USER || 'jack',
    password: process.env.LH_PASS || 'CHANGE_ME'  // Set via: $env:LH_PASS="yourpassword"
};

// Pages to audit (most important ones)
const PAGES_TO_AUDIT = [
    { name: 'dashboard', path: '/static/dashboard.html' },
    { name: 'analytics', path: '/static/analytics-unified.html' },
    { name: 'risk-dashboard', path: '/static/risk-dashboard.html' },
    { name: 'saxo-dashboard', path: '/static/saxo-dashboard.html' },
    { name: 'settings', path: '/static/settings.html' },
];

async function login(page) {
    console.log('Logging in...');
    await page.goto(`${BASE_URL}/static/login.html`, { waitUntil: 'networkidle0', timeout: 60000 });

    // Check if already logged in (redirected to dashboard)
    if (page.url().includes('dashboard')) {
        console.log('Already logged in');
        return;
    }

    console.log('Filling form...');
    await page.type('#username', LOGIN_CREDENTIALS.username);
    await page.type('#password', LOGIN_CREDENTIALS.password);

    console.log('Clicking login button...');
    await page.click('#login-button');

    // Wait a bit for any error message or redirect
    await new Promise(r => setTimeout(r, 3000));

    // Check for error message
    const errorText = await page.evaluate(() => {
        const errorEl = document.querySelector('#error-text');
        return errorEl ? errorEl.textContent : null;
    });

    if (errorText) {
        console.error('Login error:', errorText);
        throw new Error('Login failed: ' + errorText);
    }

    // Check current URL
    const currentUrl = page.url();
    console.log('Current URL after login:', currentUrl);

    if (currentUrl.includes('login')) {
        // Still on login page - try waiting for navigation
        console.log('Still on login page, waiting for redirect...');
        try {
            await page.waitForNavigation({ timeout: 10000 });
        } catch {
            console.log('No navigation detected');
        }
    }

    console.log('Final URL:', page.url());
}

async function runLighthouse(browser, url, pageName) {
    console.log(`\nAuditing: ${pageName} (${url})`);

    const { lhr } = await lighthouse(url, {
        port: new URL(browser.wsEndpoint()).port,
        output: 'json',
        logLevel: 'error',
        onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
    });

    // Collect failed audits for debugging
    const failedAudits = [];
    const bpCategory = lhr.categories['best-practices'];
    if (bpCategory) {
        for (const ref of bpCategory.auditRefs || []) {
            const audit = lhr.audits[ref.id];
            if (audit && audit.score !== null && audit.score < 1) {
                failedAudits.push({
                    id: ref.id,
                    title: audit.title,
                    score: audit.score,
                    description: audit.description?.substring(0, 100)
                });
            }
        }
    }

    // Collect console errors
    const consoleErrors = [];
    const errorsAudit = lhr.audits['errors-in-console'];
    if (errorsAudit && errorsAudit.details?.items) {
        for (const item of errorsAudit.details.items) {
            consoleErrors.push(item.description || item.source || 'Unknown error');
        }
    }

    // Collect accessibility failures
    const a11yFailures = [];
    const a11yCategory = lhr.categories['accessibility'];
    if (a11yCategory) {
        for (const ref of a11yCategory.auditRefs || []) {
            const audit = lhr.audits[ref.id];
            if (audit && audit.score === 0) {
                a11yFailures.push({
                    id: ref.id,
                    title: audit.title,
                    items: audit.details?.items?.length || 0
                });
            }
        }
    }

    return {
        name: pageName,
        url: url,
        scores: {
            performance: Math.round((lhr.categories.performance?.score || 0) * 100),
            accessibility: Math.round((lhr.categories.accessibility?.score || 0) * 100),
            bestPractices: Math.round((lhr.categories['best-practices']?.score || 0) * 100),
            seo: Math.round((lhr.categories.seo?.score || 0) * 100),
        },
        metrics: {
            fcp: lhr.audits['first-contentful-paint']?.displayValue || 'N/A',
            lcp: lhr.audits['largest-contentful-paint']?.displayValue || 'N/A',
            tbt: lhr.audits['total-blocking-time']?.displayValue || 'N/A',
            cls: lhr.audits['cumulative-layout-shift']?.displayValue || 'N/A',
        },
        failedBestPractices: failedAudits.filter(a => a.score === 0),
        consoleErrors: consoleErrors.slice(0, 5), // Limit to first 5
        a11yFailures: a11yFailures,
    };
}

function printResults(results) {
    console.log('\n' + '='.repeat(80));
    console.log('LIGHTHOUSE AUDIT RESULTS - SmartFolio');
    console.log('='.repeat(80));

    // Targets from DESIGN_SYSTEM.md
    const targets = { performance: 85, accessibility: 90, bestPractices: 90, seo: 80 };

    console.log('\n| Page | Perf | A11y | BP | SEO | Status |');
    console.log('|------|------|------|-----|-----|--------|');

    let allPass = true;
    for (const r of results) {
        const perfOk = r.scores.performance >= targets.performance;
        const a11yOk = r.scores.accessibility >= targets.accessibility;
        const bpOk = r.scores.bestPractices >= targets.bestPractices;
        const seoOk = r.scores.seo >= targets.seo;
        const status = perfOk && a11yOk && bpOk && seoOk ? 'PASS' : 'FAIL';
        if (status === 'FAIL') allPass = false;

        console.log(`| ${r.name.padEnd(14)} | ${r.scores.performance.toString().padStart(3)} | ${r.scores.accessibility.toString().padStart(3)} | ${r.scores.bestPractices.toString().padStart(3)} | ${r.scores.seo.toString().padStart(3)} | ${status} |`);
    }

    console.log('\nTargets: Perf >= 85, A11y >= 90, BP >= 90, SEO >= 80');
    console.log('\n' + (allPass ? 'ALL PAGES PASS' : 'SOME PAGES NEED IMPROVEMENT'));

    // Detailed metrics
    console.log('\n--- Performance Metrics ---');
    for (const r of results) {
        console.log(`${r.name}: FCP=${r.metrics.fcp}, LCP=${r.metrics.lcp}, TBT=${r.metrics.tbt}, CLS=${r.metrics.cls}`);
    }

    // Failed Best Practices audits
    console.log('\n--- Failed Best Practices Audits ---');
    for (const r of results) {
        if (r.failedBestPractices && r.failedBestPractices.length > 0) {
            console.log(`${r.name}:`);
            for (const audit of r.failedBestPractices) {
                console.log(`  - ${audit.title}`);
            }
        }
    }

    // Console errors
    console.log('\n--- Console Errors ---');
    for (const r of results) {
        if (r.consoleErrors && r.consoleErrors.length > 0) {
            console.log(`${r.name}:`);
            for (const err of r.consoleErrors) {
                console.log(`  - ${err.substring(0, 150)}`);
            }
        }
    }

    // Accessibility failures
    console.log('\n--- Accessibility Failures ---');
    for (const r of results) {
        if (r.a11yFailures && r.a11yFailures.length > 0) {
            console.log(`${r.name}:`);
            for (const audit of r.a11yFailures) {
                console.log(`  - ${audit.title} (${audit.items} elements)`);
            }
        }
    }
}

async function checkServer() {
    try {
        const response = await fetch(`${BASE_URL}/health`);
        return response.ok;
    } catch {
        return false;
    }
}

async function main() {
    console.log('Starting Lighthouse audit...');
    console.log(`Credentials: ${LOGIN_CREDENTIALS.username} / ${LOGIN_CREDENTIALS.password === 'CHANGE_ME' ? '(NOT SET!)' : '***'}`);
    console.log('');

    // Check server
    const serverUp = await checkServer();
    if (!serverUp) {
        console.error('ERROR: Server not running at ' + BASE_URL);
        console.error('Start it with: python -m uvicorn api.main:app --port 8080');
        process.exit(1);
    }
    console.log('Server is running.\n');

    const DEBUG = process.env.DEBUG === '1';
    const browser = await puppeteer.launch({
        headless: DEBUG ? false : 'new',
        args: ['--remote-debugging-port=9222'],
        slowMo: DEBUG ? 100 : 0,  // Slow down for debugging
    });
    if (DEBUG) console.log('DEBUG MODE: Browser visible');

    try {
        const page = await browser.newPage();
        await login(page);

        const results = [];
        for (const pageInfo of PAGES_TO_AUDIT) {
            const url = `${BASE_URL}${pageInfo.path}`;
            const result = await runLighthouse(browser, url, pageInfo.name);
            results.push(result);
        }

        printResults(results);

        // Save JSON report
        const reportPath = 'lighthouse-report.json';
        fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
        console.log(`\nDetailed report saved to: ${reportPath}`);

    } finally {
        await browser.close();
    }
}

main().catch(console.error);
