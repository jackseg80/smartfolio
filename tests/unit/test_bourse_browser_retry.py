"""Execute browser JavaScript, including transient failures and stale source changes."""
import shutil
import subprocess
from pathlib import Path
import pytest


def test_retry_sections_preserve_success_and_invalidate_old_requests():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser regression tests')
    root = Path(__file__).resolve().parents[2]
    script = r"""
    const fs = require('node:fs');
    const vm = require('node:vm');
    const assert = require('node:assert/strict');
    vm.runInThisContext(fs.readFileSync('static/core/retryable-sections.js', 'utf8'));
    // Syntax-check every inline script of both changed pages as well.
    for (const file of ['bourse-analytics.html', 'bourse-recommendations.html']) {
        const html = fs.readFileSync('static/' + file, 'utf8');
        for (const [,attrs,body] of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)) {
            if (!attrs.includes('src=') && !attrs.includes('type="module"')) new vm.Script(body, {filename:file});
        }
    }
    (async () => {
        const sections = createRetryableSections();
        let goodCalls = 0, badCalls = 0;
        const good = () => {goodCalls++;};
        const bad = () => {badCalls++; return badCalls > 1;};
        assert.deepEqual(await Promise.all([sections.run('good', good), sections.run('bad', bad)]), [true, false]);
        assert.deepEqual(await Promise.all([sections.run('good', good), sections.run('bad', bad)]), [true, true]);
        assert.equal(goodCalls, 1);
        assert.equal(badCalls, 2);
        let requests = 0, resolve;
        const loader = async () => { requests++; await new Promise(r => {resolve=r;}); };
        const first = sections.run('pending', loader);
        const duplicate = sections.run('pending', loader);
        await Promise.resolve();
        assert.equal(requests, 1);
        resolve();
        assert.deepEqual(await Promise.all([first, duplicate]), [true, true]);
        let release, mayWrite;
        const obsolete = sections.run('old', async current => {await new Promise(r => {release=r;}); mayWrite=current();});
        await Promise.resolve();
        sections.reset(); release();
        assert.equal(await obsolete, false);
        assert.equal(mayWrite, false);
        let newCalls = 0;
        assert.equal(await sections.run('old', () => {newCalls++;}), true);
        assert.equal(newCalls, 1);
        assert.equal(await sections.run('throws', () => {throw new Error('temporary');}), false);
        assert.equal(await sections.run('throws', () => true), true);
    })().catch(error => { console.error(error); process.exitCode=1; });
    """
    result = subprocess.run([node, '-e', script], cwd=root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_recommendations_dom_contracts():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser regression tests')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, '--experimental-vm-modules', 'tests/unit/bourse_page_contracts.cjs'], cwd=root,
                            capture_output=True, text=True, timeout=30)
    if "Cannot find module 'jsdom'" in result.stderr:
        pytest.skip('Install npm dev dependencies to execute DOM contracts')
    assert result.returncode == 0, result.stderr
