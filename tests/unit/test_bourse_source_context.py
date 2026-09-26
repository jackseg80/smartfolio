"""Exercise browser source resolution without substituting source ids for filenames."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_stock_source_selection_supports_v2_and_legacy_files():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser source contract')
    root = Path(__file__).resolve().parents[2]
    script = r'''
        const fs = require('node:fs');
        const vm = require('node:vm');
        const assert = require('node:assert/strict');
        global.window = { getApiBase: () => 'http://localhost' };
        let fetches = 0;
        global.fetch = async () => {
            fetches++;
            return {ok: true, json: async () => ({sources:[{key:'legacy_key',file_path:'data/selected.csv'}]})};
        };
        vm.runInThisContext(fs.readFileSync('static/core/bourse-source-context.js', 'utf8'));
        (async () => {
            for (const source of ['saxobank_csv', 'saxo:saxobank_csv', 'saxobank', 'saxo:saxobank']) {
                assert.deepEqual(await window.resolveBourseSourceSelection(source), {type:'csv',fileKey:null});
            }
            assert.equal(fetches, 0);
            assert.deepEqual(await window.resolveBourseSourceSelection('saxo:chosen.csv'), {type:'csv',fileKey:'chosen.csv'});
            assert.deepEqual(await window.resolveBourseSourceSelection('saxo:legacy_key'), {type:'csv',fileKey:'selected.csv'});
            assert.deepEqual(await window.resolveBourseSourceSelection('saxobank_api'), {type:'api',fileKey:null});
            assert.deepEqual(await window.resolveBourseSourceSelection('manual_bourse'), {type:'manual',fileKey:null});
        })().catch(error => {console.error(error);process.exit(1)});
    '''
    result = subprocess.run([node, '-e', script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
