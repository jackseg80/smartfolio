#!/usr/bin/env node

/**
 * Script de migration console.log → console.debug
 *
 * Safe car console.debug existe nativement et debugLogger l'intercepte.
 * Usage: node scripts/dev_tools/migrate-console-to-debug.cjs [--dry-run]
 */

const fs = require('fs');
const path = require('path');

const DRY_RUN = process.argv.includes('--dry-run');
const SINGLE_FILE = process.argv.find(arg => arg.endsWith('.js') && !arg.includes('migrate-'));

console.log(`🔄 Migration console.log → console.debug ${DRY_RUN ? '(DRY RUN)' : ''}`);

// Patterns à remplacer
const replacements = [
    // console.log → console.debug (sauf si déjà debug)
    {
        pattern: /console\.log\(/g,
        replacement: 'console.debug(',
        category: 'log→debug'
    },
    // console.warn reste console.warn mais on peut tracker
    // (déjà géré par debugLogger, on ne touche pas)
];

function migrateFile(filePath) {
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        const originalContent = content;
        let stats = { 'log→debug': 0 };

        for (const rule of replacements) {
            const matches = content.match(rule.pattern);
            if (matches) {
                stats[rule.category] = matches.length;
                content = content.replace(rule.pattern, rule.replacement);
            }
        }

        const totalChanges = Object.values(stats).reduce((a, b) => a + b, 0);

        if (totalChanges > 0) {
            if (!DRY_RUN) {
                fs.writeFileSync(filePath, content, 'utf8');
            }
            console.log(`✅ ${filePath}: ${totalChanges} changes`, stats);
            return stats;
        }

        return null;
    } catch (error) {
        console.error(`❌ Error processing ${filePath}:`, error.message);
        return null;
    }
}

function findJSFiles() {
    const files = [];

    function scan(dir) {
        try {
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);
                if (entry.isDirectory() && !entry.name.includes('archive') && entry.name !== 'node_modules') {
                    scan(fullPath);
                } else if (entry.isFile() && entry.name.endsWith('.js') && !entry.name.includes('.min.')) {
                    // Exclure debug-logger.js lui-même
                    if (!entry.name.includes('debug-logger')) {
                        files.push(fullPath);
                    }
                }
            }
        } catch (e) {
            // Ignore permission errors
        }
    }

    scan('static');
    return files;
}

function main() {
    let files;

    if (SINGLE_FILE) {
        files = [SINGLE_FILE];
        console.log(`\n🎯 Single file mode: ${SINGLE_FILE}\n`);
    } else {
        files = findJSFiles();
        console.log(`\n🔍 Found ${files.length} JavaScript files to process\n`);
    }

    let totalStats = { 'log→debug': 0, files: 0 };

    for (const file of files) {
        const stats = migrateFile(file);
        if (stats) {
            totalStats.files++;
            Object.keys(stats).forEach(key => {
                totalStats[key] = (totalStats[key] || 0) + stats[key];
            });
        }
    }

    console.log('\n📊 Migration Summary:');
    console.log(`Files modified: ${totalStats.files}`);
    console.log(`console.log → console.debug: ${totalStats['log→debug']}`);

    if (DRY_RUN) {
        console.log('\n💡 Run without --dry-run to apply changes');
    } else {
        console.log('\n✅ Migration completed!');
        console.log('ℹ️  console.debug is intercepted by debugLogger');
        console.log('ℹ️  Use debugOn()/debugOff() in console to toggle visibility');
    }
}

main();
