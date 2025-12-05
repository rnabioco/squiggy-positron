#!/usr/bin/env node

/**
 * Version synchronization script
 *
 * Syncs version number from package.json (source of truth) to:
 * - squiggy/__init__.py (__version__)
 * - pyproject.toml (version)
 * - package-lock.json (version)
 *
 * Note: Version is displayed in the status bar (see extension.ts),
 * not in the sidebar title.
 *
 * Run automatically before builds to keep versions in sync.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Paths
const rootDir = path.join(__dirname, '..');
const packageJsonPath = path.join(rootDir, 'package.json');
const pyInitPath = path.join(rootDir, 'squiggy', '__init__.py');
const pyprojectPath = path.join(rootDir, 'pyproject.toml');

// Read package.json
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
const version = packageJson.version;

if (!version) {
    console.error('Error: No version found in package.json');
    process.exit(1);
}

console.log(`Syncing version: ${version}`);

// Update squiggy/__init__.py
let pyInit = fs.readFileSync(pyInitPath, 'utf8');
const pyInitUpdated = pyInit.replace(
    /__version__\s*=\s*["'][^"']*["']/,
    `__version__ = "${version.replace('-alpha', '').replace('-beta', '').replace('-rc', '')}"`
);

if (pyInit !== pyInitUpdated) {
    fs.writeFileSync(pyInitPath, pyInitUpdated);
    console.log(`  ✓ Updated squiggy/__init__.py`);
} else {
    console.log(`  ✓ squiggy/__init__.py already up to date`);
}

// Update pyproject.toml
let pyproject = fs.readFileSync(pyprojectPath, 'utf8');
const pyprojectUpdated = pyproject.replace(
    /^version\s*=\s*["'][^"']*["']/m,
    `version = "${version.replace('-alpha', '').replace('-beta', '').replace('-rc', '')}"`
);

if (pyproject !== pyprojectUpdated) {
    fs.writeFileSync(pyprojectPath, pyprojectUpdated);
    console.log(`  ✓ Updated pyproject.toml`);
} else {
    console.log(`  ✓ pyproject.toml already up to date`);
}

// Update package-lock.json
try {
    execSync('npm install --package-lock-only', { cwd: rootDir, stdio: 'pipe' });
    console.log(`  ✓ Updated package-lock.json`);
} catch (error) {
    console.error(`  ✗ Failed to update package-lock.json: ${error.message}`);
    process.exit(1);
}

console.log(`\nVersion sync complete: ${version}`);
