# Review and Update Documentation

Your task is to review and update the Squiggy documentation to ensure it accurately reflects the current codebase.

## Steps to Complete:

### 1. Review Documentation for Accuracy

Read the following documentation files and compare against the current codebase:
- `docs/index.md` - Home page with features list
- `docs/installation.md` - Installation instructions
- `docs/usage.md` - Usage guide with plot modes, features, and examples
- `docs/development.md` - Development and contributing guide
- `README.md` - Main repository README with overview, installation, and quick start

For each file, check:
- Are all features mentioned still current?
- Are there new features in the extension not documented?
- Are plot modes accurate? (check `squiggy/plotter.py` for plot modes)
- Are normalization methods current? (check `squiggy/normalization.py`)
- Are dependencies and installation instructions accurate? (check `pyproject.toml` and `pixi.toml`)
- Are pixi commands documented? (check `pixi.toml` tasks section)
- Are extension commands current? (check `src/extension.ts` and `package.json`)
- Are file references current? (check that referenced files exist: `squiggy-reads-view-pane.ts` not `squiggy-read-explorer.ts`)

**IMPORTANT**: Review the actual source code files to verify documentation accuracy. Don't assume - check!

### 2. Update Documentation

If you find any discrepancies:
- Update the relevant documentation files with accurate information
- Add documentation for any new features discovered
- Remove or update information about deprecated/changed features
- Ensure code examples and CLI usage examples are correct

**Special attention for README.md:**
- Verify installation instructions use pixi
- Update quick start examples to use pixi commands
- Ensure feature list matches what's actually implemented
- Check that example commands use correct syntax
- Verify links to DEVELOPER.md and USER_GUIDE.md are correct

### 3. Test Documentation

Build and serve the documentation locally to verify everything works:

```bash
pixi run docs
```

Then:
- Verify the server starts successfully at http://127.0.0.1:8000
- Check that all images load correctly
- Verify markdown formatting is correct
- Confirm all links work
- Kill the server when done (Ctrl+C)

### 4. Report Findings

Provide a summary of:
- What discrepancies were found (if any)
- What documentation was updated
- Whether the documentation builds and serves correctly
- Any issues or recommendations

## Success Criteria

- [ ] All documentation files reviewed against current codebase
- [ ] Any discrepancies found and fixed
- [ ] Documentation builds and serves without errors (`pixi run docs`)
- [ ] Summary report provided to user
