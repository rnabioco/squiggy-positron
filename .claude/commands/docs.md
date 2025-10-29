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
- Are there new features in the code not documented?
- Are CLI arguments up to date? (check `src/squiggy/cli.py` and `src/squiggy/main.py`)
- Are plot modes accurate? (check `src/squiggy/constants.py` for PlotMode enum)
- Are normalization methods current? (check `src/squiggy/constants.py` for NormalizationMethod enum)
- Are dependencies and installation instructions accurate? (check `pyproject.toml`)
- Are keyboard shortcuts current? (check `src/squiggy/viewer.py`)
- Are export formats accurate? (check `src/squiggy/dialogs.py` ExportDialog)

**IMPORTANT**: Review the actual source code files to verify documentation accuracy. Don't assume - check!

### 2. Update Documentation

If you find any discrepancies:
- Update the relevant documentation files with accurate information
- Add documentation for any new features discovered
- Remove or update information about deprecated/changed features
- Ensure code examples and CLI usage examples are correct

**Special attention for README.md:**
- Ensure CLI flags and arguments match current implementation
- Verify installation instructions use uv (not pip)
- Update quick start examples to reflect current usage patterns
- Ensure feature list matches what's actually implemented
- Check that example commands use correct syntax

### 3. Ask About Regenerating Screenshots

**IMPORTANT**: Before proceeding with screenshot regeneration, use the AskUserQuestion tool to ask the user:

- Question: "Do you want to regenerate documentation screenshots?"
- Header: "Screenshots"
- Options:
  1. "Yes" - "Regenerate all screenshots using the latest code (requires the app to be functional)"
  2. "No" - "Skip screenshot regeneration and proceed with documentation review only"

If the user selects "Yes", run the screenshot generation script:

```bash
./scripts/generate_docs_screenshots.sh
```

This will regenerate all screenshots in `docs/images/` using the latest code and sample data.

If the user selects "No", skip the screenshot generation step and proceed to testing the documentation build.

### 4. Test Documentation

Build and serve the documentation locally to verify everything works:

```bash
mkdocs serve
```

Then:
- Verify the server starts successfully
- Check that all images load correctly
- Verify markdown formatting is correct
- Confirm all links work
- Kill the server when done

### 5. Report Findings

Provide a summary of:
- What discrepancies were found (if any)
- What documentation was updated
- Whether screenshots were regenerated successfully
- Whether the documentation builds and serves correctly
- Any issues or recommendations

## Success Criteria

- [ ] All documentation files reviewed against current codebase
- [ ] Any discrepancies found and fixed
- [ ] User prompted about screenshot regeneration
- [ ] Screenshots regenerated successfully (if user chose "Yes")
- [ ] Documentation builds and serves without errors
- [ ] Summary report provided to user
