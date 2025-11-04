# description: Prepare a new release with version bump and changelog

You are preparing a new release for the Squiggy project. Follow these steps carefully:

## Step 1: Parse and Validate Version

Parse the version argument provided by the user. The version can be specified in three ways:

**Option 1: Explicit version number** (e.g., `0.2.0`, `1.0.0-beta`, `0.1.1`)
- Must follow semantic versioning: MAJOR.MINOR.PATCH
- Optional pre-release suffix (e.g., 0.2.0-beta, 1.0.0-rc.1)

**Option 2: Release type** (e.g., `major`, `minor`, `patch`)
- `major`: Increment major version (0.1.0 → 1.0.0)
- `minor`: Increment minor version (0.1.0 → 0.2.0)
- `patch`: Increment patch version (0.1.0 → 0.1.1)

**IMPORTANT - Pre-release Suffix Handling:**
If the current version has a pre-release suffix (e.g., `-alpha`, `-beta`, `-rc.1`), **preserve it** in the bumped version:
- Current: `0.1.3-alpha` + `patch` → `0.1.4-alpha`
- Current: `0.1.3-alpha` + `minor` → `0.2.0-alpha`
- Current: `0.1.3-alpha` + `major` → `1.0.0-alpha`

This allows continuous development in alpha/beta without manual suffix management.

**Option 3: No argument provided**
- Use the AskUserQuestion tool to prompt the user to select release type:
  - Question: "What type of release is this?"
  - Options:
    - `major` (Breaking changes, incompatible API changes)
    - `minor` (New features, backward compatible)
    - `patch` (Bug fixes, backward compatible)
    - Custom version (Allow user to specify exact version)

After determining the version:
- Version must be newer than the current version
- Validate the version format is correct

## Step 2: Get Current Version

Read the current version from `package.json` (look for `"version": "..."`).

**Note:** `package.json` is the single source of truth for version numbers. The `scripts/sync-version.js` script automatically syncs the version to:
- `squiggy/__init__.py` (__version__)
- `pyproject.toml` (version)
- `package.json` viewsContainers title (sidebar display)
- `package-lock.json` (version)

The sync script runs automatically during build, but you should run it manually after updating the version: `npm run sync`

## Step 3: Collect Changes Since Last Release

Run `git log --oneline --since="$(git describe --tags --abbrev=0 2>/dev/null || git log --reverse --format=%H | head -1)" --format="- %s"` to get commits since the last release.

Analyze the commits and categorize them into:
- **Features**: New functionality (commits with "add", "feat", "feature", "implement", new features from PR titles)
- **Fixes**: Bug fixes (commits with "fix", "bug", "resolve")
- **Improvements**: Enhancements to existing features (commits with "improve", "enhance", "update", "refactor")
- **Documentation**: Documentation changes (commits with "docs", "documentation")
- **Internal**: Internal changes, testing, CI/CD (commits with "test", "ci", "chore", "build")

Generate a concise summary for each category (1-2 lines per item max).

## Step 4: Update CHANGELOG.md

Read the current `CHANGELOG.md` file. Add a new release section at the top with:

```markdown
# Squiggy [VERSION] ([DATE])

[Concise 1-2 sentence summary of the release]

## Features
- [Feature 1]
- [Feature 2]

## Fixes
- [Fix 1]
- [Fix 2]

## Improvements
- [Improvement 1]
- [Improvement 2]

[Previous release sections below...]
```

Only include sections that have content. Skip empty sections.

**IMPORTANT**: Get today's date by running `date +%Y-%m-%d` to ensure the correct date format (YYYY-MM-DD). Do not manually type the date to avoid errors.

## Step 5: Run Quality Checks

Before updating version numbers, run quality checks to ensure the release is ready:

```bash
pixi run check
```

This runs all linting and formatting checks for both Python and TypeScript. If there are any failures, fix them before proceeding with the release.

## Step 6: Update Version Numbers

Update the version in `package.json`:
1. Edit `package.json`: Update `"version": "NEW_VERSION"`
2. Run the sync script: `npm run sync`

This will automatically update:
- `squiggy/__init__.py` (__version__)
- `pyproject.toml` (version)
- `package.json` viewsContainers title (sidebar)
- `package-lock.json` (version)

## Step 7: Stage Changes

Stage all modified files with git:
```bash
git add package.json package-lock.json squiggy/__init__.py pyproject.toml CHANGELOG.md
```

## Step 8: Show Summary

Display a summary showing:
- Old version → New version
- Files updated:
  - `package.json` (version and sidebar title)
  - `package-lock.json` (version)
  - `squiggy/__init__.py` (__version__)
  - `pyproject.toml` (version)
  - `CHANGELOG.md`
- Changes staged with git

Show the staged diff:
```bash
git diff --cached
```

## Step 9: Prompt to Create Commit and Tag

Use the AskUserQuestion tool to ask the user:

**Question:** "Do you want to create the release commit and tag now?"

**Options:**
- **Yes** - Create commit and tag (recommended)
  - Description: "Commit the changes and create a git tag v[VERSION]"
- **No** - Stage only (manual commit later)
  - Description: "Leave changes staged for manual review and commit"

### If user selects "Yes":

1. Create the commit:
   ```bash
   git commit -m "Release v[VERSION]

   [Brief 1-2 sentence summary of key changes from changelog]"
   ```

2. Create the git tag:
   ```bash
   git tag -a v[VERSION] -m "Release v[VERSION]"
   ```

3. Show success message with next steps:
   ```
   ✅ Release v[VERSION] committed and tagged!

   Next steps:
   # Push to remote (this will trigger GitHub Actions):
   git push origin main v[VERSION]

   The GitHub Actions workflow (.github/workflows/release.yml) will automatically:
   - Run tests
   - Build the .vsix extension
   - Create GitHub release with auto-generated notes
   - Mark as pre-release if version contains -alpha, -beta, or -rc
   - Upload the .vsix artifact to the release

   # View the release after CI completes:
   gh release view v[VERSION] --web
   ```

### If user selects "No":

Show manual next steps:
```bash
# Review the changes:
git diff --cached

# When ready, commit the release:
git commit -m "Release v[VERSION]"

# Create a git tag:
git tag -a v[VERSION] -m "Release v[VERSION]"

# Push to remote (this triggers GitHub Actions to create the release):
git push origin main v[VERSION]

# GitHub Actions will automatically:
# - Run tests
# - Build the .vsix extension
# - Create GitHub release with auto-generated notes
# - Upload the .vsix artifact
```

## Important Notes

- Always show the staged diff before prompting to commit
- Only create commit and tag if the user explicitly chooses "Yes" in the prompt
- Do NOT push to remote - always leave that to the user
- Do NOT create GitHub release manually - GitHub Actions (.github/workflows/release.yml) handles this automatically when tags are pushed
- Be concise in the changelog - focus on user-facing changes
- Skip internal/testing changes in the changelog unless they're significant
- Preserve existing CHANGELOG.md content below the new release section
- Use consistent formatting that matches existing CHANGELOG.md style (if it has content)
- Use annotated tags (`git tag -a`) not lightweight tags

## Error Handling

If any step fails:
- Provide clear error messages
- Suggest fixes (e.g., "Version must be higher than 0.1.0")
- Don't proceed to the next step if there are errors
- Don't stage any changes if there were errors

## Usage Examples

```bash
# Automatic version bump by type (preserves pre-release suffix)
/release patch        # 0.1.3-alpha → 0.1.4-alpha
/release minor        # 0.1.3-alpha → 0.2.0-alpha
/release major        # 0.1.3-alpha → 1.0.0-alpha

# Without suffix
/release patch        # 0.1.0 → 0.1.1

# Explicit version number
/release 0.2.0        # Set to exactly 0.2.0
/release 1.0.0-beta   # Pre-release version
/release 0.2.0-alpha  # Continue alpha development

# Interactive prompt
/release              # Ask user to select release type
```
