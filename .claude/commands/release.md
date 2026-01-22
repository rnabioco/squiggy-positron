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

## Step 2: Check for Claude Planning Files

Before proceeding with the release, check for any Claude planning files at the project root that should not be included in the release:

```bash
# Find .md files at project root with underscores (typical of planning files)
# Exclude standard documentation files: README, CHANGELOG, LICENSE, etc.
find . -maxdepth 1 -type f -name "*_*.md" | sed 's|^\./||' | grep -v -E "^(README|CHANGELOG|LICENSE|CONTRIBUTING|CODE_OF_CONDUCT)\.md$" || true
```

If any planning files are found (e.g., `PLANNING_NOTES.md`, `FEATURE_DESIGN.md`, `IMPLEMENTATION_GUIDE.md`), use the AskUserQuestion tool to ask:

**Question:** "Found Claude planning files at project root: [LIST FILES]. These should not be included in the release. What would you like to do?"

**Options:**
- **Move to docs/guides** - Move files to docs/guides directory (recommended)
  - Description: "Move planning files to docs/guides/ for future reference"
- **Delete** - Delete the planning files
  - Description: "Permanently delete these planning files"
- **Keep** - Keep files at root (not recommended)
  - Description: "Keep files at project root (will be included in release)"

### If user selects "Move to docs/guides":

1. Create docs/guides directory if it doesn't exist:
   ```bash
   mkdir -p docs/guides
   ```

2. Move each planning file:
   ```bash
   for file in [PLANNING_FILES]; do
       git mv "$file" docs/guides/
   done
   ```

3. Stage the changes:
   ```bash
   git add docs/guides/
   ```

4. Show success message:
   ```
   ✅ Moved planning files to docs/guides/
   ```

### If user selects "Delete":

1. Remove each planning file:
   ```bash
   for file in [PLANNING_FILES]; do
       rm "$file"
   done
   ```

2. Show success message:
   ```
   ✅ Deleted planning files
   ```

### If user selects "Keep":

Continue to next step without changes.

**Note:** If no planning files are found, skip this step and proceed directly to Step 3.

## Step 3: Get Current Version

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

## Step 7: Build and Verify VSIX

Before committing the release, build the .vsix extension to verify the build and check file size:

1. Build the extension:
   ```bash
   pixi run build
   ```

2. Find the .vsix file and get its size:
   ```bash
   ls -lh *.vsix | tail -n 1
   ```

3. Display the VSIX information:
   - File name: `squiggy-positron-[VERSION].vsix`
   - File size in human-readable format (MB)
   - Path to file

4. Use the AskUserQuestion tool to prompt:

   **Question:** "The extension build is complete. File size: [SIZE] MB. Do you want to proceed with the release?"

   **Options:**
   - **Yes** - Continue with release (stage changes, commit, and tag)
     - Description: "Proceed to stage changes and create release commit and tag"
   - **No** - Abort release
     - Description: "Cancel the release process and restore original version numbers"

   ### If user selects "No":

   Abort the release and restore original state:
   ```bash
   # Restore original files
   git checkout package.json package-lock.json squiggy/__init__.py pyproject.toml CHANGELOG.md

   # Clean up build artifacts
   rm -f *.vsix
   ```

   Show message:
   ```
   ❌ Release aborted. All version changes reverted.
   ```

   Stop the release process.

   ### If user selects "Yes":

   Continue to Step 8.

## Step 8: Stage Changes

Stage all modified files with git:
```bash
git add package.json package-lock.json squiggy/__init__.py pyproject.toml CHANGELOG.md
```

## Step 9: Show Summary

Display a summary showing:
- Old version → New version
- Files updated:
  - `package.json` (version and sidebar title)
  - `package-lock.json` (version)
  - `squiggy/__init__.py` (__version__)
  - `pyproject.toml` (version)
  - `CHANGELOG.md`
- Built artifact:
  - `squiggy-positron-[VERSION].vsix` ([SIZE] MB)
- Changes staged with git

Show the staged diff:
```bash
git diff --cached
```

## Step 10: Prompt to Create Commit and Tag

Use the AskUserQuestion tool to ask the user:

**Question:** "Do you want to create the release commit and tag now?"

**Options:**
- **Yes** - Create commit and tag (recommended)
  - Description: "Commit the changes and create git tag v[VERSION]"
- **No** - Stage only (manual commit later)
  - Description: "Leave changes staged for manual review and commit"

**Note:** The .vsix file will NOT be committed to git (it's in .gitignore). It's built here only to verify the build process and file size before release.

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

3. Show the commit summary:
   ```bash
   git log -1 --oneline
   git show v[VERSION] --quiet
   ```

4. Proceed to Step 11 (Push confirmation).

### If user selects "No" (Step 10):

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

## Step 11: Prompt to Push Release

Use the AskUserQuestion tool to ask the user:

**Question:** "Release v[VERSION] is committed and tagged locally. Push to remote now?"

**Options:**
- **Yes** - Push now (recommended)
  - Description: "Push commit and tag to trigger GitHub Actions release"
- **No** - Don't push yet
  - Description: "Keep release local for now; push manually later"

### If user selects "Yes":

1. Push to remote:
   ```bash
   git push origin main v[VERSION]
   ```

2. Show success message:
   ```
   ✅ Release v[VERSION] pushed!

   The GitHub Actions workflow (.github/workflows/release.yml) will automatically:
   - Run tests
   - Build the .vsix extension
   - Create GitHub release with auto-generated notes
   - Mark as pre-release if version contains -alpha, -beta, or -rc
   - Upload the .vsix artifact to the release

   # View the release after CI completes:
   gh release view v[VERSION] --web
   ```

### If user selects "No" (Step 11):

Show manual push instructions:
```
✅ Release v[VERSION] committed and tagged locally.

When ready to publish, run:
git push origin main v[VERSION]
```

## Important Notes

- Always build the .vsix and report file size before prompting to proceed with release
- The .vsix file is built as a verification step but is NOT committed (it's in .gitignore)
- If the user aborts at the VSIX check, restore all version changes with `git checkout`
- Always show the staged diff before prompting to commit
- Only create commit and tag if the user explicitly chooses "Yes" in Step 10
- Only push to remote if the user explicitly chooses "Yes" in Step 11
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
- If the VSIX build fails, restore original version numbers and stop the release process

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
