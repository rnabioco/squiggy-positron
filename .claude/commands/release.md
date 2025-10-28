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

Read the current version from:
- `src/squiggy/__init__.py` (look for `__version__ = "..."`)
- `pyproject.toml` (look for `version = "..."`)

Verify both files have the same version. If they don't match, alert the user.

## Step 3: Collect Changes Since Last Release

Run `git log --oneline --since="$(git describe --tags --abbrev=0 2>/dev/null || git log --reverse --format=%H | head -1)" --format="- %s"` to get commits since the last release.

Analyze the commits and categorize them into:
- **Features**: New functionality (commits with "add", "feat", "feature", "implement", new features from PR titles)
- **Fixes**: Bug fixes (commits with "fix", "bug", "resolve")
- **Improvements**: Enhancements to existing features (commits with "improve", "enhance", "update", "refactor")
- **Documentation**: Documentation changes (commits with "docs", "documentation")
- **Internal**: Internal changes, testing, CI/CD (commits with "test", "ci", "chore", "build")

Generate a concise summary for each category (1-2 lines per item max).

## Step 4: Update NEWS.md

Read the current `NEWS.md` file. Add a new release section at the top with:

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

Use today's date in YYYY-MM-DD format.

## Step 5: Update Version Numbers

Update the version in these files:
1. `src/squiggy/__init__.py`: Update `__version__ = "NEW_VERSION"`
2. `pyproject.toml`: Update `version = "NEW_VERSION"`

## Step 6: Stage Changes

Stage all modified files with git:
```bash
git add src/squiggy/__init__.py pyproject.toml NEWS.md
```

## Step 7: Show Summary and Next Steps

Display a summary showing:
- Old version → New version
- Files updated
- Changes staged
- Suggested commit message:
  ```
  Release v[VERSION]

  [Brief summary of key changes]
  ```
- Next steps:
  ```bash
  # Review the changes:
  git diff --cached

  # Commit the release:
  git commit -m "Release v[VERSION]"

  # Create a git tag:
  git tag v[VERSION]

  # Push to remote:
  git push origin main --tags
  ```

## Important Notes

- Do NOT create the git commit or tag automatically - only stage the changes
- Do NOT push to remote - leave that to the user
- Be concise in the changelog - focus on user-facing changes
- Skip internal/testing changes in the changelog unless they're significant
- Preserve existing NEWS.md content below the new release section
- Use consistent formatting that matches existing NEWS.md style (if it has content)

## Error Handling

If any step fails:
- Provide clear error messages
- Suggest fixes (e.g., "Version must be higher than 0.1.0")
- Don't proceed to the next step if there are errors
- Don't stage any changes if there were errors

## Usage Examples

```bash
# Automatic version bump by type
/release patch        # 0.1.0 → 0.1.1
/release minor        # 0.1.0 → 0.2.0
/release major        # 0.1.0 → 1.0.0

# Explicit version number
/release 0.2.0        # Set to exactly 0.2.0
/release 1.0.0-beta   # Pre-release version

# Interactive prompt
/release              # Ask user to select release type
```
