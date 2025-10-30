---
description: Create git worktrees for parallel Claude Code sessions
---

# Git Worktree Manager for Parallel Development

Create isolated git worktrees and launch parallel Claude Code sessions for concurrent development without context switching.

## Arguments

You must provide one or more worktree specifications in the format: `<branch-name>:<worktree-path>` or just `<branch-name>` (auto-generates path).

Examples:
- `/worktree feature/auth` → Creates `../squiggy-feature-auth/`
- `/worktree bugfix/export:~/dev/export-fix` → Creates at specific path
- `/worktree feature/ui feature/api hotfix/tests` → Creates multiple worktrees

## What This Command Does

For each worktree requested:

1. **Validate Input**: Parse worktree specifications and validate branch names
2. **Create Worktree**: Execute `git worktree add` with proper branch tracking
3. **Setup Environment**:
   - Create TASK.md file with structured template
   - Optionally copy/link dependencies (node_modules, .venv, etc.)
   - Preserve git configuration
4. **Launch Session**: Open new Claude Code window/session for each worktree
5. **Report Status**: Display worktree locations and next steps

## Implementation Steps

### Step 1: Parse Arguments

Extract worktree specifications from the command arguments. Expected formats:
- `<branch-name>` → Auto-generate path as `../<repo>-<branch-sanitized>/`
- `<branch-name>:<path>` → Use explicit path
- Multiple space-separated specifications for parallel setup

### Step 2: Validate Repository State

Before creating any worktrees:
```bash
# Check if we're in a git repository
git rev-parse --git-dir

# Get repository name for path generation
basename $(git rev-parse --show-toplevel)

# Ensure we're on the main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Error: Must be on 'main' branch to create worktrees"
    echo "Current branch: $CURRENT_BRANCH"
    echo "Run: git checkout main"
    exit 1
fi

# Ensure main branch is up to date with remote
git fetch origin main
BEHIND=$(git rev-list --count HEAD..origin/main)
if [ "$BEHIND" -gt 0 ]; then
    echo "⚠️  Warning: 'main' branch is $BEHIND commit(s) behind origin/main"
    echo "Consider running: git pull origin main"
fi

# Check for uncommitted changes (warn but don't block)
git status --porcelain

# List existing worktrees to check for conflicts
git worktree list
```

### Step 3: Create Each Worktree

For each worktree specification:

```bash
# Always use main as the base branch (validated in Step 2)
BASE_BRANCH="main"

# Sanitize branch name for path (replace / with -)
SANITIZED_BRANCH=$(echo "$BRANCH_NAME" | tr '/' '-')

# Generate default path if not provided
WORKTREE_PATH="../${REPO_NAME}-${SANITIZED_BRANCH}"

# Create worktree with new branch from main
git worktree add -b "$BRANCH_NAME" "$WORKTREE_PATH" "$BASE_BRANCH"
```

### Step 4: Setup Worktree Environment

In each new worktree directory:

```bash
cd "$WORKTREE_PATH"

# Create TASK.md template
cat > TASK.md << 'EOF'
# Task: $BRANCH_NAME

## Description
[Describe what this worktree is for]

## Objective
[What needs to be accomplished?]

## Files to Modify
-

## Success Criteria
- [ ]
- [ ]

## Notes
- Branch: $BRANCH_NAME
- Base: main
- Created: $(date)
- Worktree: $WORKTREE_PATH
EOF

# Optionally link dependencies (uncomment if needed)
# ln -s ../../$(basename $ORIGINAL_PATH)/.venv ./.venv
# ln -s ../../$(basename $ORIGINAL_PATH)/node_modules ./node_modules
```

### Step 5: Launch Claude Code Sessions

For each worktree:

```bash
# Option 1: Open in new editor window
code "$WORKTREE_PATH"

# Option 2: Use terminal multiplexer (tmux/screen)
# tmux new-session -d -s "worktree-$SANITIZED_BRANCH" "cd $WORKTREE_PATH && code ."

# Option 3: Just report location for manual opening
echo "✅ Worktree ready at: $WORKTREE_PATH"
```

### Step 6: Provide Usage Instructions

After all worktrees are created:

```bash
echo "
🌳 Git Worktrees Created Successfully!

Worktrees:
$(git worktree list)

Next Steps:
1. Open each worktree in a separate Claude Code window
2. Each session maintains independent context
3. Work on tasks in parallel without context switching

Commands:
- List worktrees:     git worktree list
- Switch to worktree: cd $WORKTREE_PATH
- Remove worktree:    git worktree remove $WORKTREE_PATH
- Cleanup merged:     git worktree prune

Tips:
- Each worktree has its own TASK.md - update it with your goals
- Commits in any worktree immediately available in all others
- Share dependencies with symlinks to save disk space
- Use 'git worktree remove' when done with a task
"
```

## Error Handling

Handle common issues gracefully:

1. **Branch Already Exists**:
   - Check if branch exists: `git rev-parse --verify $BRANCH_NAME`
   - Offer to use existing branch or force new branch

2. **Branch Already Checked Out**:
   - List conflicting worktrees: `git worktree list`
   - Suggest removing old worktree or using different branch

3. **Path Already Exists**:
   - Check if path is a worktree: `git worktree list | grep $PATH`
   - Suggest different path or cleanup

4. **Disk Space**:
   - Warn if creating multiple large worktrees
   - Suggest using symlinks for dependencies

## Advanced Options

Consider these optional features:

### Dependency Management

```bash
# Python projects - share virtual environment
if [ -d ".venv" ]; then
    echo "Found .venv - consider sharing with: ln -s $ORIGINAL_PATH/.venv"
fi

# Node projects - use pnpm for automatic sharing
if [ -f "package.json" ]; then
    echo "Found package.json - consider using pnpm for shared dependencies"
fi
```

### Parallel Installation

```bash
# Install dependencies in parallel for all worktrees
for worktree in $WORKTREE_PATHS; do
    (cd "$worktree" && uv pip install -e ".[dev]") &
done
wait
```

### Cleanup Automation

```bash
# Function to remove merged worktrees
cleanup_merged_worktrees() {
    git worktree list | while read path branch commit; do
        branch_name=$(echo $branch | sed 's/[][]//g')
        if git branch --merged main | grep -q "$branch_name"; then
            echo "Removing merged worktree: $path"
            git worktree remove "$path"
            git branch -d "$branch_name"
        fi
    done
}
```

## Example Usage

### Single Worktree
```bash
/worktree feature/authentication
# Creates: ../squiggy-feature-authentication/
# Opens: New Claude Code window
```

### Multiple Parallel Worktrees
```bash
/worktree feature/ui-redesign bugfix/export-svg feature/bam-validation
# Creates 3 worktrees in parallel
# Opens 3 separate Claude Code sessions
# Each maintains independent context
```

### Custom Paths
```bash
/worktree feature/ml-pipeline:~/projects/squiggy-ml
# Creates worktree at specific location
```

## Workflow Benefits

1. **Context Preservation**: Each Claude session maintains full project understanding
2. **Parallel Development**: Work on multiple features simultaneously
3. **Zero Context Switching**: No git stash, no lost AI context
4. **Isolated Testing**: Test changes without affecting main work
5. **Safe Experimentation**: Delete worktree if experiment fails

## Important Notes

- **Must be on main branch**: All worktrees are created from the `main` branch for consistency
- **Main branch should be up-to-date**: Fetch latest changes before creating worktrees
- All worktrees share the same `.git` directory (efficient disk usage)
- Commits made in any worktree are immediately available in all others
- Cannot checkout the same branch in multiple worktrees simultaneously
- Use descriptive branch names for easy identification
- Clean up worktrees when done to avoid clutter

## Troubleshooting

**"Must be on 'main' branch"**: You must switch to the main branch before creating worktrees. Run `git checkout main` first.

**"main branch is behind origin/main"**: Your local main branch is out of sync. Run `git pull origin main` to update.

**"Branch already checked out"**: Another worktree is using this branch. List with `git worktree list`.

**Disk space concerns**: Share dependencies with symlinks or use containerization.

**Too many worktrees**: Use `git worktree list` and `git worktree remove` to clean up.

**Dependencies conflicts**: Use separate virtual environments or version pinning.

---

## Implementation

Execute the following bash commands to create the requested worktrees and set up parallel development sessions.
