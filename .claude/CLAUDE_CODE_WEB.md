# Claude Code Web Development Guide

This document provides guidance for developing Squiggy using Claude Code Web.

## 🌐 Understanding Claude Code Web

### Ephemeral Environment
- Each session runs in a **fresh container** cloned from your GitHub repo's default branch
- **No persistent storage** - the container is wiped when the task completes
- **SessionStart hook runs automatically** to set up the environment
- Dependencies are installed fresh each session via the hook

### Session Lifecycle

```
1. Container starts → Clone repo from GitHub
2. SessionStart hook runs → Install dependencies (automatic)
3. Development work → Environment persists during session
4. Task completes → Container terminates and wipes
```

## ⚙️ Automatic Setup (SessionStart Hook)

The `.claude/hooks/session-start` script automatically:

1. **Installs npm dependencies** (if `node_modules/` is missing)
2. **Installs Python packages** using Python 3.12 (if squiggy not importable)
3. **Sets environment variables** for the session:
   - `PYTHON=/usr/bin/python3.12`
   - `python` alias → `python3.12`
   - `pytest` alias → `python3.12 -m pytest`

**You don't need to manually install dependencies** - they're set up automatically when the session starts!

## 📋 Available Slash Commands

### For Web Environments (No pixi required)

- `/env-check` - Verify environment setup and dependencies
- `/test-web` - Run quality checks and tests (direct npm/python commands)
- `/compile` - Build the extension

### Original Commands (Require pixi)

- `/test` - Run tests using pixi (works in local dev, not Web)
- `/docs` - Build and serve documentation
- `/release` - Create a new release
- `/squiggy` - Squiggy-specific operations
- `/worktree` - Git worktree operations

## 🔧 Development Workflow

### Starting a New Session

1. **Open issue or start session** - Claude Code Web clones repo
2. **Wait for SessionStart** - Dependencies install automatically (~2-3 minutes)
3. **Verify setup** - Run `/env-check` to confirm everything installed
4. **Start developing** - Environment is ready!

### Running Tests

```bash
# TypeScript tests
npm test

# Python tests
python3.12 -m pytest tests/ -v

# Or use slash command
/test-web
```

### Building the Extension

```bash
# Compile TypeScript
npm run compile

# Package as .vsix
npm run package
```

### Linting and Formatting

```bash
# TypeScript
npm run lint
npm run format

# Python
python3.12 -m ruff check squiggy/ tests/
python3.12 -m ruff format squiggy/ tests/
```

## 🚨 Important Notes

### Python Version
- **Always use Python 3.12** - The project requires Python 3.12+
- The SessionStart hook sets `python` alias to `python3.12`
- If running commands manually, use: `python3.12` or `/usr/bin/python3.12`

### Pixi Not Available
- Pixi is **not installed** in Claude Code Web environments
- Use direct `npm` and `python3.12` commands instead
- The `/test-web` command provides pixi-free equivalents

### Package Installation
- Dependencies are installed via `--break-system-packages` flag
- This is safe in the ephemeral container environment
- Packages are NOT persisted between sessions

### Git Workflow
- Always work on feature branches (never commit to main)
- Push changes to your branch
- The container has access to push to GitHub

## 🎯 Best Practices

### 1. Use Slash Commands
Slash commands encapsulate common workflows:
```
/env-check     # Before starting work
/test-web      # Before committing
/compile       # Verify builds work
```

### 2. Verify Environment Early
After SessionStart completes, run `/env-check` to ensure:
- Dependencies installed correctly
- Python 3.12 is available
- Build tools are working

### 3. Commit Frequently
Since environments are ephemeral:
- Commit and push changes regularly
- Don't rely on local state persisting
- Use descriptive commit messages

### 4. Test Before Pushing
Always run tests before pushing:
```bash
/test-web
```

### 5. Check CLAUDE.md
The main `CLAUDE.md` file contains comprehensive project guidance.
Claude Code reads it automatically each session.

## 📝 Configuration Files

### `.claude/settings.json` (Project-level)
- Shared with all team members (checked into git)
- Defines allowed tools and permissions
- Provides additional context to Claude

### `~/.claude.json` (User-level)
- Your personal preferences (not in git)
- Applies to all your Claude Code projects
- Settings like API keys, personal aliases

### `.claude/hooks/session-start` (Automatic Setup)
- Runs at session start
- Installs dependencies
- Sets environment variables via `$CLAUDE_ENV_FILE`

## 🔍 Troubleshooting

### "Module not found" errors
- Run `/env-check` to verify dependencies
- If missing, the SessionStart hook may have failed
- Try manually running: `npm install` and `python3.12 -m pip install -e ".[dev,export]"`

### Python version mismatch
- Use `python3.12` explicitly
- Check: `which python` and `python --version`
- The alias should point to Python 3.12

### Build failures
- Verify TypeScript compiles: `npm run compile`
- Check for linting errors: `npm run lint`
- Ensure all dependencies installed: `/env-check`

### Tests failing
- Ensure you're on the correct branch
- Run `git status` to check for uncommitted changes
- Verify test data is present: `ls squiggy/data/`

## 📚 Additional Resources

- [CLAUDE.md](../CLAUDE.md) - Main project instructions
- [README.md](../README.md) - Project overview
- [DEVELOPER.md](../docs/DEVELOPER.md) - Full development guide
- [Claude Code Docs](https://docs.claude.com/en/docs/claude-code/) - Official documentation

## 💡 Tips for Claude Code Web

1. **Sessions persist across interactions** - You can continue working across multiple messages
2. **Background tasks work** - Long-running commands can continue while you work
3. **Git operations are fast** - Clone, commit, push all work normally
4. **No Docker required** - The Web environment handles sandboxing
5. **NetworkFiltering protects you** - Outbound network access is restricted

---

**Questions?** Check the main [CLAUDE.md](../CLAUDE.md) or ask Claude directly!
