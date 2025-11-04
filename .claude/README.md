# Claude Code Resources

This directory contains resources specifically for Claude Code users and AI-assisted development.

## Files in This Directory

### `CLAUDE_CODE_WEB.md` ⭐ NEW
**Essential guide for Claude Code Web development:**
- Understanding ephemeral environments
- SessionStart hook setup
- Available slash commands
- Development workflow
- Troubleshooting tips
- Web-specific best practices

### `hooks/session-start`
Automatic setup script that runs when starting a session:
- Installs npm dependencies
- Installs Python dependencies (Python 3.12)
- Sets environment variables
- Configures aliases

### `commands/` directory
Slash commands for common workflows:
- `/env-check` - Verify environment setup (Web compatible)
- `/test-web` - Run tests without pixi (Web compatible)
- `/test` - Run tests with pixi (local dev)
- `/compile` - Build the extension
- `/docs` - Serve documentation
- `/release` - Create releases
- `/squiggy` - Squiggy-specific operations
- `/worktree` - Git worktree management

### `settings.json`
Project-level Claude Code configuration:
- Allowed tools and permissions
- Additional context for Claude
- Shared with all team members

## Main Project Documentation

### For All Contributors

- **[DEVELOPER.md](../docs/DEVELOPER.md)** - Comprehensive developer guide
  - Setup instructions
  - Development workflow
  - Testing guidelines
  - Architecture overview
  - Common tasks and troubleshooting

- **[README.md](../README.md)** - Project overview and installation
  - Features and usage
  - Quick start guide
  - Installation instructions

### For AI-Assisted Development

- **[CLAUDE.md](../CLAUDE.md)** - Detailed context for Claude Code (project root)
  - Complete architecture documentation (TypeScript + Python)
  - Key components and data flow
  - Development patterns and conventions
  - Important constraints and gotchas
  - Code style guidelines
  - Positron integration patterns

- **[CLAUDE_CODE_WEB.md](./CLAUDE_CODE_WEB.md)** - Web-specific guide ⭐
  - Ephemeral environment considerations
  - SessionStart hook usage
  - Slash commands reference
  - Development workflow for Web
  - Troubleshooting

## Getting Started with Claude Code Web

### Automatic Setup (Recommended)

**Dependencies install automatically!** The SessionStart hook runs when you start a session:

1. **Start working on an issue** in Claude Code Web
2. **Wait for setup** (2-3 minutes) - The hook installs dependencies
3. **Verify with** `/env-check` - Ensure everything is ready
4. **Start developing!** - All tools are available

### Manual Setup (Local Development)

If working locally with pixi:

```bash
# Clone and install
git clone https://github.com/rnabioco/squiggy-positron.git
cd squiggy-positron
pixi install
pixi run setup

# Start developing
pixi run dev        # Watch mode
pixi run test       # Run all tests
pixi run lint       # Lint and format
```

If working locally without pixi:

```bash
# Install dependencies
npm install
python3.12 -m pip install -e ".[dev,export]"

# Build and test
npm run compile
npm test
python3.12 -m pytest tests/
```

## Development Workflow

### Quick Reference

```bash
# Verify environment
/env-check

# Run tests (Web compatible)
/test-web

# Build extension
npm run compile

# Format code
npm run format
python3.12 -m ruff format squiggy/ tests/

# Lint code
npm run lint
python3.12 -m ruff check squiggy/ tests/
```

### Sample Data

Located in `tests/data/`:
- `yeast_trna_reads.pod5` - POD5 file with 180 reads
- `yeast_trna_mappings.bam` - BAM file with alignments

## Key Context for Claude

When working with Claude Code on this project, Claude has access to:

- **Architecture**: TypeScript Positron extension + Python package
- **Frontend**: VSCode Extension API, Positron Runtime API, React 18 webviews
- **Backend**: POD5/BAM file parsing, Bokeh visualization
- **Data flow**: Extension → Python kernel → POD5/BAM → Bokeh → Webview
- **Patterns**: Positron Runtime communication, React webviews, Strategy pattern for plots
- **Constraints**: Python 3.12+, TypeScript strict mode, 100-char lines (TS), 88-char lines (Python)

## Common Claude Code Workflows

### Adding a Feature

```
"Add a new plot normalization method called [name] to the Python backend"
"Add export functionality to save plots as PNG with current zoom level"
"Create a new webview panel for filtering base modifications"
"Add support for FASTA reference sequences in the extension"
```

### Fixing Bugs

```
"Fix the issue where read IDs aren't loading from large POD5 files"
"Debug why the Reads panel isn't refreshing after loading a BAM file"
"Fix the TypeScript error in the plot options webview"
"Resolve the Python 3.12 compatibility issue with pod5 package"
```

### Refactoring

```
"Refactor the plotting code to use the Strategy pattern"
"Extract Bokeh theme configuration into a centralized ThemeManager"
"Improve error handling in the Positron Runtime communication"
"Convert the File panel to use React instead of HTML strings"
```

### Testing

```
"Write Jest tests for the ReadsViewPane component"
"Add pytest tests for the new aggregate plot strategy"
"Create integration tests for POD5 loading via Positron kernel"
"Add test coverage for base modification parsing"
```

## Issue Templates

GitHub issue templates are available in `.github/ISSUE_TEMPLATE/`:
- `bug_report.md` - For reporting bugs
- `feature_request.md` - For suggesting new features

## Pull Request Template

PR template in `.github/pull_request_template.md` includes:
- Change description and checklist
- Testing requirements
- Review guidelines
- Claude Code specific notes

## Project Philosophy

Squiggy Positron Extension is designed to be:
- **Seamless**: Integrates directly into Positron IDE workflow
- **Performant**: Lazy loading, virtualized lists, efficient data handling
- **Interactive**: Bokeh plots with zoom, pan, selection
- **Flexible**: Multiple plot modes, normalization options, filtering
- **Developer-friendly**: Clear architecture, TypeScript + Python, good documentation
- **AI-assistant ready**: Rich context for Claude Code, automated setup

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Use the bug report template
- **Features**: Use the feature request template
- **Development**: Check `CLAUDE.md` and the [Development Guide](../docs/development.md)

## License

MIT License - see [LICENSE](../LICENSE) file for details.
