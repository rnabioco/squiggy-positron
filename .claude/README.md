# Claude Code Resources

This directory contains resources specifically for Claude Code users and AI-assisted development.

## Files in This Directory

### `quick-reference.md`
Quick reference guide with:
- Common commands and file locations
- Code patterns for Qt/async/plotting
- Testing workflows
- Debugging tips
- Useful Claude Code prompts

## Main Project Documentation

### For All Contributors

- **[Development Guide](../docs/development.md)** - Comprehensive contributor guide
  - Setup instructions
  - Development workflow
  - Testing guidelines
  - PR checklist
  - Common tasks and troubleshooting

- **[README.md](../README.md)** - Project overview and installation
  - Features and usage
  - Quick start guide
  - Architecture overview

### For AI-Assisted Development

- **[CLAUDE.md](../CLAUDE.md)** - Detailed context for Claude Code (project root)
  - Complete architecture documentation
  - Key components and data flow
  - Development commands
  - Async/Qt patterns
  - Common tasks and gotchas
  - Coding style conventions

## Development Tools

### Environment Validation

```bash
# Check that your development environment is properly configured
python scripts/check_dev_env.py
```

This script validates:
- Python version (3.8+)
- Required dependencies
- Optional dependencies
- Sample data availability
- Package structure

### Sample Data

Located in `tests/data/`:
- `simplex_reads.pod5` - Small POD5 file with ~10 reads
- `simplex_reads_mapped.bam` - BAM file with basecalls and alignments

Use for development and testing:
```bash
squiggy -p tests/data/simplex_reads.pod5 -b tests/data/simplex_reads_mapped.bam
```

## Getting Started with Claude Code

1. **Clone and setup:**
   ```bash
   git clone https://github.com/rnabioco/squiggy.git
   cd squiggy
   pip install -e ".[dev]"
   pip install -e ".[macos]"  # macOS only
   ```

2. **Validate environment:**
   ```bash
   python scripts/check_dev_env.py
   ```

3. **Start developing:**
   - Claude will automatically use `CLAUDE.md` for project context
   - Reference `quick-reference.md` for common patterns
   - Follow guidelines in the [Development Guide](../docs/development.md)

4. **Test your changes:**
   ```bash
   ruff format src/ tests/
   ruff check --fix src/ tests/
   pytest tests/
   squiggy -p tests/data/simplex_reads.pod5
   ```

## Key Context for Claude

When working with Claude Code on this project, Claude has access to:

- **Architecture**: PySide6 GUI with qasync for async operations
- **Data flow**: POD5 → signal → plotnine → BytesIO → QPixmap → display
- **Patterns**: Async slots with `@qasync.asyncSlot()`, blocking ops via `asyncio.to_thread()`
- **Constraints**: Python 3.8+, 88-char lines, ruff formatting

## Common Claude Code Workflows

### Adding a Feature

```
"Add a new plot normalization method called [name] that [description]"
"Add a menu item to export the current plot as SVG"
"Create a preferences dialog for plot styling options"
```

### Fixing Bugs

```
"Fix the UI freezing issue when loading large POD5 files"
"Debug why the reference browser isn't showing read counts"
"Fix the macOS menu bar showing 'Python' instead of 'Squiggy'"
```

### Refactoring

```
"Refactor the plotting code to separate data preparation from rendering"
"Extract the BAM parsing logic into a separate module"
"Improve error handling in the POD5 file loading code"
```

### Testing

```
"Write tests for the reference browser dialog"
"Add integration tests for the BAM region search feature"
"Create mock fixtures for POD5 file testing"
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

Squiggy is designed to be:
- **Lightweight**: Minimal dependencies, no heavy frameworks
- **Cross-platform**: Works on Windows, macOS, and Linux
- **User-friendly**: Intuitive GUI with good defaults
- **Developer-friendly**: Clear architecture, good documentation
- **AI-assistant ready**: Rich context for Claude Code

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Use the bug report template
- **Features**: Use the feature request template
- **Development**: Check `CLAUDE.md` and the [Development Guide](../docs/development.md)

## License

MIT License - see [LICENSE](../LICENSE) file for details.
