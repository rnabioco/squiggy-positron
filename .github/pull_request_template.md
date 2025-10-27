# Pull Request

## Description

Brief description of what this PR does.

Fixes #(issue number)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Changes Made

List the main changes in this PR:

- Change 1
- Change 2
- Change 3

## Testing

Describe how you tested these changes:

- [ ] Tested manually with sample data (`tests/data/simplex_reads.pod5`)
- [ ] Tested with custom POD5 files
- [ ] Tested with BAM files
- [ ] All existing tests pass (`pytest tests/`)
- [ ] Added new tests for new functionality

### Test Commands Run

```bash
# Example test commands
pytest tests/
squiggy -p tests/data/simplex_reads.pod5
```

## Screenshots (if applicable)

Add screenshots demonstrating UI changes or new functionality.

## Checklist

Before submitting this PR, please ensure:

- [ ] Code follows the project's style guidelines (`ruff format` and `ruff check`)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code where necessary, especially in hard-to-understand areas
- [ ] I have made corresponding changes to documentation (if applicable)
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## For Reviewers

### Key Files Changed

List the main files modified and what changed:

- `src/squiggy/viewer.py` - Added new dialog for X
- `src/squiggy/plotter.py` - Implemented Y plotting mode
- `tests/test_plotting.py` - Added tests for Y

### Areas for Special Review

Highlight any areas that need extra attention:

- [ ] Async/Qt patterns (check `@qasync.asyncSlot()` usage)
- [ ] Error handling for edge cases
- [ ] Performance with large files
- [ ] Cross-platform compatibility

## Additional Notes

Any additional information or context for reviewers.

---

## For Claude Code Users

If this PR was developed with Claude Code assistance:
- Confirm that async patterns follow `CLAUDE.md` guidelines
- Verify type hints are Python 3.8+ compatible
- Check that blocking operations use `asyncio.to_thread()`
