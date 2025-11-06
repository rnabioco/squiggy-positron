# Review and Update Documentation

Your task is to review and update the Squiggy documentation to ensure it accurately reflects the current codebase.

## Steps to Complete:

### 1. Review Documentation for Accuracy

Read the following documentation files and compare against the current codebase:
- `docs/index.md` - Home page with features list
- `docs/user_guide.md` - Complete user guide
- `docs/quick_reference.md` - Quick reference with API cheat sheet
- `docs/multi_sample_comparison.md` - Multi-sample comparison guide
- `docs/developer.md` - Development and contributing guide
- `docs/api.md` - Python API reference (mkdocstrings)
- `README.md` - Main repository README

For each file, check:
- Are all features mentioned still current?
- Are there new features not documented?
- Are plot modes accurate? (check `squiggy/constants.py`)
- Are normalization methods current? (check `squiggy/constants.py`)
- Are code examples accurate?

**CRITICAL: Keep docs/api.md in sync with squiggy/__init__.py**

1. Read `squiggy/__init__.py` and extract all items in the `__all__` list
2. Read `docs/api.md` and verify every exported item is documented with a mkdocstrings reference like `::: squiggy.function_name`
3. Add missing items, remove deprecated items

### 2. Update Documentation

If you find any discrepancies:
- Update the relevant documentation files
- Add documentation for new features
- Remove or update deprecated information
- Ensure code examples are correct
