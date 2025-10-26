# Test Data

This directory contains test data files for the Squiggy test suite.

## Files

### POD5 Files

POD5 test files are tracked with Git LFS to avoid repository bloat.

- **sample.pod5** - Small POD5 file with 5-10 reads for basic functionality testing
  - Used by: `test_main.py`, `test_plotting.py`
  - Size: Should be <1MB
  - Contents: Representative nanopore signal data with typical read lengths

## Adding Test Data

### Adding a POD5 File

1. Ensure Git LFS is installed and initialized:
   ```bash
   git lfs install
   ```

2. Add your POD5 file to this directory:
   ```bash
   cp /path/to/your/sample.pod5 tests/data/
   ```

3. Git LFS will automatically track it (configured in `.gitattributes`)

4. Commit and push:
   ```bash
   git add tests/data/sample.pod5
   git commit -m "Add sample POD5 test data"
   git push
   ```

### Creating Minimal Test Files

To create a small POD5 file with just a few reads from a larger file:

```python
import pod5

# Read from large file, write subset to test file
with pod5.Reader("large_file.pod5") as reader:
    with pod5.Writer("tests/data/sample.pod5") as writer:
        for i, read in enumerate(reader.reads()):
            if i >= 10:  # Only take first 10 reads
                break
            writer.add_read(read)
```

## File Requirements

Test files should be:
- **Small**: Keep total test data under 5MB
- **Representative**: Include typical signal characteristics
- **Documented**: Note any special properties in this README
- **Self-contained**: Don't depend on external data sources

## Notes

- POD5 files use HDF5 format with VBZ compression
- Test files are stored with Git LFS (configured in `.gitattributes`)
- When cloning the repo, Git LFS will automatically download these files
- If tests skip due to missing data, ensure Git LFS is installed: `git lfs pull`
