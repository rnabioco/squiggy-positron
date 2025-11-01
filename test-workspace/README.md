# Squiggy Test Workspace

This workspace is automatically opened when debugging the Squiggy extension (F5 in development).

## Python Interpreter

Configured to use: `../.pixi/envs/default/bin/python`

The interpreter path is relative to this workspace folder, pointing to the pixi environment in the parent directory (extension development workspace).

## Sample Data

Place test POD5/BAM files in `sample-data/` directory:

```bash
# Option 1: Symlink to test data
ln -s ../../tests/data/yeast_trna_reads.pod5 sample-data/
ln -s ../../tests/data/yeast_trna_mappings.bam sample-data/

# Option 2: Copy your own test files
cp /path/to/your/data.pod5 sample-data/
```

## Testing Workflow

1. Open Positron in extension workspace (parent directory)
2. Press F5 to launch Extension Development Host
3. This workspace opens automatically with pixi interpreter selected
4. Test extension features with sample data

## Notes

- This workspace is for extension testing only
- Sample data files are gitignored (add via symlink or copy manually)
- Settings are in `.vscode/settings.json`
