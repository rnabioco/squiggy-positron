#!/usr/bin/env bash
# Setup test workspace for extension development
# This creates the test-workspace directory structure on the fly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Use simple test-workspace directory
TEST_WORKSPACE="$PROJECT_ROOT/test-workspace"

# Only create if it doesn't exist
if [ -d "$TEST_WORKSPACE" ]; then
    echo "Test workspace already exists at: $TEST_WORKSPACE"
    exit 0
fi

echo "Setting up test workspace at: $TEST_WORKSPACE"

# Create directory structure
mkdir -p "$TEST_WORKSPACE/.vscode"
mkdir -p "$TEST_WORKSPACE/sample-data"

# Create .vscode/settings.json with absolute path to pixi Python
PIXI_PYTHON="$PROJECT_ROOT/.pixi/envs/default/bin/python"
cat > "$TEST_WORKSPACE/.vscode/settings.json" <<EOF
{
    "python.defaultInterpreterPath": "$PIXI_PYTHON",
    "positron.interpreters.automaticStartup": true,
    "positron.interpreters.preferredRuntime": "Python",
    "positron.r.session.automaticStartup": false
}
EOF

# Create README.md
cat > "$TEST_WORKSPACE/README.md" <<'EOF'
# Squiggy Test Workspace

This workspace is automatically opened when debugging the Squiggy extension (F5 in development).

**Note**: This directory is generated on the fly by `scripts/setup-test-workspace.sh` and is not tracked in git.

## Python Interpreter

Configured to use: `../.pixi/envs/default/bin/python`

The interpreter path is relative to this workspace folder, pointing to the pixi environment in the parent directory (extension development workspace).

## Sample Data

Place test POD5/BAM files in `sample-data/` directory:

```bash
# Option 1: Symlink to test data
ln -s ../../squiggy/data/yeast_trna_reads.pod5 sample-data/
ln -s ../../squiggy/data/yeast_trna_mappings.bam sample-data/

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
- Sample data files are not tracked in git
- This directory is regenerated automatically when you press F5
EOF

echo "✓ Test workspace created successfully at: test-workspace"
