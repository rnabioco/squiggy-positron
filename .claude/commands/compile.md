# Compile Extension

Re-generate the .vsix extension package for the Squiggy Positron extension.

## Steps

1. Clean any previous builds
2. Compile TypeScript to JavaScript
3. Package the extension as a .vsix file

## Commands

```bash
# Clean previous builds
rm -f *.vsix

# Compile TypeScript
npm run compile

# Package extension
npm run package
```

The resulting .vsix file will be created in the project root and can be installed in Positron via:
- Extensions → ... → Install from VSIX...
