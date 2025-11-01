# Compile Extension

Re-generate the .vsix extension package for the Squiggy Positron extension.

## Steps

1. Clean any previous builds
2. Compile TypeScript to JavaScript (using webpack for bundling)
3. Package the extension as a .vsix file

## Commands

```bash
# Clean previous builds
rm -f *.vsix

# Compile TypeScript (uses webpack to create extension.js + webview.js bundles)
npm run compile

# Package extension
npm run package
```

**Note:** The compile step uses webpack to create two bundles:
- `build/extension.js` - Node.js bundle for extension host
- `build/webview.js` - Browser bundle for React UI components

The resulting .vsix file will be created in the `build/` directory and can be installed in Positron via:
- Extensions → ... → Install from VSIX...
