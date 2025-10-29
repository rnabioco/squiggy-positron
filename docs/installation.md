# Installation

## Standalone Application (Recommended)

Download the pre-built macOS application from the [releases page](https://github.com/rnabioco/squiggy/releases):

### macOS
1. Download `Squiggy-macos.dmg`
2. Open the DMG file
3. Drag Squiggy to your Applications folder
4. Launch from Applications (you may need to allow the app in System Preferences → Security & Privacy)

## Install from Source

If you want to run or develop Squiggy from source:

### Prerequisites
- Python 3.8 or later
- pip or uv package manager

### Steps

```bash
# Clone the repository
git clone https://github.com/rnabioco/squiggy.git
cd squiggy

# Install dependencies
pip install -e ".[dev]"

# Run the application
squiggy
```

### Development Installation

For development with editable installation:

```bash
pip install -e ".[dev]"
```

### Optional: Export Dependencies

**For standalone builds** (downloaded from releases):
- PNG and SVG export work out of the box - no additional setup required!

**For source installations** (installed via pip/uv):
- HTML export works immediately
- PNG and SVG export require additional dependencies:

```bash
# Install export dependencies
pip install -e ".[export]"

# Or with uv
uv pip install -e ".[export]"
```

This installs `selenium` and `pillow`, which enable Bokeh's image export functionality. Firefox (or another browser) must be available on your system, which selenium will use for headless rendering.

**Note:** Most users don't need these dependencies unless they specifically want to export PNG or SVG images from source installations. Standalone builds include everything needed for all export formats.

## Building from Source

To build the macOS application yourself:

```bash
# Install development dependencies (includes PyInstaller)
uv pip install -e ".[dev]"

# Build from the build directory
cd build
pyinstaller squiggy.spec

# Output will be in dist/Squiggy.app

# Optional: Create a DMG installer
brew install create-dmg
create-dmg \
  --volname "Squiggy Installer" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --app-drop-link 600 185 \
  Squiggy-macos.dmg \
  dist/Squiggy.app
```
