# Installation

## Standalone Executables (Recommended)

Download the pre-built executable for your platform from the [releases page](https://github.com/rnabioco/squiggler/releases):

### Windows
1. Download `Squiggy-windows-x64.zip`
2. Extract the ZIP file
3. Run `Squiggy.exe`

### macOS
1. Download `Squiggy-macos.dmg`
2. Open the DMG file
3. Drag Squiggy to your Applications folder
4. Launch from Applications (you may need to allow the app in System Preferences → Security)

### Linux
1. Download `Squiggy-linux-x64.tar.gz`
2. Extract: `tar -xzf Squiggy-linux-x64.tar.gz`
3. Run: `./Squiggy`

## Install from Source

If you want to run or develop Squiggy from source:

### Prerequisites
- Python 3.8 or later
- pip or uv package manager

### Steps

```bash
# Clone the repository
git clone https://github.com/rnabioco/squiggler.git
cd squiggler

# Install dependencies
pip install -r requirements.txt

# Run the application
python squiggy/squiggy/main.py
```

### Development Installation

For development with editable installation:

```bash
pip install -e ".[dev]"
```

## Building from Source

To build standalone executables yourself:

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller squiggy.spec

# Output will be in dist/
# - dist/Squiggy (Linux)
# - dist/Squiggy.exe (Windows)
# - dist/Squiggy.app (macOS)
```
