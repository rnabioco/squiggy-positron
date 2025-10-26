# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../src/squiggy/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../src/squiggy/data/*.pod5', 'squiggy/data'),
        ('../src/squiggy/data/README.md', 'squiggy/data'),
        ('../src/squiggy/data/squiggy.png', 'squiggy/data'),
        ('../src/squiggy/data/squiggy.ico', 'squiggy/data'),
        ('../src/squiggy/data/squiggy.icns', 'squiggy/data'),
        # Also bundle icons at root level for easy access
        ('squiggy.png', '.'),
        ('squiggy.ico', '.'),
        ('squiggy.icns', '.'),
    ],
    hiddenimports=[
        'pod5',
        'plotnine',
        'pandas',
        'numpy',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'importlib.resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Squiggy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='squiggy.icns',  # macOS and Linux will use this
)

# For macOS, create an app bundle
app = BUNDLE(
    exe,
    name='Squiggy.app',
    icon='squiggy.icns',
    bundle_identifier='com.rnabioco.squiggy',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleDisplayName': 'Squiggy',
        'CFBundleName': 'Squiggy',
        'CFBundleShortVersionString': '0.1.0',
    },
)
