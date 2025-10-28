# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import sys

a = Analysis(
    ['../src/squiggy/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../tests/data/*.pod5', 'squiggy/data'),
        ('../tests/data/README.md', 'squiggy/data'),
        ('squiggy.png', 'squiggy/data'),
        ('squiggy.ico', 'squiggy/data'),
        ('squiggy.icns', 'squiggy/data'),
    ],
    hiddenimports=[
        'pod5',
        'numpy',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'bokeh',
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
    icon='squiggy.ico' if sys.platform == 'win32' else 'squiggy.icns',
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
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
