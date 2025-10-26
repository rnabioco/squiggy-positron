# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['squiggy/squiggy/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('squiggy/squiggy/data/*.pod5', 'squiggy/data'),
        ('squiggy/squiggy/data/README.md', 'squiggy/data'),
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
    icon=None,  # Add icon='icon.ico' if you have one
)

# For macOS, create an app bundle
app = BUNDLE(
    exe,
    name='Squiggy.app',
    icon=None,
    bundle_identifier='com.yourname.squiggy',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
    },
)
