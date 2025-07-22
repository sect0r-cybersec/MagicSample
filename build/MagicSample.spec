# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\MagicSample.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\harri\\Documents\\Cursor\\Magic sample\\MagicSample\\build\\..\\config.json', '.'), ('C:\\Users\\harri\\Documents\\Cursor\\Magic sample\\MagicSample\\build\\..\\venv\\Lib\\site-packages\\demucs\\remote\\files.txt', 'demucs/remote'), ('C:\\Users\\harri\\.cache\\torch\\hub\\checkpoints\\955717e8-8726e21a.th', 'torch/hub/checkpoints')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MagicSample',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\harri\\Documents\\Cursor\\Magic sample\\MagicSample\\build\\MagicSample_icon.ico'],
)
