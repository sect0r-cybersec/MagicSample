# -*- mode: python ; coding: utf-8 -*-

# MagicSample version: 0.0.1

block_cipher = None

a = Analysis(
    ['../MagicSampleDemucs.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../config.json', '.'),
        ('../MagicSampleOld/icons/octopus.png', 'icons'),
    ],
    hiddenimports=[
        # Core
        'librosa', 'librosa.core', 'librosa.beat', 'librosa.effects', 'librosa.feature',
        'soundfile', 'numpy', 'scipy', 'scipy.signal',
        'torch', 'torchaudio',
        'demucs', 'demucs.pretrained', 'demucs.apply', 'demucs.audio',
        'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtGui', 'PyQt6.QtCore',
        'matplotlib', 'matplotlib.backends.backend_qt5agg',
        # sklearn Cython modules (for librosa.decompose)
        'sklearn.utils._cython_blas',
        'sklearn.utils._isfinite',
        'sklearn._cyutility',
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
    name='MagicSample',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='MagicSample_icon.ico',
) 