@echo off
echo MagicSample Demucs - Windows Installation Script
echo ================================================
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo Error: requirements.txt not found!
    echo Please run this script from the MagicSample folder.
    pause
    exit /b 1
)

if not exist "MagicSampleDemucs.py" (
    echo Error: MagicSampleDemucs.py not found!
    echo Please run this script from the MagicSample folder.
    pause
    exit /b 1
)

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking version...
python -c "import sys; print(f'Python {sys.version}')"

REM Upgrade pip first
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install numpy first (often causes issues)
echo.
echo Installing numpy first (this fixes the common installation error)...
pip install numpy>=1.20.0
if errorlevel 1 (
    echo.
    echo Numpy installation failed. Trying alternative method...
    pip install --only-binary=all numpy>=1.20.0
    if errorlevel 1 (
        echo.
        echo Numpy installation still failed. Please try the conda method.
        echo Run: install_conda.bat
        pause
        exit /b 1
    )
)

REM Install PyTorch with CPU support (more compatible)
echo.
echo Installing PyTorch (CPU version)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo PyTorch installation failed. Trying alternative method...
    pip install torch torchaudio --only-binary=all
)

REM Install other dependencies one by one
echo.
echo Installing other dependencies...
pip install scipy>=1.7.0
pip install librosa>=0.9.0
pip install soundfile>=0.10.0
pip install PyQt6>=6.0.0
pip install matplotlib>=3.5.0

REM Install Demucs last
echo.
echo Installing Demucs...
pip install demucs>=4.0.0

echo.
echo Installation completed!
echo.
echo Testing installation...
if exist "test_installation.py" (
    python test_installation.py
) else (
    echo Test script not found. Creating basic test...
    python -c "import numpy, scipy, librosa, soundfile, PyQt6, torch, demucs; print('All packages imported successfully!')"
)

echo.
echo If you see any errors above, try the following:
echo 1. Install Visual Studio Build Tools
echo 2. Use conda instead of pip (run install_conda.bat)
echo 3. Try installing packages one by one manually
echo.
echo To run the application:
echo python MagicSampleDemucs.py
echo.
pause 