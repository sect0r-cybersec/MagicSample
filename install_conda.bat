@echo off
echo MagicSample Demucs - Conda Installation Script
echo ===============================================
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

REM Check if conda is available
conda --version >nul 2>&1
if errorlevel 1 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda from https://docs.conda.io
    pause
    exit /b 1
)

echo Conda found. Creating environment...
echo.

REM Create new conda environment
conda create -n magicsample python=3.10 -y

REM Activate environment
call conda activate magicsample

REM Install packages via conda (more stable on Windows)
echo Installing packages via conda...
conda install -c conda-forge numpy scipy librosa soundfile matplotlib -y
conda install -c conda-forge pyqt -y

REM Install PyTorch via conda
conda install pytorch torchaudio cpuonly -c pytorch -y

REM Install Demucs via pip (not available in conda)
echo Installing Demucs via pip...
pip install demucs

echo.
echo Installation completed!
echo.
echo To activate the environment in the future, run:
echo conda activate magicsample
echo.
echo Testing installation...
if exist "test_installation.py" (
    python test_installation.py
) else (
    echo Test script not found. Creating basic test...
    python -c "import numpy, scipy, librosa, soundfile, PyQt6, torch, demucs; print('All packages imported successfully!')"
)

echo.
echo If everything looks good, you can now run:
echo python MagicSampleDemucs.py
echo.
pause 