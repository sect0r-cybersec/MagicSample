@echo off
echo MagicSample Demucs Launcher
echo ===========================
echo.

REM Check if we're in the right directory
if not exist "MagicSample.py" (
    echo Error: MagicSample.py not found!
    echo Please run this script from the MagicSample folder.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ or later
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
if exist "test_installation.py" (
    python test_installation.py >nul 2>&1
    if errorlevel 1 (
        echo Dependencies test failed. Installing...
        if exist "install_windows.bat" (
            call install_windows.bat
        ) else (
            echo Installation script not found. Please run install_windows.bat manually.
            pause
            exit /b 1
        )
    ) else (
        echo Dependencies check passed.
    )
) else (
    echo Test script not found. Creating basic test...
    python -c "import numpy, scipy, librosa, soundfile, PyQt6, torch, demucs; print('All packages imported successfully!')" >nul 2>&1
    if errorlevel 1 (
        echo Dependencies not found. Installing...
        if exist "install_windows.bat" (
            call install_windows.bat
        ) else (
            echo Installation script not found. Please run install_windows.bat manually.
            pause
            exit /b 1
        )
    )
)

echo.
echo Starting MagicSample Demucs...
python MagicSample.py

pause 