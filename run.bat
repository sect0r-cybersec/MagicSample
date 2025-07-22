@echo off
echo MagicSample Demucs Launcher
echo ===========================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10 or later
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python test_installation.py
if errorlevel 1 (
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
    echo Testing installation again...
    python test_installation.py
)

echo.
echo Starting MagicSample Demucs...
python MagicSampleDemucs.py

pause 