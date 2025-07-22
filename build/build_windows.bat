@echo off
echo MagicSample Demucs - Windows Build Script
echo =========================================
echo.

REM Check if we're in the right directory
if not exist "build_windows.bat" (
    echo Error: Please run this script from the build folder.
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Check if all dependencies are installed
echo Checking dependencies...
python ../test_installation.py
if errorlevel 1 (
    echo Error: Dependencies not installed. Please run install/windows/install_windows.bat first.
    pause
    exit /b 1
)

echo.
echo Building standalone executable...
echo This may take several minutes and create a large file (~500MB-1GB)...

REM Build the executable using python -m PyInstaller
python -m PyInstaller --clean MagicSample.spec

if errorlevel 1 (
    echo.
    echo Build failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo.
echo The executable is located at: dist/MagicSample.exe
echo.
echo Note: The executable file will be large (~500MB-1GB) due to included libraries.
echo.
pause 