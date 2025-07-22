@echo off
REM MagicSample Windows Build Script (Robust, No Conda)
REM Can be run from any directory

REM Get the directory of this script
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Check for required files
if not exist "%PROJECT_ROOT%\requirements.txt" (
    echo [ERROR] requirements.txt not found in project root: %PROJECT_ROOT%
    pause
    exit /b 1
)
if not exist "%PROJECT_ROOT%\MagicSample.py" (
    echo [ERROR] MagicSample.py not found in project root: %PROJECT_ROOT%
    pause
    exit /b 1
)

REM Check Python 3.11 version
python3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 is not installed or not in PATH.
    pause
    exit /b 1
)

REM Check pip3.11
python3.11 -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip for Python 3.11 is not installed. Please install pip for Python 3.11.
    pause
    exit /b 1
)

REM Upgrade pip
python3.11 -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip for Python 3.11.
    pause
    exit /b 1
)

REM Check for Visual Studio Build Tools (cl.exe)
where cl >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Visual Studio Build Tools not found. Some packages may fail to build. See TROUBLESHOOTING.md.
)

REM Clean previous build output
if exist "%PROJECT_ROOT%\build\dist" rmdir /s /q "%PROJECT_ROOT%\build\dist"
if exist "%PROJECT_ROOT%\build\build" rmdir /s /q "%PROJECT_ROOT%\build\build"
if exist "%PROJECT_ROOT%\build\MagicSample.spec" del "%PROJECT_ROOT%\build\MagicSample.spec"
if exist "%PROJECT_ROOT%\build\version.txt" del "%PROJECT_ROOT%\build\version.txt"

REM Set absolute path for icon file
set ICONFILE=%PROJECT_ROOT%\build\MagicSample_icon.ico

REM Confirm icon file exists and print its last modified time
if not exist "%ICONFILE%" (
    echo [ERROR] Icon file not found: %ICONFILE%
    pause
    exit /b 1
)
for %%F in ("%ICONFILE%") do (
    echo Using icon: %%F, Last modified: %%~tF
)

REM Install numpy first (fixes common issues)
python3.11 -m pip install numpy<2.0
if errorlevel 1 (
    echo [ERROR] Failed to install numpy for Python 3.11. See TROUBLESHOOTING.md.
    pause
    exit /b 1
)

REM Install PyInstaller if needed
python3.11 -m pip install pyinstaller
if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller for Python 3.11.
    pause
    exit /b 1
)

REM Build with PyInstaller using Python 3.11
cd /d "%PROJECT_ROOT%"
python3.11 -m PyInstaller --onefile --name MagicSample --icon "%ICONFILE%" --distpath build/dist --workpath build/build --specpath build MagicSample.py
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Build completed! The standalone executable is in build/dist.
echo. 