#!/bin/bash
# MagicSample Mac Build Script (Robust, No Conda required)
# Always run this script from the build directory

# Move to project root
dirname=$(dirname "$0")
cd "$dirname/.."

# Check for required files
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found in project root."
    exit 1
fi
if [ ! -f "MagicSample.py" ]; then
    echo "[ERROR] MagicSample.py not found in project root."
    exit 1
fi

# Check Python version (>=3.8)
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi
PYVER=$(python3 -c 'import sys; print(sys.version_info[:2] >= (3,8))')
if [ "$PYVER" != "True" ]; then
    echo "[ERROR] Python 3.8 or higher is required."
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "[ERROR] pip3 is not installed. Please install pip3."
    exit 1
fi

# Upgrade pip
python3 -m pip install --upgrade pip || { echo "[ERROR] Failed to upgrade pip."; exit 1; }

# Install Homebrew dependencies
if ! command -v brew &> /dev/null; then
    echo "[WARNING] Homebrew not found. Skipping system dependency install."
else
    echo "Installing system dependencies via Homebrew..."
    brew install portaudio ffmpeg
fi

# Clean previous build output
rm -rf build/dist build/build build/MagicSample.spec build/version.txt

# Install numpy first (fixes common issues)
pip3 install numpy>=1.20.0 || { echo "[ERROR] Failed to install numpy. See TROUBLESHOOTING.md."; exit 1; }

# Install PyInstaller if needed
pip3 install --user pyinstaller || { echo "[ERROR] Failed to install PyInstaller."; exit 1; }

# Build with PyInstaller
python3 -m PyInstaller --onefile --name MagicSample --distpath build/dist --workpath build/build --specpath build MagicSample.py
if [ $? -ne 0 ]; then
    echo "[ERROR] PyInstaller build failed."
    exit 1
fi

# Return to build directory
cd build

echo
echo "[SUCCESS] Build completed! The standalone executable is in build/dist."
echo 