#!/bin/bash

echo "MagicSample Demucs - Linux Build Script"
echo "======================================="
echo

# Check if we're in the right directory
if [ ! -f "build_linux.sh" ]; then
    echo "Error: Please run this script from the build folder."
    exit 1
fi

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" &> /dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Check if all dependencies are installed
echo "Checking dependencies..."
python3 ../test_installation.py
if [ $? -ne 0 ]; then
    echo "Error: Dependencies not installed. Please run install/linux/install_linux.sh first."
    exit 1
fi

echo
echo "Building standalone executable..."
echo "This may take several minutes and create a large file (~500MB-1GB)..."

# Build the executable using python -m PyInstaller
python3 -m PyInstaller --clean MagicSample.spec

if [ $? -ne 0 ]; then
    echo
    echo "Build failed! Check the error messages above."
    exit 1
fi

echo
echo "Build completed successfully!"
echo
echo "The executable is located at: dist/MagicSample"
echo
echo "Note: The executable file will be large (~500MB-1GB) due to included libraries."
echo 