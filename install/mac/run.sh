#!/bin/bash

echo "MagicSample Demucs Launcher"
echo "==========================="
echo

# Check if we're in the right directory
if [ ! -f "MagicSample.py" ]; then
    echo "Error: MagicSample.py not found!"
    echo "Please run this script from the MagicSample folder."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ or later"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Check if requirements are installed
echo "Checking dependencies..."
if [ -f "test_installation.py" ]; then
    if python3 test_installation.py >/dev/null 2>&1; then
        echo "Dependencies check passed."
    else
        echo "Dependencies test failed. Installing..."
        if [ -f "install_conda.bat" ]; then
            echo "Please run install_conda.bat on Windows or install dependencies manually."
        else
            echo "Installation script not found. Please install dependencies manually."
            echo "Run: pip install -r requirements.txt"
        fi
        exit 1
    fi
else
    echo "Test script not found. Creating basic test..."
    if python3 -c "import numpy, scipy, librosa, soundfile, PyQt6, torch, demucs; print('All packages imported successfully!')" >/dev/null 2>&1; then
        echo "Dependencies check passed."
    else
        echo "Dependencies not found. Installing..."
        if [ -f "install_conda.bat" ]; then
            echo "Please run install_conda.bat on Windows or install dependencies manually."
        else
            echo "Installation script not found. Please install dependencies manually."
            echo "Run: pip install -r requirements.txt"
        fi
        exit 1
    fi
fi

echo
echo "Starting MagicSample Demucs..."
python3 MagicSample.py 