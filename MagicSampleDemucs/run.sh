#!/bin/bash

echo "MagicSample Demucs Launcher"
echo "==========================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.10 or later"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Check if requirements are installed
echo "Checking dependencies..."
python3 test_installation.py
if [ $? -ne 0 ]; then
    echo
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    echo
    echo "Testing installation again..."
    python3 test_installation.py
fi

echo
echo "Starting MagicSample Demucs..."
python3 MagicSampleDemucs.py 