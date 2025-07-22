#!/bin/bash

echo "MagicSample Demucs - Linux Installation Script"
echo "=============================================="
echo

# Check if we're in the right directory
if [ ! -f "../../requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    echo "Please run this script from the install/linux folder."
    exit 1
fi

if [ ! -f "../../MagicSampleDemucs.py" ]; then
    echo "Error: MagicSampleDemucs.py not found!"
    echo "Please run this script from the install/linux folder."
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ or later"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

echo "Python found. Checking version..."
python3 --version

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed"
    echo "Please install pip3:"
    echo "Ubuntu/Debian: sudo apt install python3-pip"
    echo "Fedora: sudo dnf install python3-pip"
    exit 1
fi

# Upgrade pip
echo
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install system dependencies (if needed)
echo
echo "Installing system dependencies..."
if command -v apt &> /dev/null; then
    # Ubuntu/Debian
    sudo apt update
    sudo apt install -y build-essential libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0
elif command -v dnf &> /dev/null; then
    # Fedora
    sudo dnf install -y gcc gcc-c++ alsa-lib-devel portaudio-devel
elif command -v pacman &> /dev/null; then
    # Arch Linux
    sudo pacman -S --needed base-devel portaudio
fi

# Install Python packages
echo
echo "Installing Python packages..."

# Install numpy first
echo "Installing numpy..."
pip3 install numpy>=1.20.0

# Install PyTorch (CPU version for better compatibility)
echo "Installing PyTorch (CPU version)..."
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip3 install scipy>=1.7.0
pip3 install librosa>=0.9.0
pip3 install soundfile>=0.10.0
pip3 install PyQt6>=6.0.0
pip3 install matplotlib>=3.5.0

# Install Demucs last
echo "Installing Demucs..."
pip3 install demucs>=4.0.0

echo
echo "Installation completed!"
echo
echo "Testing installation..."
cd ../..
if [ -f "test_installation.py" ]; then
    python3 test_installation.py
else
    echo "Test script not found. Creating basic test..."
    python3 -c "import numpy, scipy, librosa, soundfile, PyQt6, torch, demucs; print('All packages imported successfully!')"
fi

echo
echo "If you see any errors above, try the following:"
echo "1. Install development tools: sudo apt install build-essential"
echo "2. Install audio libraries: sudo apt install libasound2-dev portaudio19-dev"
echo "3. Try installing packages one by one manually"
echo
echo "To run the application:"
echo "cd ../.. && python3 MagicSampleDemucs.py"
echo 