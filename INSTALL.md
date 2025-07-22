# Installation Guide

## Quick Fix for Numpy Installation Error

If you're getting the numpy installation error, follow these steps:

### Method 1: Use the Windows Installation Script (Recommended)

1. **Navigate to the MagicSample folder** (where this file is located)
2. **Double-click** `install_windows.bat`
3. This script will install packages in the correct order
4. Wait for completion and test with `python test_installation.py`

### Method 2: Manual Installation (Step by Step)

Open Command Prompt, **navigate to the MagicSample folder**, and run these commands **one by one**:

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install numpy first (this often fixes the issue)
pip install numpy>=1.20.0

# 3. Install PyTorch CPU version (more compatible)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install other packages
pip install scipy>=1.7.0
pip install librosa>=0.9.0
pip install soundfile>=0.10.0
pip install PyQt6>=6.0.0
pip install matplotlib>=3.5.0

# 5. Install Demucs last
pip install demucs>=4.0.0
```

### Method 3: Use Conda (Best for Windows)

1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **Navigate to the MagicSample folder**
3. **Double-click** `install_conda.bat`
4. This creates a dedicated environment for the project

### Method 4: If You Still Have Issues

1. **Install Visual Studio Build Tools**:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install with "C++ build tools" selected
   - Restart your computer

2. **Try installing pre-built wheels**:
   ```bash
   pip install --only-binary=all numpy scipy librosa soundfile PyQt6
   ```

## Testing Installation

After installation, run:
```bash
python test_installation.py
```

## Running the Application

Once installation is successful:
```bash
python MagicSample.py
```

## Project Structure

```
MagicSample/
├── MagicSample.py      # Main application
├── requirements.txt          # Dependencies
├── install_windows.bat       # Windows installation script
├── install_conda.bat         # Conda installation script
├── test_installation.py      # Installation test
├── demo_classification.py    # Demo script
├── config.json              # Configuration
├── README.md                # Documentation
├── TROUBLESHOOTING.md       # Troubleshooting guide
├── run.bat                  # Application launcher
├── run.sh                   # Linux/Mac launcher
├── MagicSampleDemucs/       # Additional resources
└── MagicSampleOld/          # Old version files
    └── icons/               # Icon files
```

## Common Issues

- **"Microsoft Visual C++ 14.0 required"**: Install Visual Studio Build Tools
- **"Permission denied"**: Run Command Prompt as Administrator
- **"Package not found"**: Try `pip install --upgrade pip` first
- **"requirements.txt not found"**: Make sure you're in the MagicSample folder

## Need Help?

Check `TROUBLESHOOTING.md` for detailed solutions to common problems. 