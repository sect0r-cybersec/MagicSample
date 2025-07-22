# Troubleshooting Guide

## Common Installation Issues

### 1. Numpy Installation Error

**Error**: `numpy_f1dbe557d9574369b472e95498e8cb35\.mesonpy-1mo7ooyy\meson-logs\meson-log.txt`

**Solutions**:

#### Option A: Use the Windows Installation Script
```bash
# Run the provided Windows installation script
install_windows.bat
```

#### Option B: Install Numpy First
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install numpy separately first
pip install numpy>=1.20.0

# Then install other requirements
pip install -r requirements.txt
```

#### Option C: Use Conda (Recommended for Windows)
```bash
# Run the conda installation script
install_conda.bat
```

### 2. Visual Studio Build Tools Required

**Error**: `Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
1. Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. During installation, select "C++ build tools"
3. Restart your computer
4. Try installation again

### 3. PyTorch Installation Issues

**Error**: PyTorch installation fails

**Solutions**:

#### Option A: Install CPU-only PyTorch
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Option B: Use Conda for PyTorch
```bash
conda install pytorch torchaudio cpuonly -c pytorch
```

### 4. Demucs Installation Issues

**Error**: Demucs installation fails

**Solutions**:

#### Option A: Install Dependencies First
```bash
# Install PyTorch first
pip install torch torchaudio

# Then install Demucs
pip install demucs
```

#### Option B: Use Specific Version
```bash
pip install demucs==4.0.0
```

### 5. PyQt6 Installation Issues

**Error**: PyQt6 installation fails

**Solutions**:

#### Option A: Use Conda
```bash
conda install -c conda-forge pyqt
```

#### Option B: Install Pre-built Wheels
```bash
pip install PyQt6 --only-binary=all
```

## Alternative Installation Methods

### Method 1: Conda Environment (Recommended)

1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Run the conda installation script:
   ```bash
   install_conda.bat
   ```

### Method 2: Virtual Environment

```bash
# Create virtual environment
python -m venv magicsample_env

# Activate environment
# Windows:
magicsample_env\Scripts\activate
# Linux/Mac:
source magicsample_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Method 3: Manual Installation

Install packages one by one in this order:

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install numpy
pip install numpy>=1.20.0

# 3. Install PyTorch
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

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space
- Windows 10/11, macOS 10.14+, or Linux

### Recommended Requirements
- Python 3.10
- 8GB RAM
- 4GB free disk space
- CUDA-compatible GPU (for faster processing)

## Testing Installation

After installation, run the test script:

```bash
python test_installation.py
```

This will verify that all components are working correctly.

## Common Runtime Issues

### 1. CUDA Out of Memory
**Solution**: The application will automatically fall back to CPU processing if CUDA is not available or runs out of memory.

### 2. Audio File Format Issues
**Solution**: Ensure your audio files are in supported formats (WAV, MP3, FLAC, OGG, M4A).

### 3. Slow Processing
**Solutions**:
- Use shorter audio files for testing
- Close other applications to free up memory
- Use WAV files for best performance
- Ensure you have enough free disk space

## Getting Help

If you continue to have issues:

1. Check the error messages carefully
2. Try the alternative installation methods above
3. Ensure you have the latest Python version
4. Consider using Anaconda/Miniconda for easier package management

## Known Issues

- **Windows**: Some packages may require Visual Studio Build Tools
- **macOS**: May need to install Xcode Command Line Tools
- **Linux**: May need to install development libraries

For these issues, follow the specific solutions provided above. 