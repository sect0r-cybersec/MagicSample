# MagicSample Build Instructions

## Overview

This directory contains scripts to build standalone executables using PyInstaller. The executables will include all dependencies and can run without requiring Python installation.

## ⚠️ Important Notes

### File Size
- **Expected size**: 500MB - 1GB
- **Reason**: Includes PyTorch, Demucs, and all audio processing libraries
- **Trade-off**: Large file size vs. no Python installation required

### Platform Limitations
- **Windows**: Build on Windows for Windows executable
- **Linux**: Build on Linux for Linux executable  
- **macOS**: Build on macOS for macOS executable
- **Cross-compilation**: Not supported by PyInstaller

## Prerequisites

### 1. Install Dependencies
Before building, ensure all dependencies are installed:

**Windows:**
```bash
cd install/windows
install_windows.bat
```

**Linux:**
```bash
cd install/linux
./install_linux.sh
```

**macOS:**
```bash
cd install/mac
./install_mac.sh
```

### 2. Install PyInstaller
```bash
pip install pyinstaller
```

## Building Executables

### Windows
```bash
cd build
build_windows.bat
```

### Linux
```bash
cd build
./build_linux.sh
```

### macOS
```bash
cd build
./build_linux.sh  # Uses the same script
```

## Output

After successful build:
- **Windows**: `dist/MagicSample.exe`
- **Linux**: `dist/MagicSample`
- **macOS**: `dist/MagicSample`

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Run the appropriate install script first
   - Ensure all packages are installed correctly

2. **Large File Size**
   - This is normal due to PyTorch and Demucs
   - Consider using the Python version for development

3. **Audio Library Issues**
   - Some audio libraries may not work in frozen executables
   - Test thoroughly with your target audio files

4. **PyInstaller Errors**
   - Check that all hidden imports are included in the spec file
   - Try building with `--debug=all` for more information

### Debug Mode

To build with console output for debugging:
1. Edit `MagicSample.spec`
2. Change `console=False` to `console=True`
3. Rebuild the executable

## Distribution

### What to Include
- The executable file
- Any required data files (config.json, icons)
- README with usage instructions

### What NOT to Include
- Python installation
- Source code (unless desired)
- Development tools

## Performance Considerations

### Pros of Standalone Executable
- No Python installation required
- Easier distribution
- Consistent environment

### Cons of Standalone Executable
- Large file size
- Slower startup time
- Potential compatibility issues
- Harder to debug

## Alternative: Python Distribution

For users who prefer the Python version:
1. Distribute the source code
2. Include installation scripts
3. Smaller download size
4. Easier updates

## Recommendations

### For End Users
- **Standalone executable**: If they don't have Python
- **Python version**: If they're comfortable with Python

### For Development
- **Python version**: Faster iteration and debugging
- **Standalone**: For testing distribution

### For Distribution
- **Both options**: Provide both Python and standalone versions
- **Documentation**: Clear instructions for both approaches 