# MagicSample

![Version](https://img.shields.io/badge/version-0.0.1-blue)

---

# MagicSample Demucs Version

An enhanced audio sample extraction and drumkit creation tool that uses the Demucs library for superior stem separation.

## Features

- **Advanced Stem Separation**: Uses Demucs for high-quality separation of drums, bass, vocals, and other instruments
- **BPM Detection**: Automatically detects and labels samples with BPM information
- **Pitch Detection**: Estimates and labels samples with musical pitch information
- **Drum Classification**: Automatically classifies drum samples into categories (hi-hat, snare, kick, tom, etc.)
- **Organized Output**: Creates a structured drumkit folder with subfolders for each instrument type
- **Multiple Output Formats**: Supports WAV, FLAC, and OGG formats
- **User-Friendly GUI**: Modern PyQt6 interface with progress tracking

## Installation

### Option 1: Python Installation (Recommended for Development)

Choose your platform and run the appropriate installer:

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

### Option 2: Standalone Executable (No Python Required)

Build a standalone executable using PyInstaller:

**Windows:**
```bash
cd build
build_windows.bat
```

**Linux/macOS:**
```bash
cd build
./build_linux.sh
```

**Note:** Standalone executables are large (~500MB-1GB) but don't require Python installation.

### Quick Start

1. **Install dependencies** using one of the methods above
2. **Run the application:**
   - Python version: `python MagicSampleDemucs.py`
   - Standalone: `./dist/MagicSample` (Linux/macOS) or `dist\MagicSample.exe` (Windows)

## Project Structure

```
MagicSample/
├── MagicSampleDemucs.py      # Main application
├── requirements.txt          # Python dependencies
├── config.json              # Application configuration
├── test_installation.py     # Dependency checker
├── README.md                # This file
├── INSTALL.md               # Detailed installation guide
├── TROUBLESHOOTING.md       # Problem-solving guide
├── install/                 # Installation scripts by platform
│   ├── windows/            # Windows installers
│   ├── linux/              # Linux installers
│   └── mac/                # macOS installers
├── build/                  # PyInstaller build scripts
│   ├── MagicSample.spec    # PyInstaller configuration
│   ├── build_windows.bat   # Windows build script
│   └── build_linux.sh      # Linux/macOS build script
└── MagicSampleOld/         # Reference material (original version)
```

## Usage

1. **Run the application**: `python MagicSampleDemucs.py`
2. **Select Input File**: Choose an audio file (WAV, MP3, FLAC, OGG, M4A)
3. **Choose Output Directory**: Select where to save the drumkit
4. **Configure Options**:
   - **Split to stems**: Separate into drums, bass, vocals, other
   - **Detect BPM**: Automatically detect and label BPM
   - **Detect pitch**: Estimate and label musical pitch
   - **Classify drums**: Automatically categorize drum samples
   - **Sensitivity**: Adjust sample detection sensitivity (5-30 dB)
   - **Output Format**: Choose WAV, FLAC, or OGG
   - **Drumkit Name**: Name for your drumkit folder
5. **Start Processing**: Click "Start Processing" and wait for completion

## Output Structure

The tool creates a drumkit folder with the following structure:

```
MyDrumkit/
├── metadata.json          # Processing information and settings
├── DRUMS/                 # Drum samples (split into subfolders)
│   ├── KICKS/            # Kick drum samples (low frequency: 40-250 Hz)
│   ├── SNARES/           # Snare drum samples (broad range: 120 Hz-10 kHz)
│   ├── CLAPS/            # Clap samples (mid-high frequency: 800 Hz-10 kHz)
│   └── HI_HATS/          # Hi-hat samples (high frequency: 5-10 kHz)
├── BASS/                  # Bass samples (individual samples)
├── VOCALS/                # Vocal stem (whole acapella file)
└── OTHER/                 # Other instrument samples (individual samples)
```

## Sample Naming Convention

Samples are automatically named with the following format:
- `{INSTRUMENT}_{number}_{BPM}BPM_{pitch}.{FORMAT}`

Examples:
- `KICK_001_120BPM_C4.WAV`
- `HIHAT_002_120BPM_N/A.WAV`
- `SNARE_003_120BPM_A2.WAV`
- `CLAP_001_120BPM_N/A.WAV`
- `BASS_001_120BPM_A2.WAV`
- `VOCALS_120BPM.WAV` (whole acapella file)

## Installation Issues?

If you encounter installation problems:

1. **Check** `INSTALL.md` for detailed installation instructions
2. **Check** `TROUBLESHOOTING.md` for solutions to common problems
3. **Try the conda method**: Run `install_conda.bat` (requires Anaconda/Miniconda)

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

## Dependencies

- **librosa**: Audio analysis and processing
- **demucs**: Stem separation
- **PyQt6**: GUI framework
- **soundfile**: Audio file I/O
- **numpy**: Numerical computing
- **torch**: Deep learning framework
- **scipy**: Scientific computing

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Changelog

### Version 2.0 (Demucs)
- Replaced Spleeter with Demucs for better separation quality
- Added drum classification system
- Enhanced pitch detection
- Improved GUI with progress tracking
- Added metadata generation
- Better file organization structure 