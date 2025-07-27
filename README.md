# MagicSample

![Version](https://img.shields.io/badge/version-0.0.8-blue)

---

# MagicSample Demucs Version

An enhanced audio sample extraction and drumkit creation tool that uses the Demucs library for superior stem separation.

## Features

- **Advanced Stem Separation**: Uses Demucs for high-quality separation of drums, bass, vocals, and other instruments
- **BPM Detection**: Automatically detects and labels samples with BPM information
- **Advanced Dominant Frequency Detection**: Multi-algorithm dominant frequency detection with Scientific Pitch Notation (SPN)
- **Drum Classification**: Automatically classifies drum samples into categories (Kick, HiHat, Perc)
- **Sample Similarity Detection**: Prevents duplicate samples with configurable similarity threshold
- **Organized Output**: Creates a structured drumkit folder with subfolders for each instrument type
- **Multiple Output Formats**: Supports WAV, FLAC, and OGG formats
- **User-Friendly GUI**: Modern PyQt6 interface with progress tracking and real-time logging
- **Comprehensive Error Handling**: Graceful error recovery with detailed logging
- **Safe Stop & Cleanup**: Preserves processed samples when stopping mid-process
- **Skip Functionality**: Skip individual samples during processing
- **Sample Timeout Protection**: Configurable timeout to prevent hanging on complex samples
- **YouTube Integration**: Download and process YouTube videos, playlists, and channels directly
- **Hybrid Sample Detection**: Advanced transient detection and energy-based slicing for improved one-shot detection
- **Minimum Amplitude Threshold**: Configurable threshold to filter out unusably quiet samples

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
   - Python version: `python MagicSample.py`
   - Standalone: `./dist/MagicSample` (Linux/macOS) or `dist\MagicSample.exe` (Windows)

## Project Structure

```
MagicSample/
├── MagicSample.py      # Main application
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

1. **Run the application**: `python MagicSample.py`
2. **Select Input Files**: Choose audio files (WAV, MP3, FLAC, OGG, M4A) and/or add YouTube URLs
3. **Choose Output Directory**: Select where to save the drumkit

### YouTube Integration

MagicSample supports processing YouTube content directly:

- **Single Videos**: Add YouTube video URLs to download and process
- **Playlists**: Process entire playlists automatically
- **Channels**: Download videos from YouTube channels
- **Automatic Audio Extraction**: Converts videos to high-quality audio
- **Mixed Processing**: Combine local files and YouTube URLs in the same session
- **Automatic Cleanup**: Removes downloaded files after processing

**Supported YouTube URL Formats:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://www.youtube.com/playlist?list=PLAYLIST_ID`
- `https://www.youtube.com/channel/CHANNEL_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/shorts/VIDEO_ID`
4. **Configure Options**:
   - **Split to stems**: Separate into drums, bass, vocals, other
   - **Detect BPM**: Automatically detect and label BPM
   - **Detect Dominant Frequency**: Estimate and label the most prominent frequency
   - **Classify drums**: Automatically categorize drum samples
   - **Sample Detection Sensitivity**: Adjust sample detection sensitivity (5-30)
   - **Sample Similarity Threshold**: Control duplicate detection (0-100%)
   - **Sample Timeout**: Maximum processing time per sample (milliseconds)
   - **Min Amplitude (dBFS)**: Minimum amplitude threshold to filter out quiet samples
   - **Output Format**: Choose WAV, FLAC, or OGG
   - **Drumkit Name**: Name for your drumkit folder
5. **Start Processing**: Click "Start Processing" and monitor progress
6. **Use Controls**: 
   - **Skip Sample**: Skip current sample and continue processing
   - **Stop**: Safely stop processing and save completed samples
   - **Log Tab**: Monitor real-time processing information
   - **Help Tab**: Access comprehensive documentation

## Output Structure

The tool creates a drumkit folder with the following structure:

```
MyDrumkit/
├── metadata.json          # Processing information and settings
├── Drums/                 # Drum samples (split into subfolders)
│   ├── Kick/             # Kick drum samples (low frequency: <200 Hz)
│   ├── HiHat/            # Hi-hat samples (high frequency: >2000 Hz)
│   └── Perc/             # Percussion samples (mid frequency: 200-2000 Hz)
├── Bass/                  # Bass samples (individual samples)
├── Vocals/                # Vocal stem (whole acapella file)
└── Other/                 # Other instrument samples (individual samples)
```

## Sample Naming Convention

Samples are automatically named with the following format:
- `{Type}_{number}_{BPM}BPM_{Pitch}.{Format}`

Examples:
- `Kick_001_120BPM_C2.WAV`
- `HiHat_002_120BPM_G#4.WAV`
- `Bass_001_120BPM_F1.WAV`
- `Vocals_120BPM.WAV` (whole acapella file)
- `Other_001_120BPM_A3.WAV`

**Note**: Dominant frequency information uses Scientific Pitch Notation (SPN) and is only included when dominant frequency detection is enabled and successful.

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
- **yt-dlp**: YouTube video downloading
- **scipy**: Scientific computing

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Changelog

### Version 0.0.6 (Latest)
- **Comprehensive Exception Handling**: Added extensive error handling throughout the application
- **Safe Stop & Cleanup**: Implemented proper cleanup when stopping processing mid-way
- **Skip Sample Button**: Added ability to skip individual samples during processing
- **Real-time Logging**: Enhanced logging system with dedicated Log tab
- **Help Documentation**: Added comprehensive Help tab with detailed parameter explanations
- **Sample Timeout Protection**: Configurable timeout to prevent hanging on complex samples
- **Sample Similarity Detection**: Prevents duplicate samples with configurable threshold
- **Advanced Dominant Frequency Detection**: Multi-algorithm dominant frequency detection with Scientific Pitch Notation
- **Improved UI**: Better button management and user feedback

### Version 0.0.5
- **Comprehensive Exception Handling**: Added extensive error handling throughout the application
- **Safe Stop & Cleanup**: Implemented proper cleanup when stopping processing mid-way
- **Skip Sample Button**: Added ability to skip individual samples during processing
- **Real-time Logging**: Enhanced logging system with dedicated Log tab
- **Help Documentation**: Added comprehensive Help tab with detailed parameter explanations
- **Sample Timeout Protection**: Configurable timeout to prevent hanging on complex samples
- **Sample Similarity Detection**: Prevents duplicate samples with configurable threshold
- **Advanced Pitch Detection**: Multi-algorithm pitch detection with Scientific Pitch Notation
- **Improved UI**: Better button management and user feedback

### Version 0.0.4
- Fixed Demucs model loading issues
- Improved stem separation reliability
- Enhanced error reporting and logging

### Version 0.0.3
- Added sample similarity comparison within categories
- Implemented custom multi-algorithm pitch detection
- Added sample timeout functionality

### Version 0.0.2 (Demucs)
- Replaced Spleeter with Demucs for better separation quality
- Added drum classification system
- Enhanced pitch detection
- Improved GUI with progress tracking
- Added metadata generation
- Better file organization structure 