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

## Quick Start

### Windows Users
1. **Double-click** `install_windows.bat` to install dependencies
2. **Double-click** `run.bat` to start the application

### Linux/Mac Users
1. Run `./run.sh` to install dependencies and start the application

### Manual Installation
1. Install Python 3.8+ from [python.org](https://python.org)
2. Run `pip install -r requirements.txt`
3. Run `python MagicSampleDemucs.py`

## Project Structure

```
MagicSample/
├── MagicSampleDemucs.py      # Main application
├── requirements.txt          # Dependencies
├── install_windows.bat       # Windows installation script
├── install_conda.bat         # Conda installation script
├── test_installation.py      # Installation test
├── demo_classification.py    # Demo script
├── config.json              # Configuration
├── README.md                # This file
├── INSTALL.md               # Detailed installation guide
├── TROUBLESHOOTING.md       # Troubleshooting guide
├── run.bat                  # Windows launcher
├── run.sh                   # Linux/Mac launcher
└── MagicSampleOld/          # Old version files
    └── icons/               # Icon files
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
├── drums/                 # Drum samples (split into subfolders)
│   ├── kick/             # Kick drum samples (low frequency)
│   ├── percs/            # Percussion samples (mid frequency: snares, toms, etc.)
│   └── hihats/           # Hi-hat samples (high frequency)
├── bass/                  # Bass samples (individual samples)
├── vocals/                # Vocal stem (whole acapella file)
└── other/                 # Other instrument samples (individual samples)
```

## Sample Naming Convention

Samples are automatically named with the following format:
- `{instrument}_{number}_{bpm}bpm_{pitch}.{format}`

Examples:
- `kick_001_120bpm_C4.wav`
- `hihat_002_120bpm_N/A.wav`
- `bass_001_120bpm_A2.wav`
- `vocals_120bpm.wav` (whole acapella file)

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