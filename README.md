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

1. **Install Python 3.10** (required for compatibility)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### System Requirements

- Python 3.10
- CUDA-compatible GPU (recommended for faster processing)
- At least 8GB RAM
- 2GB free disk space

## Usage

1. **Run the application**:
   ```bash
   python MagicSampleDemucs.py
   ```

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
├── drums/                 # Drum samples
│   ├── hihat/            # Hi-hat samples
│   ├── snare/            # Snare samples
│   ├── kick/             # Kick/bass drum samples
│   ├── tom/              # Tom samples
│   ├── clap/             # Clap samples
│   └── percussion/       # Other percussion
├── bass/                  # Bass samples
├── vocals/                # Vocal samples
└── other/                 # Other instrument samples
```

## Sample Naming Convention

Samples are automatically named with the following format:
- `{instrument}_{number}_{bpm}bpm_{pitch}.{format}`

Examples:
- `drums_001_120bpm_C4.wav`
- `hihat_002_120bpm_N/A.wav`
- `bass_001_120bpm_A2.wav`

## Technical Details

### Stem Separation
- Uses Demucs `htdemucs` model for high-quality separation
- Supports 4 stems: drums, bass, vocals, other
- Processes at 44.1kHz sample rate

### BPM Detection
- Uses librosa's beat tracking algorithm
- Provides accurate tempo estimation
- Labels all samples with detected BPM

### Pitch Detection
- Uses librosa's pitch tracking
- Converts frequencies to musical note names (C4, A2, etc.)
- Handles both pitched and unpitched sounds

### Drum Classification
- Analyzes spectral characteristics (centroid, rolloff, bandwidth)
- Uses RMS energy and zero-crossing rate
- Automatically categorizes into drum types

### Sample Detection
- Uses librosa's onset detection
- Configurable sensitivity (5-30 dB)
- Filters out very short samples (< 10ms)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU processing
2. **Slow Processing**: Ensure you have a CUDA-compatible GPU
3. **Poor Separation Quality**: Try different audio files or adjust sensitivity
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Performance Tips

- Use WAV files for best quality
- Process shorter files for faster results
- Use GPU acceleration when available
- Close other applications to free up memory

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