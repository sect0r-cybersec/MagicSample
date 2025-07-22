#!/usr/bin/env python3.10
"""
MagicSample
Enhanced audio sample extraction and drumkit creation tool

This version uses Demucs for stem separation and includes:
- BPM detection
- Pitch estimation
- Drum classification (hi-hat, snare, bass drum, etc.)
- Organized drumkit folder structure
"""
__version__ = '0.0.2'

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# PyQt6 imports
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

# Audio processing imports
import librosa
import librosa.core
import librosa.beat
import librosa.effects
import librosa.feature
import soundfile as sf
import numpy as np
import scipy.signal as signal

# Demucs imports
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

# Pitch detection
try:
    import pyin
except ImportError:
    print("Warning: pyin not available, using librosa for pitch detection")

class DrumClassifier:
    """Classifies drum samples into different categories"""
    
    def __init__(self):
        self.categories = {
            'hihat': ['hihat', 'hi-hat', 'hi_hat', 'cymbal', 'crash', 'ride'],
            'snare': ['snare', 'clap'],
            'kick': ['kick', 'bass', 'bassdrum', 'bass_drum'],
            'tom': ['tom', 'floor_tom', 'rack_tom'],
            'clap': ['clap', 'hand_clap'],
            'percussion': ['perc', 'percussion', 'shaker', 'tambourine', 'cowbell']
        }
    
    def classify_sample(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Classify a drum sample based on its spectral characteristics"""
        try:
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            
            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Get average values
            avg_centroid = np.mean(spectral_centroid)
            avg_rolloff = np.mean(spectral_rolloff)
            avg_bandwidth = np.mean(spectral_bandwidth)
            avg_rms = np.mean(rms)
            avg_zcr = np.mean(zcr)
            
            # Classification logic based on spectral characteristics
            if avg_centroid > 4000 and avg_zcr > 0.1:
                return 'hihat'
            elif avg_centroid < 1000 and avg_rms > 0.3:
                return 'kick'
            elif 1000 < avg_centroid < 3000 and avg_bandwidth > 2000:
                return 'snare'
            elif avg_centroid < 2000 and avg_rolloff < 3000:
                return 'tom'
            elif avg_zcr > 0.15:
                return 'clap'
            else:
                return 'percussion'
                
        except Exception as e:
            print(f"Classification error: {e}")
            return 'unknown'

class PitchDetector:
    """Detects pitch information from audio samples"""
    
    def __init__(self):
        self.min_freq = 50
        self.max_freq = 2000
    
    def detect_pitch(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Detect the fundamental pitch of an audio sample"""
        try:
            # Use librosa's pitch detection
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate, 
                                                 fmin=self.min_freq, fmax=self.max_freq)
            
            # Find the pitch with maximum magnitude
            max_magnitude_idx = np.argmax(magnitudes, axis=0)
            pitches_filtered = pitches[max_magnitude_idx, np.arange(pitches.shape[1])]
            
            # Filter out zero pitches and get the most common pitch
            pitches_filtered = pitches_filtered[pitches_filtered > 0]
            
            if len(pitches_filtered) == 0:
                return None
            
            # Get the median pitch (more robust than mean)
            median_pitch = np.median(pitches_filtered)
            
            # Convert frequency to note name
            note_name = self.freq_to_note(median_pitch)
            return note_name
            
        except Exception as e:
            print(f"Pitch detection error: {e}")
            return None
    
    def freq_to_note(self, freq: float) -> str:
        """Convert frequency to musical note name"""
        if freq <= 0:
            return "N/A"
        
        # A4 = 440 Hz
        A4 = 440.0
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Calculate semitones from A4
        semitones = 12 * np.log2(freq / A4)
        
        # Round to nearest semitone
        semitones_rounded = round(semitones)
        
        # Calculate note and octave
        note_index = (semitones_rounded + 9) % 12  # A is index 9
        octave = (semitones_rounded + 9) // 12 + 4  # A4 is octave 4
        
        return f"{note_names[note_index]}{octave}"

class DemucsProcessor:
    """Handles Demucs stem separation"""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_rate = 44100
    
    def load_model(self, model_name: str = "htdemucs"):
        """Load the Demucs model"""
        try:
            self.model = get_model(model_name)
            self.model.to(self.device)
            print(f"Loaded Demucs model: {model_name}")
        except Exception as e:
            print(f"Error loading Demucs model: {e}")
            raise
    
    def separate_stems(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Separate audio into stems using Demucs"""
        if self.model is None:
            self.load_model()
        
        try:
            print(f"Separating stems for: {audio_path}")
            print(f"Output directory: {output_dir}")
            
            # Load audio using librosa to ensure correct format
            wav, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
            print(f"Loaded audio with librosa: shape={wav.shape}, sample_rate={sr}")
            
            # Convert to torch tensor and ensure correct shape
            import torch
            if len(wav.shape) == 1:
                # Mono audio - convert to stereo
                wav = np.stack([wav, wav])
            elif wav.shape[0] == 1:
                # Mono audio - convert to stereo
                wav = np.repeat(wav, 2, axis=0)
            
            # Convert to torch tensor
            wav_tensor = torch.from_numpy(wav).float()
            print(f"Converted to torch tensor: shape={wav_tensor.shape}")
            
            # Normalize
            ref = wav_tensor.mean(0)
            wav_tensor = (wav_tensor - ref.mean()) / ref.std()
            
            # Separate stems
            print("Applying Demucs model...")
            sources = apply_model(self.model, wav_tensor[None], device=self.device)[0]
            sources = sources * ref.std() + ref.mean()
            print(f"Separated into {len(sources)} stems")
            
            # Save stems
            stem_paths = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            for i, (source, name) in enumerate(zip(sources, stem_names)):
                stem_path = os.path.join(output_dir, f"{name}.wav")
                print(f"Saving {name} stem to: {stem_path}")
                
                # Convert back to numpy and save
                source_np = source.numpy()
                import soundfile as sf
                sf.write(stem_path, source_np.T, self.sample_rate)
                
                # Verify the file was created and has content
                if os.path.exists(stem_path) and os.path.getsize(stem_path) > 0:
                    stem_paths[name] = stem_path
                    print(f"✓ {name} stem saved successfully ({os.path.getsize(stem_path)} bytes)")
                else:
                    print(f"✗ {name} stem file is empty or missing")
            
            print(f"Successfully created {len(stem_paths)} stem files")
            return stem_paths
            
        except Exception as e:
            print(f"Error in stem separation: {e}")
            import traceback
            traceback.print_exc()
            raise

class MainWindow(QWidget):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.demucs_processor = DemucsProcessor()
        self.drum_classifier = DrumClassifier()
        self.pitch_detector = PitchDetector()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("MagicSample")
        self.setGeometry(100, 100, 800, 600)  # More compact size
        
        # Main layout with centered alignment
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("MagicSample")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout = QGridLayout()
        file_layout.setSpacing(10)
        
        # Input file selection
        input_label = QLabel("Input Audio File:")
        input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(input_label, 0, 0)
        
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select input audio file...")
        self.input_path_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(self.input_path_edit, 0, 1)
        
        input_btn = QPushButton("Browse")
        input_btn.clicked.connect(self.select_input_file)
        input_btn.setFixedWidth(80)
        file_layout.addWidget(input_btn, 0, 2)
        
        # Output directory selection
        output_label = QLabel("Output Directory:")
        output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(output_label, 1, 0)
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output directory...")
        self.output_path_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(self.output_path_edit, 1, 1)
        
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.select_output_dir)
        output_btn.setFixedWidth(80)
        file_layout.addWidget(output_btn, 1, 2)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Options section
        options_group = QGroupBox("Processing Options")
        options_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout = QGridLayout()
        options_layout.setSpacing(8)
        
        # Checkboxes in a more compact layout
        self.stems_checkbox = QCheckBox("Split to Stems")
        self.stems_checkbox.setChecked(True)
        options_layout.addWidget(self.stems_checkbox, 0, 0)
        
        self.bpm_checkbox = QCheckBox("Detect BPM")
        self.bpm_checkbox.setChecked(True)
        options_layout.addWidget(self.bpm_checkbox, 0, 1)
        
        self.pitch_checkbox = QCheckBox("Detect Pitch")
        self.pitch_checkbox.setChecked(True)
        options_layout.addWidget(self.pitch_checkbox, 1, 0)
        
        self.drum_classify_checkbox = QCheckBox("Classify Drums")
        self.drum_classify_checkbox.setChecked(True)
        options_layout.addWidget(self.drum_classify_checkbox, 1, 1)
        
        # Sensitivity slider
        sensitivity_label = QLabel("Sample Detection Sensitivity (lower = more samples)")
        sensitivity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(sensitivity_label, 2, 0, 1, 2)
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(5, 30)
        self.sensitivity_slider.setValue(15)
        self.sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sensitivity_slider.setTickInterval(5)
        options_layout.addWidget(self.sensitivity_slider, 3, 0, 1, 2)
        
        # Output format and drumkit name in a row
        format_label = QLabel("Output Format:")
        format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(format_label, 4, 0)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['WAV', 'FLAC', 'OGG'])
        options_layout.addWidget(self.format_combo, 4, 1)
        
        drumkit_label = QLabel("Drumkit Name:")
        drumkit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(drumkit_label, 5, 0)
        
        self.drumkit_name_edit = QLineEdit()
        self.drumkit_name_edit.setPlaceholderText("MyDrumkit")
        self.drumkit_name_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(self.drumkit_name_edit, 5, 1)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout = QVBoxLayout()
        progress_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.setSpacing(20)
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setFixedWidth(150)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedWidth(100)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def select_input_file(self):
        """Select input audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        if file_path:
            self.input_path_edit.setText(file_path)
    
    def select_output_dir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path_edit.setText(dir_path)
    
    def start_processing(self):
        """Start the audio processing"""
        if not self.input_path_edit.text() or not self.output_path_edit.text():
            QMessageBox.warning(self, "Error", "Please select input file and output directory")
            return
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Start processing in a separate thread
        self.worker = ProcessingWorker(
            self.input_path_edit.text(),
            self.output_path_edit.text(),
            self.drumkit_name_edit.text() or "MyDrumkit",
            self.format_combo.currentText(),
            self.stems_checkbox.isChecked(),
            self.bpm_checkbox.isChecked(),
            self.pitch_checkbox.isChecked(),
            self.drum_classify_checkbox.isChecked(),
            self.sensitivity_slider.value()
        )
        
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()
    
    def stop_processing(self):
        """Stop the audio processing"""
        if hasattr(self, 'worker'):
            self.worker.stop()
        self.processing_finished()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
    
    def processing_finished(self):
        """Called when processing is finished"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing completed!")

class ProcessingWorker(QThread):
    """Worker thread for audio processing"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, input_path, output_path, drumkit_name, output_format, 
                 split_stems, detect_bpm, detect_pitch, classify_drums, sensitivity):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.drumkit_name = drumkit_name
        self.output_format = output_format
        self.split_stems = split_stems
        self.detect_bpm = detect_bpm
        self.detect_pitch = detect_pitch
        self.classify_drums = classify_drums
        self.sensitivity = sensitivity
        self.stop_flag = False
        
        # Initialize processors
        self.demucs_processor = DemucsProcessor()
        self.drum_classifier = DrumClassifier()
        self.pitch_detector = PitchDetector()
    
    def run(self):
        """Main processing function"""
        try:
            self.status_updated.emit("Loading audio file...")
            self.progress_updated.emit(5)
            
            # Load audio file
            audio_data, sample_rate = librosa.load(self.input_path, sr=None, mono=False)
            print(f"Loaded audio: {audio_data.shape}, sample rate: {sample_rate}")
            
            # Detect BPM if requested
            bpm = None
            if self.detect_bpm:
                self.status_updated.emit("Detecting BPM...")
                self.progress_updated.emit(10)
                bpm = self.detect_bpm_from_audio(audio_data, sample_rate)
                print(f"Detected BPM: {bpm}")
            
            # Create output directory structure
            drumkit_path = os.path.join(self.output_path, self.drumkit_name.capitalize())
            os.makedirs(drumkit_path, exist_ok=True)
            
            if self.split_stems:
                self.status_updated.emit("Separating stems with Demucs...")
                self.progress_updated.emit(20)
                
                # Create temporary directory for stems
                temp_dir = os.path.join(drumkit_path, "temp_stems")
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    # Separate stems
                    stem_paths = self.demucs_processor.separate_stems(self.input_path, temp_dir)
                    print(f"Created stems: {list(stem_paths.keys())}")
                    
                    # Process each stem according to its type
                    stem_count = len(stem_paths)
                    for i, (stem_name, stem_path) in enumerate(stem_paths.items()):
                        if self.stop_flag:
                            break
                        
                        progress = 20 + (i * 60 // stem_count)
                        self.status_updated.emit(f"Processing {stem_name} stem...")
                        self.progress_updated.emit(progress)
                        
                        # Check if stem file exists and has content
                        if not os.path.exists(stem_path) or os.path.getsize(stem_path) == 0:
                            print(f"Warning: Stem file {stem_path} is empty or missing")
                            continue
                        
                        # Load stem audio
                        stem_audio, stem_sr = librosa.load(stem_path, sr=None, mono=False)
                        print(f"Loaded {stem_name} stem: {stem_audio.shape}")
                        
                        # Create stem directory
                        stem_dir = os.path.join(drumkit_path, stem_name.capitalize())
                        os.makedirs(stem_dir, exist_ok=True)
                        
                        # Process stem according to its type
                        if stem_name == "vocals":
                            # Save vocals as whole file (no sample splitting)
                            vocals_filename = f"Vocals_{bpm}BPM.{self.output_format.upper()}" if bpm else f"Vocals.{self.output_format.upper()}"
                            vocals_path = os.path.join(stem_dir, vocals_filename)
                            self.save_audio_sample(stem_audio, stem_sr, vocals_path)
                            print(f"Saved vocals as whole file: {vocals_filename}")
                            
                        elif stem_name == "drums":
                            # Process drums with frequency-based subfolders
                            self.process_drums_with_subfolders(stem_audio, stem_sr, stem_dir, bpm)
                            
                        else:
                            # Process other stems (bass, other) as individual samples
                            sample_count = self.process_stem_into_samples(stem_audio, stem_sr, stem_dir, stem_name.capitalize(), bpm)
                            print(f"Created {sample_count} samples for {stem_name}")
                    
                    # Clean up temporary files
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                except Exception as e:
                    print(f"Error in stem separation: {e}")
                    self.status_updated.emit(f"Stem separation failed: {str(e)}")
                    # Fall back to processing the whole file
                    self.status_updated.emit("Falling back to processing whole file...")
                    samples_dir = os.path.join(drumkit_path, "Samples")
                    os.makedirs(samples_dir, exist_ok=True)
                    sample_count = self.process_stem_into_samples(audio_data, sample_rate, samples_dir, "Sample", bpm)
                    print(f"Created {sample_count} samples from whole file")
            else:
                # Process entire file as one stem
                self.status_updated.emit("Processing audio into samples...")
                self.progress_updated.emit(40)
                
                samples_dir = os.path.join(drumkit_path, "Samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                sample_count = self.process_stem_into_samples(audio_data, sample_rate, samples_dir, "Sample", bpm)
                print(f"Created {sample_count} samples from whole file")
            
            # self.status_updated.emit("Creating drumkit metadata...")
            # self.progress_updated.emit(90)
            # self.create_drumkit_metadata(drumkit_path, bpm)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Drumkit creation completed!")
            
        except Exception as e:
            self.status_updated.emit(f"Error: {str(e)}")
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def detect_bpm_from_audio(self, audio_data, sample_rate):
        """Detect BPM from audio"""
        try:
            # Convert to mono for BPM detection
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            tempo, beats = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
            # Convert numpy array to scalar and round
            if hasattr(tempo, '__iter__'):
                tempo = float(tempo[0])
            else:
                tempo = float(tempo)
            return int(round(tempo))
        except Exception as e:
            print(f"BPM detection error: {e}")
            return None
    
    def process_stem_into_samples(self, audio_data, sample_rate, output_dir, stem_name, bpm):
        """Process a stem into individual samples"""
        try:
            # Convert to mono for sample detection
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            print(f"Processing {stem_name}: audio length {len(audio_mono)/sample_rate:.2f}s, sensitivity {self.sensitivity}dB")
            
            # Try different sensitivity levels if no samples are found
            sensitivities_to_try = [self.sensitivity, self.sensitivity + 5, self.sensitivity + 10, 30]
            sample_boundaries = []
            
            for sensitivity in sensitivities_to_try:
                sample_boundaries = librosa.effects.split(audio_mono, top_db=sensitivity)
                print(f"Tried sensitivity {sensitivity}dB: found {len(sample_boundaries)} potential samples")
                
                if len(sample_boundaries) > 0:
                    break
            
            if len(sample_boundaries) == 0:
                print(f"No samples detected for {stem_name} even with high sensitivity. Creating single sample.")
                # Create one sample from the entire audio
                sample_boundaries = [(0, len(audio_mono))]
            
            sample_count = 0
            for i, (start, end) in enumerate(sample_boundaries):
                if self.stop_flag:
                    break
                
                # Extract sample
                if len(audio_data.shape) > 1:
                    sample_audio = audio_data[:, start:end]
                else:
                    sample_audio = audio_data[start:end]
                
                # Skip very short samples (unless it's the only sample)
                sample_duration = sample_audio.shape[-1] / sample_rate
                if sample_duration < 0.01 and len(sample_boundaries) > 1:  # Less than 10ms
                    print(f"Skipping sample {i+1}: too short ({sample_duration*1000:.1f}ms)")
                    continue
                
                # Skip very quiet samples (unless it's the only sample)
                sample_rms = np.sqrt(np.mean(sample_audio**2))
                if sample_rms < 0.001 and len(sample_boundaries) > 1:  # Very quiet
                    print(f"Skipping sample {i+1}: too quiet (RMS: {sample_rms:.6f})")
                    continue
                
                print(f"Processing sample {i+1}: duration {sample_duration:.3f}s, RMS {sample_rms:.6f}")
                
                # Generate filename
                filename_parts = [f"{stem_name.capitalize()}_{i+1:03d}"]
                
                if bpm:
                    filename_parts.append(f"{bpm}BPM")
                
                # Detect pitch if requested
                if self.detect_pitch:
                    pitch = self.pitch_detector.detect_pitch(sample_audio, sample_rate)
                    if pitch and pitch != "N/A":
                        filename_parts.append(pitch)
                
                # Classify drums if requested and this is a drums stem
                if self.classify_drums and stem_name == "drums":
                    drum_type = self.drum_classifier.classify_sample(sample_audio, sample_rate)
                    if drum_type != "unknown":
                        # Create subdirectory for drum type
                        drum_type_dir = os.path.join(output_dir, drum_type)
                        os.makedirs(drum_type_dir, exist_ok=True)
                        sample_dir = drum_type_dir
                    else:
                        sample_dir = output_dir
                else:
                    sample_dir = output_dir
                
                # Create final filename
                filename = "_".join(filename_parts) + f".{self.output_format.upper()}"
                filepath = os.path.join(sample_dir, filename)
                
                # Save sample
                self.save_audio_sample(sample_audio, sample_rate, filepath)
                sample_count += 1
            
            print(f"Successfully created {sample_count} samples for {stem_name}")
            return sample_count
                
        except Exception as e:
            print(f"Error processing stem {stem_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0  # Return 0 samples processed on error
    
    def process_drums_with_subfolders(self, audio_data, sample_rate, output_dir, bpm):
        """Process drums into frequency-based subfolders: Kick, Perc, HiHat"""
        try:
            # Convert to mono for sample detection
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            print(f"Processing drums: audio length {len(audio_mono)/sample_rate:.2f}s, sensitivity {self.sensitivity}dB")
            
            # Try different sensitivity levels if no samples are found
            sensitivities_to_try = [self.sensitivity, self.sensitivity + 5, self.sensitivity + 10, 30]
            sample_boundaries = []
            
            for sensitivity in sensitivities_to_try:
                sample_boundaries = librosa.effects.split(audio_mono, top_db=sensitivity)
                print(f"Tried sensitivity {sensitivity}dB: found {len(sample_boundaries)} potential drum samples")
                
                if len(sample_boundaries) > 0:
                    break
            
            if len(sample_boundaries) == 0:
                print(f"No drum samples detected even with high sensitivity. Creating single sample.")
                # Create one sample from the entire audio
                sample_boundaries = [(0, len(audio_mono))]
            
            # Create subfolders for different drum types
            kick_dir = os.path.join(output_dir, "Kick")
            perc_dir = os.path.join(output_dir, "Perc")
            hihat_dir = os.path.join(output_dir, "HiHat")
            
            os.makedirs(kick_dir, exist_ok=True)
            os.makedirs(perc_dir, exist_ok=True)
            os.makedirs(hihat_dir, exist_ok=True)
            
            sample_count = 0
            for i, (start, end) in enumerate(sample_boundaries):
                if self.stop_flag:
                    break
                
                # Extract sample
                if len(audio_data.shape) > 1:
                    sample_audio = audio_data[:, start:end]
                else:
                    sample_audio = audio_data[start:end]
                
                # Skip very short samples (unless it's the only sample)
                sample_duration = sample_audio.shape[-1] / sample_rate
                if sample_duration < 0.01 and len(sample_boundaries) > 1:  # Less than 10ms
                    print(f"Skipping drum sample {i+1}: too short ({sample_duration*1000:.1f}ms)")
                    continue
                
                # Skip very quiet samples (unless it's the only sample)
                sample_rms = np.sqrt(np.mean(sample_audio**2))
                if sample_rms < 0.001 and len(sample_boundaries) > 1:  # Very quiet
                    print(f"Skipping drum sample {i+1}: too quiet (RMS: {sample_rms:.6f})")
                    continue
                
                print(f"Processing drum sample {i+1}: duration {sample_duration:.3f}s, RMS {sample_rms:.6f}")
                
                # Classify drum sample based on frequency characteristics
                drum_type = self.classify_drum_by_frequency(sample_audio, sample_rate)
                print(f"Classified as: {drum_type}")
                
                # Determine target directory
                if drum_type == "Kick":
                    target_dir = kick_dir
                    prefix = "Kick"
                elif drum_type == "HiHat":
                    target_dir = hihat_dir
                    prefix = "HiHat"
                else:  # Perc
                    target_dir = perc_dir
                    prefix = "Perc"
                
                # Generate filename
                filename_parts = [f"{prefix}_{i+1:03d}"]
                
                if bpm:
                    filename_parts.append(f"{bpm}BPM")
                
                # Detect pitch if requested
                if self.detect_pitch:
                    pitch = self.pitch_detector.detect_pitch(sample_audio, sample_rate)
                    if pitch and pitch != "N/A":
                        filename_parts.append(pitch)
                
                # Create final filename
                filename = "_".join(filename_parts) + f".{self.output_format.capitalize()}"
                filepath = os.path.join(target_dir, filename)
                
                # Save sample
                self.save_audio_sample(sample_audio, sample_rate, filepath)
                sample_count += 1
            
            print(f"Successfully created {sample_count} drum samples:")
            print(f"  - Kick: {len(os.listdir(kick_dir))} samples")
            print(f"  - Perc: {len(os.listdir(perc_dir))} samples")
            print(f"  - HiHat: {len(os.listdir(hihat_dir))} samples")
            return sample_count
                
        except Exception as e:
            print(f"Error processing drums: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def classify_drum_by_frequency(self, audio_data, sample_rate):
        """Classify drum sample as Kick, HiHat, or Perc based on frequency characteristics"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sample_rate)[0]
            rms = librosa.feature.rms(y=audio_mono)[0]
            zcr = librosa.feature.zero_crossing_rate(audio_mono)[0]
            
            avg_centroid = np.mean(spectral_centroid)
            avg_bandwidth = np.mean(spectral_bandwidth)
            avg_rms = np.mean(rms)
            avg_zcr = np.mean(zcr)
            
            # Kick: low centroid, high energy
            if avg_centroid < 1000 and avg_rms > 0.2:
                return "Kick"
            # HiHat: high centroid, high zero crossing
            elif avg_centroid > 4000 and avg_zcr > 0.1:
                return "HiHat"
            # Perc: everything else
            else:
                return "Perc"
        except Exception as e:
            print(f"Classification error: {e}")
            return "Perc"
    
    def save_audio_sample(self, audio_data, sample_rate, filepath):
        """Save audio sample to file"""
        try:
            # Ensure audio data is in the correct format for soundfile
            if len(audio_data.shape) == 1:
                # Mono audio
                sf.write(filepath, audio_data, sample_rate)
            else:
                # Stereo audio - transpose for soundfile
                audio_transposed = np.transpose(audio_data)
                sf.write(filepath, audio_transposed, sample_rate)
        except Exception as e:
            print(f"Error saving audio sample: {e}")
    
    def create_drumkit_metadata(self, drumkit_path, bpm):
        """Create metadata file for the drumkit"""
        try:
            metadata = {
                "drumkit_name": self.drumkit_name,
                "source_file": self.input_path,
                "bpm": bpm,
                "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_options": {
                    "split_stems": self.split_stems,
                    "detect_bpm": self.detect_bpm,
                    "detect_pitch": self.detect_pitch,
                    "classify_drums": self.classify_drums,
                    "sensitivity": self.sensitivity,
                    "output_format": self.output_format
                }
            }
            
            metadata_path = os.path.join(drumkit_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error creating metadata: {e}")
    
    def stop(self):
        """Stop the processing"""
        self.stop_flag = True

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 