#!/usr/bin/env python3.10
"""
MagicSample Demucs Version
Enhanced audio sample extraction and drumkit creation tool

This version uses Demucs for stem separation and includes:
- BPM detection
- Pitch estimation
- Drum classification (hi-hat, snare, bass drum, etc.)
- Organized drumkit folder structure
"""

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
            # Load audio
            wav, sr = AudioFile(audio_path).read(streams=0, samplerate=self.sample_rate, channels=2)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            
            # Separate stems
            sources = apply_model(self.model, wav[None], device=self.device)[0]
            sources = sources * ref.std() + ref.mean()
            
            # Save stems
            stem_paths = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            for source, name in zip(sources, stem_names):
                stem_path = os.path.join(output_dir, f"{name}.wav")
                save_audio(stem_path, source, self.sample_rate)
                stem_paths[name] = stem_path
            
            return stem_paths
            
        except Exception as e:
            print(f"Error in stem separation: {e}")
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
        self.setWindowTitle("MagicSample Demucs")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        layout = QVBoxLayout()
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()
        
        # Input file selection
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Input Audio File:"))
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select input audio file...")
        input_layout.addWidget(self.input_path_edit)
        
        input_btn = QPushButton("Browse")
        input_btn.clicked.connect(self.select_input_file)
        input_layout.addWidget(input_btn)
        
        file_layout.addLayout(input_layout)
        
        # Output directory selection
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output directory...")
        output_layout.addWidget(self.output_path_edit)
        
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(output_btn)
        
        file_layout.addLayout(output_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Options section
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout()
        
        # Checkboxes
        self.stems_checkbox = QCheckBox("Split to stems")
        self.stems_checkbox.setChecked(True)
        options_layout.addWidget(self.stems_checkbox, 0, 0)
        
        self.bpm_checkbox = QCheckBox("Detect BPM")
        self.bpm_checkbox.setChecked(True)
        options_layout.addWidget(self.bpm_checkbox, 0, 1)
        
        self.pitch_checkbox = QCheckBox("Detect pitch")
        self.pitch_checkbox.setChecked(True)
        options_layout.addWidget(self.pitch_checkbox, 0, 2)
        
        self.drum_classify_checkbox = QCheckBox("Classify drums")
        self.drum_classify_checkbox.setChecked(True)
        options_layout.addWidget(self.drum_classify_checkbox, 0, 3)
        
        # Sensitivity slider
        options_layout.addWidget(QLabel("Sample Detection Sensitivity:"), 1, 0)
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(5, 30)
        self.sensitivity_slider.setValue(15)
        self.sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sensitivity_slider.setTickInterval(5)
        options_layout.addWidget(self.sensitivity_slider, 1, 1, 1, 3)
        
        # Output format
        options_layout.addWidget(QLabel("Output Format:"), 2, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['wav', 'flac', 'ogg'])
        options_layout.addWidget(self.format_combo, 2, 1)
        
        # Drumkit name
        options_layout.addWidget(QLabel("Drumkit Name:"), 2, 2)
        self.drumkit_name_edit = QLineEdit()
        self.drumkit_name_edit.setPlaceholderText("MyDrumkit")
        options_layout.addWidget(self.drumkit_name_edit, 2, 3)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
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
            
            # Detect BPM if requested
            bpm = None
            if self.detect_bpm:
                self.status_updated.emit("Detecting BPM...")
                self.progress_updated.emit(10)
                bpm = self.detect_bpm_from_audio(audio_data, sample_rate)
            
            # Create output directory structure
            drumkit_path = os.path.join(self.output_path, self.drumkit_name)
            os.makedirs(drumkit_path, exist_ok=True)
            
            if self.split_stems:
                self.status_updated.emit("Separating stems with Demucs...")
                self.progress_updated.emit(20)
                
                # Create temporary directory for stems
                temp_dir = os.path.join(drumkit_path, "temp_stems")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Separate stems
                stem_paths = self.demucs_processor.separate_stems(self.input_path, temp_dir)
                
                # Process each stem
                stem_count = len(stem_paths)
                for i, (stem_name, stem_path) in enumerate(stem_paths.items()):
                    if self.stop_flag:
                        break
                    
                    progress = 20 + (i * 60 // stem_count)
                    self.status_updated.emit(f"Processing {stem_name} stem...")
                    self.progress_updated.emit(progress)
                    
                    # Load stem audio
                    stem_audio, stem_sr = librosa.load(stem_path, sr=None, mono=False)
                    
                    # Create stem directory
                    stem_dir = os.path.join(drumkit_path, stem_name)
                    os.makedirs(stem_dir, exist_ok=True)
                    
                    # Process stem into samples
                    self.process_stem_into_samples(stem_audio, stem_sr, stem_dir, stem_name, bpm)
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_dir)
            else:
                # Process entire file as one stem
                self.status_updated.emit("Processing audio into samples...")
                self.progress_updated.emit(40)
                
                samples_dir = os.path.join(drumkit_path, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                self.process_stem_into_samples(audio_data, sample_rate, samples_dir, "sample", bpm)
            
            self.status_updated.emit("Creating drumkit metadata...")
            self.progress_updated.emit(90)
            
            # Create metadata file
            self.create_drumkit_metadata(drumkit_path, bpm)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Drumkit creation completed!")
            
        except Exception as e:
            self.status_updated.emit(f"Error: {str(e)}")
            print(f"Processing error: {e}")
    
    def detect_bpm_from_audio(self, audio_data, sample_rate):
        """Detect BPM from audio"""
        try:
            # Convert to mono for BPM detection
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            tempo, beats = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
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
            
            # Detect sample boundaries
            sample_boundaries = librosa.effects.split(audio_mono, top_db=self.sensitivity)
            
            for i, (start, end) in enumerate(sample_boundaries):
                if self.stop_flag:
                    break
                
                # Extract sample
                sample_audio = audio_data[:, start:end]
                
                # Skip very short samples
                if sample_audio.shape[1] < sample_rate * 0.01:  # Less than 10ms
                    continue
                
                # Generate filename
                filename_parts = [f"{stem_name}_{i+1:03d}"]
                
                if bpm:
                    filename_parts.append(f"{bpm}bpm")
                
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
                filename = "_".join(filename_parts) + f".{self.output_format}"
                filepath = os.path.join(sample_dir, filename)
                
                # Save sample
                self.save_audio_sample(sample_audio, sample_rate, filepath)
                
        except Exception as e:
            print(f"Error processing stem {stem_name}: {e}")
    
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