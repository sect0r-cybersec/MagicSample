#!/usr/bin/env python3.10
"""
MagicSample
Enhanced audio sample extraction and drumkit creation tool

This version uses Demucs for stem separation and includes:
- BPM detection
- Advanced multi-algorithm pitch detection (Autocorrelation, HPS, Cepstrum, YIN)
- Scientific Pitch Notation (SPN) integration in filenames
- Drum classification (hi-hat, snare, bass drum, etc.)
- Organized drumkit folder structure
- Similarity comparison to avoid duplicate samples
- Sample processing timeout protection
"""
__version__ = '0.0.5'

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import time
import json
import logging
import threading
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
from scipy.spatial.distance import cosine

# Demucs imports
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

# Pitch detection - using our own advanced multi-algorithm implementation

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)

class LogHandler(logging.Handler, QObject):
    """Custom log handler that emits signals for GUI updates"""
    log_signal = pyqtSignal(str)
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
    
    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class SampleSimilarityChecker:
    """Checks similarity between audio samples to avoid duplicates"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.saved_samples = {}  # category -> list of feature vectors
    
    def calculate_sample_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Calculate feature vector for similarity comparison"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Calculate MFCC features (good for timbral similarity)
            mfcc = librosa.feature.mfcc(y=audio_mono, sr=sample_rate, n_mfcc=13)
            
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sample_rate)[0]
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio_mono)[0]
            
            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_mono)[0]
            
            # Get average values
            avg_mfcc = np.mean(mfcc, axis=1)
            avg_centroid = np.mean(spectral_centroid)
            avg_rolloff = np.mean(spectral_rolloff)
            avg_bandwidth = np.mean(spectral_bandwidth)
            avg_rms = np.mean(rms)
            avg_zcr = np.mean(zcr)
            
            # Combine all features into a single vector
            features = np.concatenate([
                avg_mfcc,
                [avg_centroid, avg_rolloff, avg_bandwidth, avg_rms, avg_zcr]
            ])
            
            # Normalize the feature vector
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"Error calculating sample features: {e}")
            return None
    
    def is_similar_to_existing(self, audio_data: np.ndarray, sample_rate: int, category: str) -> bool:
        """Check if a sample is too similar to existing samples in the same category"""
        try:
            # Calculate features for the current sample
            current_features = self.calculate_sample_features(audio_data, sample_rate)
            if current_features is None:
                return False  # If we can't calculate features, allow the sample
            
            # Initialize category if it doesn't exist
            if category not in self.saved_samples:
                self.saved_samples[category] = []
                return False  # First sample in category is always allowed
            
            # Compare with existing samples in the same category
            for existing_features in self.saved_samples[category]:
                # Calculate cosine similarity (1 = identical, 0 = completely different)
                similarity = 1 - cosine(current_features, existing_features)
                
                if similarity >= self.similarity_threshold:
                    print(f"Sample rejected: {similarity:.3f} similarity to existing {category} sample")
                    return True  # Too similar
            
            # Not too similar to any existing sample, add to saved samples
            self.saved_samples[category].append(current_features)
            return False
            
        except Exception as e:
            print(f"Error in similarity check: {e}")
            return False  # Allow sample if similarity check fails
    
    def set_threshold(self, threshold: float):
        """Update the similarity threshold (0.0 to 1.0)"""
        self.similarity_threshold = max(0.0, min(1.0, threshold))

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
    """Advanced pitch detection using multiple algorithms for robust results"""
    
    def __init__(self):
        self.min_freq = 50
        self.max_freq = 2000
        self.frame_length = 2048
        self.hop_length = 512
        
        # Note frequencies for A4 = 440Hz reference
        self.A4 = 440.0
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Confidence thresholds
        self.min_confidence = 0.3
        self.min_peak_ratio = 0.1
    
    def detect_pitch(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Detect the fundamental pitch using multiple algorithms"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Normalize audio
            audio_mono = audio_mono / (np.max(np.abs(audio_mono)) + 1e-8)
            
            # Apply multiple pitch detection methods
            results = []
            
            # Method 1: Autocorrelation (good for low-mid frequencies)
            autocorr_pitch = self._autocorrelation_method(audio_mono, sample_rate)
            if autocorr_pitch:
                results.append(autocorr_pitch)
            
            # Method 2: Harmonic Product Spectrum (robust to noise)
            hps_pitch = self._harmonic_product_spectrum(audio_mono, sample_rate)
            if hps_pitch:
                results.append(hps_pitch)
            
            # Method 3: Cepstrum (good for harmonic signals)
            cepstrum_pitch = self._cepstrum_method(audio_mono, sample_rate)
            if cepstrum_pitch:
                results.append(cepstrum_pitch)
            
            # Method 4: YIN algorithm (modern, accurate)
            yin_pitch = self._yin_algorithm(audio_mono, sample_rate)
            if yin_pitch:
                results.append(yin_pitch)
            
            # Combine results using voting/consensus
            if not results:
                return None
            
            # Find the most common pitch or use median if no clear consensus
            if len(results) >= 2:
                # Check if results are within a semitone of each other
                semitone_tolerance = 0.5
                consensus_results = []
                
                for i, freq1 in enumerate(results):
                    for j, freq2 in enumerate(results[i+1:], i+1):
                        semitone_diff = abs(12 * np.log2(freq1 / freq2))
                        if semitone_diff <= semitone_tolerance:
                            consensus_results.extend([freq1, freq2])
                
                if consensus_results:
                    # Use median of consensus results
                    final_freq = np.median(consensus_results)
                else:
                    # Use median of all results
                    final_freq = np.median(results)
            else:
                final_freq = results[0]
            
            # Convert to note name
            note_name = self.freq_to_note(final_freq)
            return note_name
            
        except Exception as e:
            print(f"Pitch detection error: {e}")
            return None
    
    def _autocorrelation_method(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """Autocorrelation-based pitch detection"""
        try:
            # Calculate autocorrelation
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks = self._find_peaks(autocorr)
            
            if len(peaks) == 0:
                return None
            
            # Convert lag to frequency
            lags = peaks / sample_rate
            frequencies = 1.0 / lags
            
            # Filter frequencies within range
            valid_freqs = frequencies[(frequencies >= self.min_freq) & (frequencies <= self.max_freq)]
            
            if len(valid_freqs) == 0:
                return None
            
            # Return the lowest valid frequency (fundamental)
            return np.min(valid_freqs)
            
        except Exception as e:
            print(f"Autocorrelation error: {e}")
            return None
    
    def _harmonic_product_spectrum(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """Harmonic Product Spectrum method"""
        try:
            # Compute FFT
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft)
            
            # Use only positive frequencies
            magnitude = magnitude[:len(magnitude)//2]
            
            # Create harmonic product spectrum
            hps = magnitude.copy()
            for harmonic in range(2, 5):  # Use harmonics 2, 3, 4
                if len(magnitude) >= harmonic:
                    # Ensure arrays have the same length for multiplication
                    harmonic_length = len(magnitude) // harmonic
                    hps[:harmonic_length] *= magnitude[::harmonic][:harmonic_length]
            
            # Find peak in HPS
            peak_idx = np.argmax(hps)
            frequency = peak_idx * sample_rate / len(audio_data)
            
            # Validate frequency range
            if self.min_freq <= frequency <= self.max_freq:
                return frequency
            
            return None
            
        except Exception as e:
            print(f"HPS error: {e}")
            return None
    
    def _cepstrum_method(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """Cepstrum-based pitch detection"""
        try:
            # Compute FFT
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft)
            
            # Apply log
            log_magnitude = np.log(magnitude + 1e-8)
            
            # Compute inverse FFT (cepstrum)
            cepstrum = np.fft.ifft(log_magnitude)
            cepstrum = np.abs(cepstrum)
            
            # Find peaks in cepstrum
            peaks = self._find_peaks(cepstrum[:len(cepstrum)//2])
            
            if len(peaks) == 0:
                return None
            
            # Convert quefrency to frequency
            quefrencies = peaks / sample_rate
            frequencies = 1.0 / quefrencies
            
            # Filter frequencies within range
            valid_freqs = frequencies[(frequencies >= self.min_freq) & (frequencies <= self.max_freq)]
            
            if len(valid_freqs) == 0:
                return None
            
            # Return the lowest valid frequency
            return np.min(valid_freqs)
            
        except Exception as e:
            print(f"Cepstrum error: {e}")
            return None
    
    def _yin_algorithm(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """YIN algorithm for pitch detection"""
        try:
            # YIN algorithm implementation
            frame_length = min(len(audio_data), self.frame_length)
            audio_frame = audio_data[:frame_length]
            
            # Step 1: Difference function
            diff = np.zeros(frame_length)
            for tau in range(1, frame_length):
                diff[tau] = np.sum((audio_frame[tau:] - audio_frame[:-tau])**2)
            
            # Step 2: Normalized difference function
            running_sum = np.cumsum(diff)
            normalized_diff = np.zeros_like(diff)
            normalized_diff[1:] = diff[1:] / (running_sum[1:] / np.arange(1, frame_length))
            
            # Step 3: Absolute threshold
            threshold = 0.1
            for tau in range(1, frame_length):
                if normalized_diff[tau] < threshold:
                    # Find the minimum in the neighborhood
                    while (tau + 1 < frame_length and 
                           normalized_diff[tau + 1] < normalized_diff[tau]):
                        tau += 1
                    
                    # Convert to frequency
                    frequency = sample_rate / tau
                    
                    # Validate frequency range
                    if self.min_freq <= frequency <= self.max_freq:
                        return frequency
                    break
            
            return None
            
        except Exception as e:
            print(f"YIN algorithm error: {e}")
            return None
    
    def _find_peaks(self, signal: np.ndarray, min_distance: int = 10) -> np.ndarray:
        """Find peaks in a signal with minimum distance constraint"""
        try:
            peaks = []
            for i in range(1, len(signal) - 1):
                if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and 
                    signal[i] > np.max(signal) * self.min_peak_ratio):
                    # Check minimum distance from previous peaks
                    if not peaks or i - peaks[-1] >= min_distance:
                        peaks.append(i)
            
            return np.array(peaks)
            
        except Exception as e:
            print(f"Peak finding error: {e}")
            return np.array([])
    
    def freq_to_note(self, freq: float) -> str:
        """Convert frequency to Scientific Pitch Notation (SPN)"""
        if freq <= 0:
            return "N/A"
        
        # Calculate semitones from A4 (440 Hz)
        semitones = 12 * np.log2(freq / self.A4)
        
        # Round to nearest semitone
        semitones_rounded = round(semitones)
        
        # Calculate note and octave
        note_index = (semitones_rounded + 9) % 12  # A is index 9
        octave = (semitones_rounded + 9) // 12 + 4  # A4 is octave 4
        
        # Handle edge cases for very low/high frequencies
        if octave < 0:
            octave = 0
        elif octave > 9:
            octave = 9
        
        return f"{self.note_names[note_index]}{octave}"
    
    def get_pitch_confidence(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate confidence score for pitch detection"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Calculate signal-to-noise ratio
            signal_power = np.mean(audio_mono**2)
            noise_floor = np.percentile(audio_mono**2, 10)
            snr = 10 * np.log10(signal_power / (noise_floor + 1e-8))
            
            # Normalize SNR to 0-1 range
            confidence = min(1.0, max(0.0, (snr + 20) / 40))
            
            return confidence
            
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return 0.0

class DemucsProcessor:
    """Handles Demucs stem separation"""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_rate = 44100
    
    def load_model(self, model_name: str = "htdemucs"):
        """Load the Demucs model"""
        try:
            logging.info(f"Loading Demucs model: {model_name}")
            # Use standard Demucs model loading
            self.model = get_model(model_name)
            self.model.to(self.device)
            logging.info(f"Successfully loaded Demucs model: {model_name}")
            print(f"Loaded Demucs model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading Demucs model: {e}")
            print(f"Error loading Demucs model: {e}")
            raise
    
    def separate_stems(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Separate audio into stems using Demucs"""
        if self.model is None:
            self.load_model()
        
        try:
            logging.info(f"Separating stems for: {audio_path}")
            logging.info(f"Output directory: {output_dir}")
            print(f"Separating stems for: {audio_path}")
            print(f"Output directory: {output_dir}")
            
            # Load audio using librosa to ensure correct format
            wav, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
            logging.info(f"Loaded audio with librosa: shape={wav.shape}, sample_rate={sr}")
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
            logging.info(f"Converted to torch tensor: shape={wav_tensor.shape}")
            print(f"Converted to torch tensor: shape={wav_tensor.shape}")
            
            # Normalize
            ref = wav_tensor.mean(0)
            wav_tensor = (wav_tensor - ref.mean()) / ref.std()
            
            # Separate stems
            logging.info("Applying Demucs model...")
            print("Applying Demucs model...")
            sources = apply_model(self.model, wav_tensor[None], device=self.device)[0]
            sources = sources * ref.std() + ref.mean()
            logging.info(f"Separated into {len(sources)} stems")
            print(f"Separated into {len(sources)} stems")
            
            # Save stems
            stem_paths = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            for i, (source, name) in enumerate(zip(sources, stem_names)):
                stem_path = os.path.join(output_dir, f"{name}.wav")
                logging.info(f"Saving {name} stem to: {stem_path}")
                print(f"Saving {name} stem to: {stem_path}")
                
                # Convert back to numpy and save
                source_np = source.numpy()
                import soundfile as sf
                sf.write(stem_path, source_np.T, self.sample_rate)
                
                # Verify the file was created and has content
                if os.path.exists(stem_path) and os.path.getsize(stem_path) > 0:
                    stem_paths[name] = stem_path
                    logging.info(f"✓ {name} stem saved successfully ({os.path.getsize(stem_path)} bytes)")
                    print(f"✓ {name} stem saved successfully ({os.path.getsize(stem_path)} bytes)")
                else:
                    logging.warning(f"✗ {name} stem file is empty or missing")
                    print(f"✗ {name} stem file is empty or missing")
            
            logging.info(f"Successfully created {len(stem_paths)} stem files")
            print(f"Successfully created {len(stem_paths)} stem files")
            return stem_paths
            
        except Exception as e:
            logging.error(f"Error in stem separation: {e}")
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
        self.similarity_checker = SampleSimilarityChecker()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("MagicSample")
        self.setGeometry(100, 100, 1000, 700)  # Larger size for tabs
        
        # Set application icon
        icon_path = resource_path("MagicSample_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            logging.info(f"Application icon loaded from: {icon_path}")
        else:
            logging.warning(f"Application icon not found at: {icon_path}")
        
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
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Main processing tab
        main_tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(10)
        
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
        main_layout.addWidget(file_group)
        
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
        
        # Similarity slider
        similarity_label = QLabel("Sample Similarity Threshold (0% = identical, 100% = very different)")
        similarity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(similarity_label, 4, 0, 1, 2)
        
        self.similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_slider.setRange(0, 100)  # 0% to 100%
        self.similarity_slider.setValue(80)  # Default 80%
        self.similarity_slider.setTickInterval(10)
        options_layout.addWidget(self.similarity_slider, 5, 0, 1, 2)
        
        # Sample timeout input
        timeout_label = QLabel("Sample Timeout (ms):")
        timeout_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(timeout_label, 6, 0)
        
        self.timeout_input = QLineEdit()
        self.timeout_input.setPlaceholderText("2000")
        self.timeout_input.setText("2000")  # Default 2000ms
        self.timeout_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timeout_input.setFixedWidth(100)
        options_layout.addWidget(self.timeout_input, 6, 1)
        
        # Output format and drumkit name in a row
        format_label = QLabel("Output Format:")
        format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(format_label, 7, 0)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['WAV', 'FLAC', 'OGG'])
        options_layout.addWidget(self.format_combo, 7, 1)
        
        drumkit_label = QLabel("Drumkit Name:")
        drumkit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(drumkit_label, 8, 0)
        
        self.drumkit_name_edit = QLineEdit()
        self.drumkit_name_edit.setPlaceholderText("MyDrumkit")
        self.drumkit_name_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(self.drumkit_name_edit, 8, 1)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
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
        main_layout.addWidget(progress_group)
        
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
        
        main_layout.addLayout(button_layout)
        main_tab.setLayout(main_layout)
        
        # Logging tab
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        # Log controls
        log_controls = QHBoxLayout()
        
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        log_controls.addWidget(self.clear_log_button)
        
        self.save_log_button = QPushButton("Save Log")
        self.save_log_button.clicked.connect(self.save_log)
        log_controls.addWidget(self.save_log_button)
        
        log_layout.addLayout(log_controls)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        log_tab.setLayout(log_layout)
        
        # Help tab
        help_tab = QWidget()
        help_layout = QVBoxLayout()
        
        # Help text area
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setFont(QFont("Arial", 10))
        
        # Create comprehensive help content
        help_content = """
<h2>MagicSample - Audio Sample Extraction Tool</h2>

<h3>📁 File Selection</h3>
<p><b>Input Audio File:</b> Select your source audio file (WAV, MP3, FLAC, OGG, M4A). This is the file that will be processed into individual samples.</p>
<p><b>Output Directory:</b> Choose where your drumkit folder will be created. The program will create a new folder with your drumkit name inside this directory.</p>

<h3>⚙️ Processing Options</h3>

<h4>🔧 Core Features</h4>
<p><b>Split to Stems:</b> Uses Demucs AI to separate your audio into 4 stems: Drums, Bass, Vocals, and Other. When enabled, samples are organized into separate folders for each stem type. When disabled, processes the whole file as one category.</p>

<p><b>Detect BPM:</b> Analyzes the audio to find the tempo (beats per minute). The detected BPM is included in sample filenames (e.g., "Kick_001_120BPM_C2.WAV").</p>

<p><b>Detect Pitch:</b> Uses advanced multi-algorithm pitch detection to find the musical note of each sample. Results are included in filenames using Scientific Pitch Notation (e.g., "C5", "F#3").</p>

<p><b>Classify Drums:</b> Automatically categorizes drum samples into subfolders: Kick, HiHat, and Perc (percussion). Only applies to the Drums stem.</p>

<h4>🎛️ Sensitivity Controls</h4>
<p><b>Sample Detection Sensitivity:</b> Controls how the program detects individual samples within the audio.</p>
<ul>
<li><b>Lower values (5-15):</b> More sensitive - detects more samples, including quieter and shorter sounds</li>
<li><b>Higher values (20-30):</b> Less sensitive - detects fewer samples, only the most prominent sounds</li>
<li><b>Recommended:</b> Start with 15, adjust based on your audio content</li>
</ul>

<p><b>Sample Similarity Threshold:</b> Controls how similar samples need to be before one is rejected as a duplicate.</p>
<ul>
<li><b>0%:</b> Very strict - only identical samples are considered duplicates</li>
<li><b>50%:</b> Balanced - moderately similar samples are rejected</li>
<li><b>80%:</b> Default - similar samples are rejected (recommended)</li>
<li><b>100%:</b> Very permissive - only very different samples are kept</li>
</ul>

<h4>⏱️ Performance Settings</h4>
<p><b>Sample Timeout (ms):</b> Maximum time allowed for processing each individual sample. If a sample takes longer than this time to process (pitch detection, similarity check, etc.), it will be skipped but any information gathered before the timeout will still be included in the filename.</p>
<ul>
<li><b>1000ms:</b> Fast processing, may skip complex samples</li>
<li><b>2000ms:</b> Default - good balance of speed and accuracy</li>
<li><b>5000ms:</b> Slower but more thorough processing</li>
</ul>

<h4>📁 Output Settings</h4>
<p><b>Output Format:</b> Choose the audio format for your samples.</p>
<ul>
<li><b>WAV:</b> Uncompressed, highest quality, larger file sizes</li>
<li><b>FLAC:</b> Lossless compression, high quality, smaller than WAV</li>
<li><b>OGG:</b> Lossy compression, smaller file sizes, good quality</li>
</ul>

<p><b>Drumkit Name:</b> The name of the folder that will be created to contain all your samples. This will be the main folder containing all your organized samples.</p>

<h3>📂 Output Structure</h3>

<p><b>With "Split to Stems" enabled:</b></p>
<pre>
YourDrumkit/
├── Drums/
│   ├── Kick/
│   │   ├── Kick_001_120BPM_C2.WAV
│   │   └── Kick_002_120BPM_F1.WAV
│   ├── HiHat/
│   │   ├── HiHat_001_120BPM_G#4.WAV
│   │   └── HiHat_002_120BPM_A4.WAV
│   └── Perc/
│       ├── Perc_001_120BPM_D3.WAV
│       └── Perc_002_120BPM_E3.WAV
├── Bass/
│   ├── Bass_001_120BPM_F1.WAV
│   └── Bass_002_120BPM_C2.WAV
├── Vocals/
│   └── Vocals_120BPM.WAV
└── Other/
    ├── Other_001_120BPM_A3.WAV
    └── Other_002_120BPM_D4.WAV
</pre>

<p><b>With "Split to Stems" disabled:</b></p>
<pre>
YourDrumkit/
└── Samples/
    ├── Sample_001_120BPM_C2.WAV
    ├── Sample_002_120BPM_F1.WAV
    └── Sample_003_120BPM_G#4.WAV
</pre>

<h3>🎵 Filename Format</h3>
<p>Samples are named using this pattern:</p>
<p><code>[Type]_[Number]_[BPM]BPM_[Pitch].[Format]</code></p>

<p><b>Examples:</b></p>
<ul>
<li><code>Kick_001_120BPM_C2.WAV</code> - Kick drum, sample 1, 120 BPM, C2 pitch</li>
<li><code>Bass_002_140BPM_F1.WAV</code> - Bass, sample 2, 140 BPM, F1 pitch</li>
<li><code>Vocals_120BPM.WAV</code> - Vocals (whole file), 120 BPM, no pitch</li>
</ul>

<h3>🔍 Troubleshooting</h3>

<h4>Common Issues</h4>
<p><b>No samples created:</b> Try lowering the Sample Detection Sensitivity or increasing the timeout.</p>

<p><b>Too many similar samples:</b> Increase the Sample Similarity Threshold.</p>

<p><b>Processing is slow:</b> Reduce the Sample Timeout or disable pitch detection for faster processing.</p>

<p><b>Stem separation fails:</b> Check the Log tab for detailed error messages. The program will fall back to processing the whole file.</p>

<h4>Performance Tips</h4>
<ul>
<li>Use WAV or FLAC input files for best quality</li>
<li>Disable pitch detection if you don't need pitch information</li>
<li>Adjust sensitivity based on your audio content</li>
<li>Use the Log tab to monitor processing progress</li>
</ul>

<h3>📊 Advanced Features</h3>

<h4>Pitch Detection Algorithms</h4>
<p>The program uses 4 different algorithms and combines their results:</p>
<ul>
<li><b>Autocorrelation:</b> Time-domain periodicity detection</li>
<li><b>Harmonic Product Spectrum (HPS):</b> Frequency-domain fundamental detection</li>
<li><b>Cepstrum:</b> Frequency-domain deconvolution</li>
<li><b>YIN Algorithm:</b> Robust time-domain pitch detection</li>
</ul>

<h4>Similarity Detection</h4>
<p>Compares samples using multiple audio features:</p>
<ul>
<li>MFCC (Mel-frequency cepstral coefficients)</li>
<li>Spectral centroid (brightness)</li>
<li>Spectral rolloff (frequency distribution)</li>
<li>RMS energy (loudness)</li>
<li>Zero crossing rate (noisiness)</li>
</ul>

<h4>Drum Classification</h4>
<p>Classifies drum samples based on frequency characteristics:</p>
<ul>
<li><b>Kick:</b> Low frequency content (typically below 200Hz)</li>
<li><b>HiHat:</b> High frequency content (typically above 2000Hz)</li>
<li><b>Perc:</b> Mid-frequency content (everything else)</li>
</ul>

<p><i>For more detailed information and troubleshooting, check the Log tab during processing.</i></p>
        """
        
        self.help_text.setHtml(help_content)
        help_layout.addWidget(self.help_text)
        
        help_tab.setLayout(help_layout)
        
        # Add tabs to widget
        self.tab_widget.addTab(main_tab, "Processing")
        self.tab_widget.addTab(help_tab, "Help")
        self.tab_widget.addTab(log_tab, "Log")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging to the GUI"""
        # Create custom log handler
        self.log_handler = LogHandler()
        self.log_handler.log_signal.connect(self.add_log_message)
        
        # Configure formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
        
        # Log initial message
        self.add_log_message("MagicSample started - Logging initialized")
    
    def add_log_message(self, message):
        """Add a message to the log display"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def clear_log(self):
        """Clear the log display"""
        self.log_text.clear()
        self.add_log_message("Log cleared")
    
    def save_log(self):
        """Save the log to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log File", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.add_log_message(f"Log saved to: {file_path}")
            except Exception as e:
                self.add_log_message(f"Error saving log: {e}")
    
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
        
        # Validate timeout input
        try:
            timeout_value = int(self.timeout_input.text() or "2000")
            if timeout_value <= 0:
                QMessageBox.warning(self, "Error", "Timeout must be a positive number")
                return
        except ValueError:
            QMessageBox.warning(self, "Error", "Timeout must be a valid number")
            return
        
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
            self.sensitivity_slider.value(),
            self.similarity_slider.value() / 100.0,  # Convert percentage to 0.0-1.0 range
            timeout_value
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
                 split_stems, detect_bpm, detect_pitch, classify_drums, sensitivity, similarity_threshold, timeout_ms):
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
        self.similarity_threshold = similarity_threshold
        self.timeout_ms = timeout_ms
        self.stop_flag = False
        
        # Initialize processors
        self.demucs_processor = DemucsProcessor()
        self.drum_classifier = DrumClassifier()
        self.pitch_detector = PitchDetector()
        self.similarity_checker = SampleSimilarityChecker(similarity_threshold)
    
    def run_with_timeout(self, func, *args, **kwargs):
        """Run a function with timeout handling"""
        import signal
        import threading
        import time
        
        result = [None]
        exception = [None]
        completed = [False]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
            finally:
                completed[0] = True
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait for completion or timeout
        start_time = time.time()
        while not completed[0] and (time.time() - start_time) * 1000 < self.timeout_ms:
            time.sleep(0.01)  # Small sleep to prevent busy waiting
        
        if not completed[0]:
            # Timeout occurred
            return None, "timeout"
        elif exception[0]:
            # Exception occurred
            return None, str(exception[0])
        else:
            # Success
            return result[0], "success"
    
    def run(self):
        """Main processing function with improved progress updates."""
        try:
            logging.info("Starting audio processing...")
            self.status_updated.emit("Loading audio file...")
            self.progress_updated.emit(5)
            audio_data, sample_rate = librosa.load(self.input_path, sr=None, mono=False)
            logging.info(f"Loaded audio: {audio_data.shape}, sample rate: {sample_rate}")
            print(f"Loaded audio: {audio_data.shape}, sample rate: {sample_rate}")
            bpm = None
            if self.detect_bpm:
                self.status_updated.emit("Detecting BPM...")
                self.progress_updated.emit(10)
                logging.info("Detecting BPM...")
                bpm = self.detect_bpm_from_audio(audio_data, sample_rate)
                logging.info(f"Detected BPM: {bpm}")
                print(f"Detected BPM: {bpm}")
            drumkit_path = os.path.join(self.output_path, self.drumkit_name.capitalize())
            os.makedirs(drumkit_path, exist_ok=True)
            logging.info(f"Created drumkit directory: {drumkit_path}")
            if self.split_stems:
                self.status_updated.emit("Separating stems with Demucs...")
                self.progress_updated.emit(20)
                logging.info("Starting stem separation with Demucs...")
                temp_dir = os.path.join(drumkit_path, "temp_stems")
                os.makedirs(temp_dir, exist_ok=True)
                try:
                    stem_paths = self.demucs_processor.separate_stems(self.input_path, temp_dir)
                    logging.info(f"Created stems: {list(stem_paths.keys())}")
                    print(f"Created stems: {list(stem_paths.keys())}")
                    stem_count = len(stem_paths)
                    for i, (stem_name, stem_path) in enumerate(stem_paths.items()):
                        if self.stop_flag:
                            break
                        progress = 20 + (i * 60 // stem_count)
                        self.status_updated.emit(f"Processing {stem_name} stem ({i+1}/{stem_count})...")
                        self.progress_updated.emit(progress)
                        logging.info(f"Processing {stem_name} stem ({i+1}/{stem_count})...")
                        if not os.path.exists(stem_path) or os.path.getsize(stem_path) == 0:
                            logging.warning(f"Stem file {stem_path} is empty or missing")
                            print(f"Warning: Stem file {stem_path} is empty or missing")
                            continue
                        stem_audio, stem_sr = librosa.load(stem_path, sr=None, mono=False)
                        logging.info(f"Loaded {stem_name} stem: {stem_audio.shape}")
                        print(f"Loaded {stem_name} stem: {stem_audio.shape}")
                        stem_dir = os.path.join(drumkit_path, stem_name.capitalize())
                        os.makedirs(stem_dir, exist_ok=True)
                        if stem_name == "vocals":
                            vocals_filename = f"Vocals_{bpm}BPM.{self.output_format.upper()}" if bpm else f"Vocals.{self.output_format.upper()}"
                            vocals_path = os.path.join(stem_dir, vocals_filename)
                            self.save_audio_sample(stem_audio, stem_sr, vocals_path)
                            logging.info(f"Saved vocals as whole file: {vocals_filename}")
                            print(f"Saved vocals as whole file: {vocals_filename}")
                        elif stem_name == "drums":
                            self.status_updated.emit("Splitting drums into one-shots...")
                            logging.info("Processing drums with subfolder classification...")
                            self.process_drums_with_subfolders(stem_audio, stem_sr, stem_dir, bpm)
                        else:
                            self.status_updated.emit(f"Detecting one-shots in {stem_name}...")
                            logging.info(f"Processing {stem_name} into individual samples...")
                            sample_count = self.process_stem_into_samples(stem_audio, stem_sr, stem_dir, stem_name.capitalize(), bpm)
                            logging.info(f"Created {sample_count} samples for {stem_name}")
                            print(f"Created {sample_count} samples for {stem_name}")
                    import shutil
                    shutil.rmtree(temp_dir)
                    logging.info("Cleaned up temporary stem files")
                except Exception as e:
                    logging.error(f"Error in stem separation: {e}")
                    print(f"Error in stem separation: {e}")
                    self.status_updated.emit(f"Stem separation failed: {str(e)}")
                    self.status_updated.emit("Falling back to processing whole file...")
                    logging.info("Falling back to processing whole file...")
                    samples_dir = os.path.join(drumkit_path, "Samples")
                    os.makedirs(samples_dir, exist_ok=True)
                    sample_count = self.process_stem_into_samples(audio_data, sample_rate, samples_dir, "Sample", bpm)
                    logging.info(f"Created {sample_count} samples from whole file")
                    print(f"Created {sample_count} samples from whole file")
            else:
                self.status_updated.emit("Processing audio into samples...")
                self.progress_updated.emit(40)
                logging.info("Processing whole file into samples...")
                samples_dir = os.path.join(drumkit_path, "Samples")
                os.makedirs(samples_dir, exist_ok=True)
                sample_count = self.process_stem_into_samples(audio_data, sample_rate, samples_dir, "Sample", bpm)
                logging.info(f"Created {sample_count} samples from whole file")
                print(f"Created {sample_count} samples from whole file")
            self.progress_updated.emit(100)
            self.status_updated.emit("Drumkit creation completed!")
            logging.info("Drumkit creation completed!")
        except Exception as e:
            logging.error(f"Processing error: {e}")
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
        """Process a stem into individual samples, with improved one-shot detection for bass/other."""
        try:
            logging.info(f"Processing {stem_name} into samples...")
            # Convert to mono for sample detection
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data

            min_duration = 0.05  # 50ms
            sample_count = 0
            sample_boundaries = []
            use_onset = stem_name.lower() in ["bass", "other"]

            if use_onset:
                # Verbose progress for onset detection
                if hasattr(self, 'status_updated'):
                    self.status_updated.emit(f"Detecting note onsets in {stem_name} stem...")
                onset_frames = librosa.onset.onset_detect(y=audio_mono, sr=sample_rate, backtrack=True)
                onset_samples = librosa.frames_to_samples(onset_frames)
                # Add end of audio as last boundary
                if len(onset_samples) == 0 or onset_samples[-1] < len(audio_mono):
                    onset_samples = np.append(onset_samples, len(audio_mono))
                # Build (start, end) pairs
                for i in range(len(onset_samples) - 1):
                    start = onset_samples[i]
                    end = onset_samples[i + 1]
                    if end - start > 0:
                        sample_boundaries.append((start, end))
                if hasattr(self, 'status_updated'):
                    self.status_updated.emit(f"Found {len(sample_boundaries)} note onsets in {stem_name} stem.")
            else:
                # Try different sensitivity levels if no samples are found
                sensitivities_to_try = [self.sensitivity, self.sensitivity + 5, self.sensitivity + 10, 30]
                for sensitivity in sensitivities_to_try:
                    sample_boundaries = librosa.effects.split(audio_mono, top_db=sensitivity)
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Tried sensitivity {sensitivity}dB: found {len(sample_boundaries)} potential samples in {stem_name}.")
                    if len(sample_boundaries) > 0:
                        break
                if len(sample_boundaries) == 0:
                    sample_boundaries = [(0, len(audio_mono))]

            total_samples = len(sample_boundaries)
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
                if sample_duration < min_duration and total_samples > 1:
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Skipping {stem_name} sample {i+1}: too short ({sample_duration*1000:.1f}ms)")
                    continue
                # Skip very quiet samples (unless it's the only sample)
                sample_rms = np.sqrt(np.mean(sample_audio**2))
                if sample_rms < 0.001 and total_samples > 1:
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Skipping {stem_name} sample {i+1}: too quiet (RMS: {sample_rms:.6f})")
                    continue
                # Verbose progress for each sample
                if hasattr(self, 'status_updated'):
                    self.status_updated.emit(f"Saving {stem_name} sample {i+1}/{total_samples} (duration: {sample_duration:.2f}s)...")
                if hasattr(self, 'progress_updated'):
                    self.progress_updated.emit(40 + int(50 * (i+1) / max(1, total_samples)))
                # Generate filename with timeout handling
                filename_parts = [f"{stem_name.capitalize()}_{i+1:03d}"]
                if bpm:
                    filename_parts.append(f"{bpm}BPM")
                
                # Pitch detection with timeout
                pitch = None
                if self.detect_pitch:
                    pitch_result, pitch_status = self.run_with_timeout(
                        self.pitch_detector.detect_pitch, sample_audio, sample_rate
                    )
                    if pitch_status == "success" and pitch_result and pitch_result != "N/A":
                        pitch = pitch_result
                        filename_parts.append(pitch)
                    elif pitch_status == "timeout":
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Pitch detection timeout for {stem_name} sample {i+1}, continuing...")
                    else:
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Pitch detection failed for {stem_name} sample {i+1}: {pitch_status}")
                
                filename = "_".join(filename_parts) + f".{self.output_format.upper()}"
                filepath = os.path.join(output_dir, filename)
                
                # Similarity check with timeout
                similarity_result, similarity_status = self.run_with_timeout(
                    self.similarity_checker.is_similar_to_existing, sample_audio, sample_rate, stem_name.capitalize()
                )
                
                if similarity_status == "timeout":
                    logging.warning(f"Similarity check timeout for {stem_name} sample {i+1}, saving sample...")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check timeout for {stem_name} sample {i+1}, saving sample...")
                    # Save sample even if similarity check times out
                    self.save_audio_sample(sample_audio, sample_rate, filepath)
                    sample_count += 1
                elif similarity_status == "success" and similarity_result:
                    logging.info(f"Skipping {stem_name} sample {i+1}: too similar to existing samples.")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Skipping {stem_name} sample {i+1}: too similar to existing samples.")
                    continue
                elif similarity_status == "success" and not similarity_result:
                    # Not similar, save the sample
                    logging.info(f"Saving {stem_name} sample {i+1}: {filename}")
                    self.save_audio_sample(sample_audio, sample_rate, filepath)
                    sample_count += 1
                else:
                    # Similarity check failed, save sample anyway
                    logging.warning(f"Similarity check failed for {stem_name} sample {i+1}: {similarity_status}, saving sample...")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check failed for {stem_name} sample {i+1}: {similarity_status}, saving sample...")
                    self.save_audio_sample(sample_audio, sample_rate, filepath)
                    sample_count += 1
            logging.info(f"Successfully created {sample_count} samples for {stem_name}")
            if hasattr(self, 'status_updated'):
                self.status_updated.emit(f"Successfully created {sample_count} samples for {stem_name}.")
            return sample_count
        except Exception as e:
            logging.error(f"Error processing stem {stem_name}: {e}")
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
                
                # Generate filename with timeout handling
                filename_parts = [f"{prefix}_{i+1:03d}"]
                
                if bpm:
                    filename_parts.append(f"{bpm}BPM")
                
                # Pitch detection with timeout
                if self.detect_pitch:
                    pitch_result, pitch_status = self.run_with_timeout(
                        self.pitch_detector.detect_pitch, sample_audio, sample_rate
                    )
                    if pitch_status == "success" and pitch_result and pitch_result != "N/A":
                        filename_parts.append(pitch_result)
                    elif pitch_status == "timeout":
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Pitch detection timeout for {drum_type} sample {i+1}, continuing...")
                    else:
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Pitch detection failed for {drum_type} sample {i+1}: {pitch_status}")
                
                # Create final filename
                filename = "_".join(filename_parts) + f".{self.output_format.capitalize()}"
                filepath = os.path.join(target_dir, filename)
                
                # Similarity check with timeout
                similarity_result, similarity_status = self.run_with_timeout(
                    self.similarity_checker.is_similar_to_existing, sample_audio, sample_rate, drum_type
                )
                
                if similarity_status == "timeout":
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check timeout for {drum_type} sample {i+1}, saving sample...")
                    # Save sample even if similarity check times out
                    self.save_audio_sample(sample_audio, sample_rate, filepath)
                    sample_count += 1
                elif similarity_status == "success" and similarity_result:
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Skipping {drum_type} sample {i+1}: too similar to existing samples.")
                    continue
                elif similarity_status == "success" and not similarity_result:
                    # Not similar, save the sample
                    self.save_audio_sample(sample_audio, sample_rate, filepath)
                    sample_count += 1
                else:
                    # Similarity check failed, save sample anyway
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check failed for {drum_type} sample {i+1}: {similarity_status}, saving sample...")
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
                    "similarity_threshold": self.similarity_threshold,
                    "sample_timeout_ms": self.timeout_ms,
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
    app.setWindowIcon(QIcon('build/MagicSample_icon.ico'))
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 