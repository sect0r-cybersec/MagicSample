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
- Fast similarity comparison to avoid duplicate samples
- Sample processing timeout protection
- Hybrid transient detection and energy-based slicing
- Minimum amplitude threshold filtering
"""
__version__ = '0.0.9'

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

# YouTube downloading
import yt_dlp
import tempfile
import re

# Advanced audio analysis imports
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter1d

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
    """Fast similarity checker using spectral centroid + RMS energy for quick comparison"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.saved_samples = {}  # category -> list of (rms, centroid, zcr) tuples
    
    def calculate_fast_features(self, audio_data: np.ndarray, sample_rate: int) -> tuple:
        """Calculate fast features for similarity comparison"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # 1. RMS Energy (fastest)
            rms = np.sqrt(np.mean(audio_mono**2))
            
            # 2. Spectral Centroid (brightness)
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate))
            
            # 3. Zero Crossing Rate (transient content)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_mono))
            
            return (rms, centroid, zcr)
            
        except Exception as e:
            print(f"Error calculating fast features: {e}")
            return (0.0, 0.0, 0.0)
    
    def fast_similarity_check(self, features1: tuple, features2: tuple) -> bool:
        """Fast similarity check using multiple lightweight features"""
        try:
            rms1, centroid1, zcr1 = features1
            rms2, centroid2, zcr2 = features2
            
            # 1. RMS Energy difference
            rms_diff = abs(rms1 - rms2) / max(rms1, rms2) if max(rms1, rms2) > 0 else 1.0
            
            # Early exit if energy is very different (saves computation)
            if rms_diff > 0.3:  # 30% energy difference
                return False
            
            # 2. Spectral Centroid difference (brightness)
            centroid_diff = abs(centroid1 - centroid2) / max(centroid1, centroid2) if max(centroid1, centroid2) > 0 else 1.0
            
            # 3. Zero Crossing Rate difference (transient content)
            zcr_diff = abs(zcr1 - zcr2) / max(zcr1, zcr2) if max(zcr1, zcr2) > 0 else 1.0
            
            # Combined similarity score (lower = more similar)
            total_diff = (rms_diff + centroid_diff + zcr_diff) / 3
            
            # Convert to similarity (higher = more similar)
            similarity = 1.0 - total_diff
            
            return similarity > self.similarity_threshold
            
        except Exception as e:
            print(f"Error in fast similarity check: {e}")
            return False
    
    def is_similar_to_existing(self, audio_data: np.ndarray, sample_rate: int, category: str) -> bool:
        """Check if a sample is too similar to existing samples in the same category"""
        try:
            # Calculate fast features for the current sample
            current_features = self.calculate_fast_features(audio_data, sample_rate)
            
            # Initialize category if it doesn't exist
            if category not in self.saved_samples:
                self.saved_samples[category] = []
                return False  # First sample in category is always allowed
            
            # Compare with existing samples in the same category
            for existing_features in self.saved_samples[category]:
                if self.fast_similarity_check(current_features, existing_features):
                    return True  # Too similar to existing sample
            
            # Not similar to any existing samples, add to saved samples
            self.saved_samples[category].append(current_features)
            return False
            
        except Exception as e:
            print(f"Error in similarity check: {e}")
            return False  # If error, allow the sample
    
    def set_threshold(self, threshold: float):
        """Update the similarity threshold"""
        self.similarity_threshold = threshold

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

class DominantFrequencyDetector:
    """Advanced dominant frequency detection using multiple algorithms for robust results"""
    
    def __init__(self):
        self.min_freq = 20  # Lower bound for dominant frequency detection
        self.max_freq = 8000  # Upper bound for dominant frequency detection
        self.frame_length = 2048
        self.hop_length = 512
        
        # Note frequencies for A4 = 440Hz reference
        self.A4 = 440.0
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Confidence thresholds
        self.min_confidence = 0.3
        self.min_peak_ratio = 0.05  # Lower threshold for dominant frequency detection
        
        # Spectral analysis parameters
        self.window_length = 1024
        self.overlap = 0.5
        self.n_fft = 2048
    
    def detect_dominant_frequency(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Detect the dominant frequency using multiple algorithms based on research"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Normalize audio
            audio_mono = audio_mono / (np.max(np.abs(audio_mono)) + 1e-8)
            
            # Apply multiple dominant frequency detection methods
            results = []
            
            # Method 1: FFT-based dominant frequency (primary method from research)
            fft_dominant = self._fft_dominant_frequency(audio_mono, sample_rate)
            if fft_dominant:
                results.append(fft_dominant)
            
            # Method 2: Welch's method for spectral density estimation
            welch_dominant = self._welch_dominant_frequency(audio_mono, sample_rate)
            if welch_dominant:
                results.append(welch_dominant)
            
            # Method 3: Autocorrelation with peak detection
            autocorr_dominant = self._autocorrelation_dominant_frequency(audio_mono, sample_rate)
            if autocorr_dominant:
                results.append(autocorr_dominant)
            
            # Method 4: Combined approach (FFT + Welch + peak validation)
            combined_dominant = self._combined_dominant_frequency(audio_mono, sample_rate)
            if combined_dominant:
                results.append(combined_dominant)
            
            # Combine results using weighted approach based on power/amplitude
            if not results:
                return None
            
            # Find the most dominant frequency (highest power)
            if len(results) >= 2:
                # Use the frequency that appears most often or has highest confidence
                # For dominant frequency, we prioritize the one with highest spectral power
                dominant_freq = self._select_most_dominant_frequency(results, audio_mono, sample_rate)
            else:
                dominant_freq = results[0]
            
            # Convert to note name
            note_name = self.freq_to_note(dominant_freq)
            return note_name
            
        except Exception as e:
            print(f"Dominant frequency detection error: {e}")
            return None
    
    def _fft_dominant_frequency(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """FFT-based dominant frequency detection (primary method from research)"""
        try:
            # Apply windowing to reduce spectral leakage
            window = np.hanning(len(audio_data))
            windowed_audio = audio_data * window
            
            # Compute FFT
            fft = np.fft.fft(windowed_audio, n=self.n_fft)
            magnitude = np.abs(fft)
            
            # Use only positive frequencies
            magnitude = magnitude[:len(magnitude)//2]
            
            # Create frequency array
            freqs = np.fft.fftfreq(self.n_fft, 1/sample_rate)[:len(magnitude)]
            
            # Filter frequencies within range
            valid_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
            valid_freqs = freqs[valid_mask]
            valid_magnitude = magnitude[valid_mask]
            
            if len(valid_freqs) == 0:
                return None
            
            # Find the frequency with maximum power (dominant frequency)
            max_idx = np.argmax(valid_magnitude)
            dominant_freq = valid_freqs[max_idx]
            
            # Validate that the peak is significant enough
            peak_power = valid_magnitude[max_idx]
            avg_power = np.mean(valid_magnitude)
            
            if peak_power > avg_power * 2:  # Peak should be at least 2x average
                return dominant_freq
            
            return None
            
        except Exception as e:
            print(f"FFT dominant frequency error: {e}")
            return None
    
    def _welch_dominant_frequency(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """Welch's method for spectral density estimation and dominant frequency detection"""
        try:
            # Parameters for Welch's method
            nperseg = min(self.window_length, len(audio_data) // 4)
            noverlap = int(nperseg * self.overlap)
            
            # Apply Welch's method
            freqs, psd = signal.welch(
                audio_data, 
                fs=sample_rate, 
                nperseg=nperseg, 
                noverlap=noverlap,
                nfft=self.n_fft
            )
            
            # Filter frequencies within range
            valid_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
            valid_freqs = freqs[valid_mask]
            valid_psd = psd[valid_mask]
            
            if len(valid_freqs) == 0:
                return None
            
            # Find the frequency with maximum power spectral density
            max_idx = np.argmax(valid_psd)
            dominant_freq = valid_freqs[max_idx]
            
            # Validate that the peak is significant enough
            peak_power = valid_psd[max_idx]
            avg_power = np.mean(valid_psd)
            
            if peak_power > avg_power * 1.5:  # Peak should be at least 1.5x average
                return dominant_freq
            
            return None
            
        except Exception as e:
            print(f"Welch dominant frequency error: {e}")
            return None
    
    def _autocorrelation_dominant_frequency(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """Autocorrelation-based dominant frequency detection"""
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
            
            # For dominant frequency, we want the frequency with highest autocorrelation peak
            # Find the peak with maximum autocorrelation value
            valid_peaks = peaks[(frequencies >= self.min_freq) & (frequencies <= self.max_freq)]
            if len(valid_peaks) == 0:
                return None
            
            # Get autocorrelation values at valid peaks
            peak_values = autocorr[valid_peaks]
            max_peak_idx = np.argmax(peak_values)
            dominant_freq = valid_freqs[max_peak_idx]
            
            return dominant_freq
            
        except Exception as e:
            print(f"Autocorrelation dominant frequency error: {e}")
            return None
    
    def _combined_dominant_frequency(self, audio_data: np.ndarray, sample_rate: int) -> Optional[float]:
        """Combined approach for robust dominant frequency detection"""
        try:
            # Apply windowing
            window = np.hanning(len(audio_data))
            windowed_audio = audio_data * window
            
            # Compute FFT
            fft = np.fft.fft(windowed_audio, n=self.n_fft)
            magnitude = np.abs(fft)
            
            # Use only positive frequencies
            magnitude = magnitude[:len(magnitude)//2]
            freqs = np.fft.fftfreq(self.n_fft, 1/sample_rate)[:len(magnitude)]
            
            # Filter frequencies within range
            valid_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
            valid_freqs = freqs[valid_mask]
            valid_magnitude = magnitude[valid_mask]
            
            if len(valid_freqs) == 0:
                return None
            
            # Find peaks in the magnitude spectrum
            peaks = self._find_peaks(valid_magnitude)
            
            if len(peaks) == 0:
                return None
            
            # Get peak frequencies and their magnitudes
            peak_freqs = valid_freqs[peaks]
            peak_magnitudes = valid_magnitude[peaks]
            
            # Find the peak with maximum magnitude (dominant frequency)
            max_idx = np.argmax(peak_magnitudes)
            dominant_freq = peak_freqs[max_idx]
            
            # Validate that the peak is significant enough
            peak_power = peak_magnitudes[max_idx]
            avg_power = np.mean(valid_magnitude)
            
            if peak_power > avg_power * 1.5:  # Peak should be at least 1.5x average
                return dominant_freq
            
            return None
            
        except Exception as e:
            print(f"Combined dominant frequency error: {e}")
            return None
    
    def _select_most_dominant_frequency(self, frequencies: list, audio_data: np.ndarray, sample_rate: int) -> float:
        """Select the most dominant frequency from multiple candidates based on spectral power"""
        try:
            if not frequencies:
                return None
            
            # Calculate spectral power for each frequency
            powers = []
            for freq in frequencies:
                # Find the closest frequency bin
                fft = np.fft.fft(audio_data, n=self.n_fft)
                magnitude = np.abs(fft)[:len(fft)//2]
                freqs = np.fft.fftfreq(self.n_fft, 1/sample_rate)[:len(magnitude)]
                
                # Find the closest frequency bin
                freq_idx = np.argmin(np.abs(freqs - freq))
                power = magnitude[freq_idx]
                powers.append(power)
            
            # Return the frequency with maximum power
            max_idx = np.argmax(powers)
            return frequencies[max_idx]
            
        except Exception as e:
            print(f"Select most dominant frequency error: {e}")
            return frequencies[0] if frequencies else 0.0
            # Fallback to median
            return np.median(frequencies)
    

    
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
    
    def __init__(self, logger=None):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_rate = 44100
        self.logger = logger
    
    def load_model(self, model_name: str = "htdemucs"):
        """Load the Demucs model"""
        try:
            if self.logger:
                self.logger.info(f"Loading Demucs model: {model_name}")
            # Use standard Demucs model loading
            self.model = get_model(model_name)
            self.model.to(self.device)
            if self.logger:
                self.logger.info(f"Successfully loaded Demucs model: {model_name}")
            print(f"Loaded Demucs model: {model_name}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading Demucs model: {e}")
            print(f"Error loading Demucs model: {e}")
            raise
    
    def separate_stems(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Separate audio into stems using Demucs"""
        if self.model is None:
            self.load_model()
        
        try:
            if self.logger:
                self.logger.info(f"Separating stems for: {audio_path}")
                self.logger.info(f"Output directory: {output_dir}")
            print(f"Separating stems for: {audio_path}")
            print(f"Output directory: {output_dir}")
            
            # Load audio using librosa to ensure correct format
            wav, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
            if self.logger:
                self.logger.info(f"Loaded audio with librosa: shape={wav.shape}, sample_rate={sr}")
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
            if self.logger:
                self.logger.info(f"Converted to torch tensor: shape={wav_tensor.shape}")
            print(f"Converted to torch tensor: shape={wav_tensor.shape}")
            
            # Normalize
            ref = wav_tensor.mean(0)
            wav_tensor = (wav_tensor - ref.mean()) / ref.std()
            
            # Separate stems
            if self.logger:
                self.logger.info("Applying Demucs model...")
            print("Applying Demucs model...")
            sources = apply_model(self.model, wav_tensor[None], device=self.device)[0]
            sources = sources * ref.std() + ref.mean()
            if self.logger:
                self.logger.info(f"Separated into {len(sources)} stems")
            print(f"Separated into {len(sources)} stems")
            
            # Save stems
            stem_paths = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            for i, (source, name) in enumerate(zip(sources, stem_names)):
                stem_path = os.path.join(output_dir, f"{name}.wav")
                if self.logger:
                    self.logger.info(f"Saving {name} stem to: {stem_path}")
                print(f"Saving {name} stem to: {stem_path}")
                
                # Convert back to numpy and save
                source_np = source.numpy()
                import soundfile as sf
                sf.write(stem_path, source_np.T, self.sample_rate)
                
                # Verify the file was created and has content
                if os.path.exists(stem_path) and os.path.getsize(stem_path) > 0:
                    stem_paths[name] = stem_path
                    if self.logger:
                        self.logger.info(f"✓ {name} stem saved successfully ({os.path.getsize(stem_path)} bytes)")
                    print(f"✓ {name} stem saved successfully ({os.path.getsize(stem_path)} bytes)")
                else:
                    if self.logger:
                        self.logger.warning(f"✗ {name} stem file is empty or missing")
                    print(f"✗ {name} stem file is empty or missing")
            
            if self.logger:
                self.logger.info(f"Successfully created {len(stem_paths)} stem files")
            print(f"Successfully created {len(stem_paths)} stem files")
            return stem_paths
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in stem separation: {e}")
            print(f"Error in stem separation: {e}")
            import traceback
            traceback.print_exc()
            raise

class YouTubeDownloader:
    """Handles YouTube video downloading and audio extraction"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.temp_dir = None
        self.downloaded_files = []
    
    def is_youtube_url(self, url):
        """Check if a string is a YouTube URL"""
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/playlist\?list=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/channel/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/c/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/user/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+'
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def download_youtube_audio(self, url, output_dir):
        """Download YouTube video and extract audio"""
        try:
            if self.logger:
                self.logger.info(f"Downloading YouTube content: {url}")
            
            # Create temporary directory for downloads
            if not self.temp_dir:
                self.temp_dir = tempfile.mkdtemp(prefix="magicsample_youtube_")
                if self.logger:
                    self.logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'ignoreerrors': False,
                'nocheckcertificate': True,
                'prefer_ffmpeg': True,
                'geo_bypass': True,
            }
            
            downloaded_files = []
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                try:
                    info = ydl.extract_info(url, download=False)
                    if self.logger:
                        self.logger.info(f"Extracted info for: {info.get('title', 'Unknown')}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to extract info for {url}: {e}")
                    return []
                
                # Handle playlists
                if 'entries' in info:
                    if self.logger:
                        self.logger.info(f"Processing playlist with {len(info['entries'])} videos")
                    
                    for i, entry in enumerate(info['entries']):
                        if entry is None:
                            continue
                        
                        try:
                            if self.logger:
                                self.logger.info(f"Downloading playlist item {i+1}/{len(info['entries'])}: {entry.get('title', 'Unknown')}")
                            
                            # Download the video
                            ydl.download([entry['webpage_url']])
                            
                            # Find the downloaded file
                            for file in os.listdir(self.temp_dir):
                                if file.endswith('.wav') and not file in downloaded_files:
                                    file_path = os.path.join(self.temp_dir, file)
                                    downloaded_files.append(file_path)
                                    if self.logger:
                                        self.logger.info(f"Downloaded: {file}")
                                    break
                                    
                        except Exception as e:
                            if self.logger:
                                self.logger.error(f"Failed to download playlist item {i+1}: {e}")
                            continue
                else:
                    # Single video
                    try:
                        if self.logger:
                            self.logger.info(f"Downloading single video: {info.get('title', 'Unknown')}")
                        
                        ydl.download([url])
                        
                        # Find the downloaded file
                        for file in os.listdir(self.temp_dir):
                            if file.endswith('.wav') and not file in downloaded_files:
                                file_path = os.path.join(self.temp_dir, file)
                                downloaded_files.append(file_path)
                                if self.logger:
                                    self.logger.info(f"Downloaded: {file}")
                                break
                                
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to download video: {e}")
                        return []
            
            # Store downloaded files for cleanup
            self.downloaded_files.extend(downloaded_files)
            
            if self.logger:
                self.logger.info(f"Successfully downloaded {len(downloaded_files)} audio files")
            
            return downloaded_files
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in YouTube download: {e}")
            return []
    
    def cleanup(self):
        """Clean up downloaded files and temporary directory"""
        try:
            # Remove downloaded files
            for file_path in self.downloaded_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        if self.logger:
                            self.logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to remove file {file_path}: {e}")
            
            # Remove temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                    if self.logger:
                        self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to remove temp directory {self.temp_dir}: {e}")
            
            # Reset
            self.downloaded_files = []
            self.temp_dir = None
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during cleanup: {e}")

class MainWindow(QWidget):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.demucs_processor = DemucsProcessor()
        self.drum_classifier = DrumClassifier()
        self.dominant_frequency_detector = DominantFrequencyDetector()
        self.similarity_checker = SampleSimilarityChecker()
        self.youtube_downloader = YouTubeDownloader()
        self.setup_ui()  # Setup UI first to create log_text widget
        self.setup_logging()  # Then setup logging
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("MagicSample")
        self.setGeometry(100, 100, 1000, 700)  # Larger size for tabs
        
        # Set application icon - use absolute path to ensure it loads
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, "MagicSample_icon.ico")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            if not icon.isNull():
                self.setWindowIcon(icon)
                print(f"Application icon loaded successfully from: {icon_path}")
            else:
                print(f"Failed to load icon from: {icon_path} - icon is null")
        else:
            print(f"Application icon not found at: {icon_path}")
            # Try alternative paths
            alt_paths = [
                os.path.join(current_dir, "icon.ico"),
                os.path.join(current_dir, "MagicSample.ico")
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    icon = QIcon(alt_path)
                    if not icon.isNull():
                        self.setWindowIcon(icon)
                        print(f"Application icon loaded from alternative path: {alt_path}")
                        break
                    else:
                        print(f"Failed to load icon from: {alt_path} - icon is null")
        
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
        input_label = QLabel("Input Audio Files & YouTube URLs:")
        input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(input_label, 0, 0)
        
        # Create a list widget to show selected files and URLs
        self.input_files_list = QListWidget()
        self.input_files_list.setMaximumHeight(100)
        self.input_files_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        file_layout.addWidget(self.input_files_list, 0, 1)
        
        # Button layout for file operations
        file_buttons_layout = QVBoxLayout()
        
        input_btn = QPushButton("Add Files")
        input_btn.clicked.connect(self.select_input_files)
        input_btn.setFixedWidth(80)
        file_buttons_layout.addWidget(input_btn)
        
        youtube_btn = QPushButton("Add YouTube")
        youtube_btn.clicked.connect(self.add_youtube_url)
        youtube_btn.setFixedWidth(80)
        file_buttons_layout.addWidget(youtube_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_selected_files)
        remove_btn.setFixedWidth(80)
        file_buttons_layout.addWidget(remove_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_input_files)
        clear_btn.setFixedWidth(80)
        file_buttons_layout.addWidget(clear_btn)
        
        file_layout.addLayout(file_buttons_layout, 0, 2)
        
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
        
        # Minimum amplitude threshold input
        amplitude_label = QLabel("Min Amplitude (dBFS):")
        amplitude_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(amplitude_label, 7, 0)
        
        self.amplitude_input = QLineEdit()
        self.amplitude_input.setPlaceholderText("-20")
        self.amplitude_input.setText("-20")  # Default -20 dBFS
        self.amplitude_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.amplitude_input.setFixedWidth(100)
        self.amplitude_input.setToolTip("Minimum amplitude threshold in dBFS. Samples below this level will be skipped.")
        options_layout.addWidget(self.amplitude_input, 7, 1)
        
        # Output format and drumkit name in a row
        format_label = QLabel("Output Format:")
        format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(format_label, 8, 0)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['WAV', 'FLAC', 'OGG'])
        options_layout.addWidget(self.format_combo, 8, 1)
        
        drumkit_label = QLabel("Drumkit Name:")
        drumkit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(drumkit_label, 9, 0)
        
        self.drumkit_name_edit = QLineEdit()
        self.drumkit_name_edit.setPlaceholderText("MyDrumkit")
        self.drumkit_name_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(self.drumkit_name_edit, 9, 1)
        
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
        
        self.skip_button = QPushButton("Skip Sample")
        self.skip_button.clicked.connect(self.skip_current_sample)
        self.skip_button.setEnabled(False)
        self.skip_button.setFixedWidth(120)
        button_layout.addWidget(self.skip_button)
        
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

<h3>🎮 Control Buttons</h3>
<p><b>Start Processing:</b> Begins the audio processing workflow. All samples processed before stopping will be saved.</p>

<p><b>Skip Sample:</b> Skips the current sample being processed and moves to the next one. Useful for avoiding redundant or unwanted samples without stopping the entire process.</p>

<p><b>Stop:</b> Safely stops processing and performs cleanup operations. All samples processed up to that point will be saved in the correct folder structure with metadata.</p>

<h3>📁 File Selection</h3>
<p><b>Input Audio Files & YouTube URLs:</b> You can mix local audio files and YouTube URLs in the same processing session.</p>

<h4>📂 Local Files</h4>
<p><b>Add Files:</b> Select your source audio files (WAV, MP3, FLAC, OGG, M4A). These files will be processed into individual samples.</p>

<h4>🎥 YouTube Support</h4>
<p><b>Add YouTube:</b> Add YouTube URLs to download and process videos, playlists, or channels.</p>

<p><b>Supported YouTube URLs:</b></p>
<ul>
<li><b>Single Videos:</b> https://www.youtube.com/watch?v=VIDEO_ID</li>
<li><b>Playlists:</b> https://www.youtube.com/playlist?list=PLAYLIST_ID</li>
<li><b>Channels:</b> https://www.youtube.com/channel/CHANNEL_ID</li>
<li><b>User Channels:</b> https://www.youtube.com/user/USERNAME</li>
<li><b>Custom URLs:</b> https://www.youtube.com/c/CUSTOM_NAME</li>
<li><b>Shorts:</b> https://www.youtube.com/shorts/VIDEO_ID</li>
<li><b>Short URLs:</b> https://youtu.be/VIDEO_ID</li>
</ul>

<p><b>YouTube Processing Features:</b></p>
<ul>
<li><b>Automatic Download:</b> Downloads videos and extracts audio automatically</li>
<li><b>Playlist Support:</b> Processes all videos in a playlist</li>
<li><b>Channel Support:</b> Downloads videos from channels</li>
<li><b>Audio Extraction:</b> Converts videos to high-quality WAV audio</li>
<li><b>Automatic Cleanup:</b> Removes downloaded files after processing</li>
<li><b>Error Handling:</b> Continues processing if individual videos fail</li>
</ul>

<p><b>Output Directory:</b> Choose where your drumkit folder will be created. The program will create a new folder with your drumkit name inside this directory.</p>

<h3>⚙️ Processing Options</h3>

<h4>🔧 Core Features</h4>
<p><b>Split to Stems:</b> Uses Demucs AI to separate your audio into 4 stems: Drums, Bass, Vocals, and Other. When enabled, samples are organized into separate folders for each stem type. When disabled, processes the whole file as one category.</p>

<p><b>Detect BPM:</b> Analyzes the audio to find the tempo (beats per minute). The detected BPM is included in sample filenames (e.g., "Kick_001_120BPM_C2.WAV").</p>

<p><b>Detect Dominant Frequency:</b> Uses advanced multi-algorithm dominant frequency detection to find the most prominent frequency in each sample. Results are included in filenames using Scientific Pitch Notation (e.g., "C5", "F#3"). This detects the frequency with the greatest power/amplitude rather than just the fundamental frequency.</p>

<p><b>Classify Drums:</b> Automatically categorizes drum samples into subfolders: Kick, HiHat, and Perc (percussion). Only applies to the Drums stem.</p>

<h4>🎛️ Sensitivity Controls</h4>
<p><b>Sample Detection Sensitivity:</b> Controls how the program detects individual samples within the audio.</p>
<ul>
<li><b>Lower values (5-15):</b> More sensitive - detects more samples, including quieter and shorter sounds</li>
<li><b>Higher values (20-30):</b> Less sensitive - detects fewer samples, only the most prominent sounds</li>
<li><b>Recommended:</b> Start with 15, adjust based on your audio content</li>
</ul>

<p><b>Sample Similarity Threshold:</b> Controls how similar samples need to be before one is rejected as a duplicate. Uses fast spectral centroid + RMS energy comparison for quick processing.</p>
<ul>
<li><b>0%:</b> Very strict - only identical samples are considered duplicates</li>
<li><b>50%:</b> Balanced - moderately similar samples are rejected</li>
<li><b>80%:</b> Default - similar samples are rejected (recommended)</li>
<li><b>100%:</b> Very permissive - only very different samples are kept</li>
</ul>
<p><b>Note:</b> Similarity checking uses fast RMS energy, spectral centroid, and zero-crossing rate features for 5-10x faster processing compared to traditional MFCC methods.</p>

<h4>⏱️ Performance Settings</h4>
<p><b>Sample Timeout (ms):</b> Maximum time allowed for processing each individual sample. If a sample takes longer than this time to process (pitch detection, similarity check, etc.), it will be skipped but any information gathered before the timeout will still be included in the filename.</p>
<ul>
<li><b>1000ms:</b> Fast processing, may skip complex samples</li>
<li><b>2000ms:</b> Default - good balance of speed and accuracy</li>
<li><b>5000ms:</b> Slower but more thorough processing</li>
</ul>

<p><b>Min Amplitude (dBFS):</b> Minimum amplitude threshold in decibels relative to full scale. Samples with amplitude below this threshold will be skipped to avoid saving unusably quiet samples.</p>
<ul>
<li><b>-30 dBFS:</b> Very permissive - keeps most samples including quiet ones</li>
<li><b>-24 dBFS:</b> Moderate - filters out very quiet samples</li>
<li><b>-20 dBFS:</b> Default - good balance, filters out unusably quiet samples</li>
<li><b>-18 dBFS:</b> Strict - only keeps relatively loud samples</li>
<li><b>-12 dBFS:</b> Very strict - only keeps loud samples</li>
</ul>
<p><b>Note:</b> dBFS (decibels relative to full scale) measures how loud a sample is. Lower values mean quieter samples. For reference: -20 dBFS is about 10% of maximum volume.</p>

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

<h3>🛡️ Error Handling & Safety Features</h3>

<h4>Comprehensive Exception Handling</h4>
<p>The program includes extensive error handling to prevent crashes:</p>
<ul>
<li><b>Graceful Degradation:</b> If one feature fails (e.g., pitch detection), the program continues with other features</li>
<li><b>Detailed Logging:</b> All errors are logged with full context for troubleshooting</li>
<li><b>User-Friendly Messages:</b> Clear error messages explain what went wrong</li>
<li><b>Automatic Recovery:</b> The program attempts to recover from errors when possible</li>
</ul>

<h4>Safe Stop & Cleanup</h4>
<p>When you press the Stop button:</p>
<ul>
<li><b>Immediate Logging:</b> The stop request is logged with timestamp</li>
<li><b>Sample Preservation:</b> All samples processed up to that point are saved</li>
<li><b>Metadata Creation:</b> A metadata.json file is created with processing information</li>
<li><b>Clean Shutdown:</b> Temporary files are cleaned up properly</li>
<li><b>UI Reset:</b> The interface returns to a ready state</li>
</ul>

<h4>Skip Functionality</h4>
<p>The Skip Sample button allows you to:</p>
<ul>
<li><b>Skip Redundant Samples:</b> Avoid processing similar or unwanted samples</li>
<li><b>Maintain Progress:</b> Continue processing without losing completed work</li>
<li><b>Selective Processing:</b> Choose which samples to keep during processing</li>
<li><b>Time Saving:</b> Skip samples that would be rejected by similarity checks</li>
</ul>

<h3>🔍 Troubleshooting</h3>

<h4>Common Issues</h4>
<p><b>No samples created:</b> Try lowering the Sample Detection Sensitivity or increasing the timeout.</p>

<p><b>Too many similar samples:</b> Increase the Sample Similarity Threshold or use the Skip button.</p>

<p><b>Processing is slow:</b> Reduce the Sample Timeout or disable pitch detection for faster processing.</p>

<p><b>Stem separation fails:</b> Check the Log tab for detailed error messages. The program will fall back to processing the whole file.</p>

<p><b>Program crashes:</b> Check the Log tab for error details. The program now includes comprehensive exception handling.</p>

<h4>Performance Tips</h4>
<ul>
<li>Use WAV or FLAC input files for best quality</li>
<li>Disable pitch detection if you don't need pitch information</li>
<li>Adjust sensitivity based on your audio content</li>
<li>Use the Log tab to monitor processing progress</li>
<li>Use the Skip button to avoid processing unwanted samples</li>
<li>Use Stop instead of closing the program to ensure proper cleanup</li>
</ul>

<h3>📊 Advanced Features</h3>

<h4>Dominant Frequency Detection Algorithms</h4>
<p>The program uses 4 different algorithms and combines their results:</p>
<ul>
<li><b>FFT-based Dominant Frequency:</b> Fast Fourier Transform to find the frequency with maximum power</li>
<li><b>Welch's Method:</b> Spectral density estimation for robust frequency analysis</li>
<li><b>Autocorrelation Peak Detection:</b> Time-domain analysis to find dominant periodic components</li>
<li><b>Combined Approach:</b> Multi-method validation for highest accuracy</li>
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

<h4>Logging & Debugging</h4>
<p>The Log tab provides comprehensive information:</p>
<ul>
<li><b>Real-time Processing:</b> See exactly what the program is doing</li>
<li><b>Error Details:</b> Full stack traces and error context</li>
<li><b>Performance Metrics:</b> Processing times and sample counts</li>
<li><b>User Actions:</b> Logs when you use Stop or Skip buttons</li>
<li><b>Save Functionality:</b> Save logs to files for later analysis</li>
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
        
        # Get the application logger and clear any existing handlers
        self.logger = logging.getLogger('MagicSample')
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add our custom handler
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)
        
        # Prevent propagation to root logger to avoid console output
        self.logger.propagate = False
        
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
    
    def select_input_files(self):
        """Select multiple input audio files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "", 
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        if file_paths:
            for file_path in file_paths:
                # Add file to list if not already present
                items = [self.input_files_list.item(i).text() for i in range(self.input_files_list.count())]
                if file_path not in items:
                    self.input_files_list.addItem(file_path)
    
    def remove_selected_files(self):
        """Remove selected files from the list"""
        selected_items = self.input_files_list.selectedItems()
        for item in selected_items:
            self.input_files_list.takeItem(self.input_files_list.row(item))
    
    def clear_input_files(self):
        """Clear all files from the list"""
        self.input_files_list.clear()
    
    def add_youtube_url(self):
        """Add YouTube URL to the input list"""
        url, ok = QInputDialog.getText(
            self, "Add YouTube URL", 
            "Enter YouTube URL (video, playlist, or channel):",
            QLineEdit.EchoMode.Normal
        )
        
        if ok and url.strip():
            url = url.strip()
            if self.youtube_downloader.is_youtube_url(url):
                # Add URL to list if not already present
                items = [self.input_files_list.item(i).text() for i in range(self.input_files_list.count())]
                if url not in items:
                    self.input_files_list.addItem(url)
                    if self.logger:
                        self.logger.info(f"Added YouTube URL: {url}")
                else:
                    QMessageBox.information(self, "Info", "This URL is already in the list.")
            else:
                QMessageBox.warning(self, "Invalid URL", "Please enter a valid YouTube URL.")
    
    def get_input_files(self):
        """Get list of all input files and URLs"""
        files = []
        for i in range(self.input_files_list.count()):
            files.append(self.input_files_list.item(i).text())
        return files
    
    def select_output_dir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path_edit.setText(dir_path)
    
    def start_processing(self):
        """Start the audio processing"""
        try:
            input_files = self.get_input_files()
            if not input_files or not self.output_path_edit.text():
                QMessageBox.warning(self, "Error", "Please select input files and output directory")
                return
            
            self.start_button.setEnabled(False)
            self.skip_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.progress_bar.setValue(0)
            
            self.logger.info("Starting audio processing...")
            
            # Validate timeout input
            try:
                timeout_value = int(self.timeout_input.text() or "2000")
                if timeout_value <= 0:
                    QMessageBox.warning(self, "Error", "Timeout must be a positive number")
                    return
            except ValueError:
                QMessageBox.warning(self, "Error", "Timeout must be a valid number")
                return
            
            # Get amplitude threshold value
            try:
                amplitude_value = float(self.amplitude_input.text())
            except ValueError:
                QMessageBox.warning(self, "Error", "Minimum amplitude must be a valid number")
                return
            
            # Start processing in a separate thread
            self.worker = ProcessingWorker(
                input_files,
                self.output_path_edit.text(),
                self.drumkit_name_edit.text() or "MyDrumkit",
                self.format_combo.currentText(),
                self.stems_checkbox.isChecked(),
                self.bpm_checkbox.isChecked(),
                self.pitch_checkbox.isChecked(),
                self.drum_classify_checkbox.isChecked(),
                self.sensitivity_slider.value(),
                self.similarity_slider.value() / 100.0,  # Convert percentage to 0.0-1.0 range
                timeout_value,
                amplitude_value,
                self.logger  # Pass the logger
            )
            
            # Pass the YouTube downloader to the worker
            self.worker.youtube_downloader = self.youtube_downloader
            
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.status_updated.connect(self.update_status)
            self.worker.finished.connect(self.processing_finished)
            self.worker.start()
            
        except Exception as e:
            self.logger.error(f"Error starting processing: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start processing: {str(e)}")
            self.processing_finished()
    
    def skip_current_sample(self):
        """Skip the current sample being processed"""
        try:
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.skip_current_sample()
                self.logger.info("User requested to skip current sample")
                self.status_label.setText("Skipping current sample...")
            else:
                self.logger.warning("Skip requested but no worker is running")
        except Exception as e:
            self.logger.error(f"Error skipping sample: {e}")
            QMessageBox.warning(self, "Error", f"Failed to skip sample: {str(e)}")
    
    def stop_processing(self):
        """Stop the audio processing"""
        try:
            if hasattr(self, 'worker'):
                self.logger.info("User requested to stop processing - initiating cleanup...")
                self.worker.stop()
                self.status_label.setText("Stopping processing and saving samples...")
            else:
                self.logger.warning("Stop requested but no worker exists")
        except Exception as e:
            self.logger.error(f"Error stopping processing: {e}")
            QMessageBox.warning(self, "Error", f"Failed to stop processing: {str(e)}")
        finally:
            self.processing_finished()
    
    def update_progress(self, value):
        """Update progress bar"""
        try:
            self.progress_bar.setValue(value)
        except Exception as e:
            self.logger.error(f"Error updating progress: {e}")
    
    def update_status(self, status):
        """Update status label"""
        try:
            self.status_label.setText(status)
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    def processing_finished(self):
        """Called when processing is finished"""
        try:
            self.start_button.setEnabled(True)
            self.skip_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.progress_bar.setValue(100)
            self.status_label.setText("Processing completed!")
            self.logger.info("Processing finished - UI reset to ready state")
        except Exception as e:
            self.logger.error(f"Error in processing_finished: {e}")

class ProcessingWorker(QThread):
    """Worker thread for audio processing"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, input_files, output_path, drumkit_name, output_format, 
                 split_stems, detect_bpm, detect_pitch, classify_drums, sensitivity, similarity_threshold, timeout_ms, min_amplitude_db, logger=None):
        super().__init__()
        self.input_files = input_files if isinstance(input_files, list) else [input_files]
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
        self.min_amplitude_db = min_amplitude_db
        self.stop_flag = False
        self.skip_flag = False
        self.logger = logger or logging.getLogger()  # Use provided logger or fallback to root logger
        
        # Initialize processors
        try:
            self.demucs_processor = DemucsProcessor(logger)
            self.drum_classifier = DrumClassifier()
            self.dominant_frequency_detector = DominantFrequencyDetector()
            self.similarity_checker = SampleSimilarityChecker(similarity_threshold)
            self.youtube_downloader = YouTubeDownloader(logger)
            self.advanced_detector = AdvancedSampleDetector(min_amplitude_db, logger)
            # Note: We'll use the main window's logger for this class
            pass  # Logging will be handled by the main window
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error initializing ProcessingWorker: {e}")
            raise
    
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
            if self.logger:
                self.logger.info(f"Starting audio processing for {len(self.input_files)} files...")
            self.status_updated.emit(f"Processing {len(self.input_files)} files...")
            self.progress_updated.emit(5)
            
            # Create drumkit directory
            try:
                drumkit_path = os.path.join(self.output_path, self.drumkit_name.capitalize())
                os.makedirs(drumkit_path, exist_ok=True)
                self.drumkit_path = drumkit_path  # Store for cleanup
                if self.logger:
                    self.logger.info(f"Created drumkit directory: {drumkit_path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to create drumkit directory: {e}")
                self.status_updated.emit(f"Error: Failed to create output directory - {str(e)}")
                return
            
            # Process each input file or YouTube URL
            total_files = len(self.input_files)
            for file_index, input_path in enumerate(self.input_files):
                # Reset similarity checker for each file so only compare within the same file
                self.similarity_checker = SampleSimilarityChecker(self.similarity_checker.similarity_threshold)
                if self.stop_flag:
                    break
                
                # Check if this is a YouTube URL
                if self.youtube_downloader.is_youtube_url(input_path):
                    if self.logger:
                        self.logger.info(f"Processing YouTube URL {file_index + 1}/{total_files}: {input_path}")
                    self.status_updated.emit(f"Processing YouTube URL {file_index + 1}/{total_files}: {input_path}")
                    
                    # Download YouTube audio
                    try:
                        downloaded_files = self.youtube_downloader.download_youtube_audio(input_path, drumkit_path)
                        if not downloaded_files:
                            if self.logger:
                                self.logger.error(f"Failed to download YouTube content: {input_path}")
                            self.status_updated.emit(f"Error: Failed to download YouTube content: {input_path}")
                            continue
                        
                        # Process each downloaded file
                        for downloaded_file in downloaded_files:
                            if self.stop_flag:
                                break
                            
                            # Use the downloaded file as input_path for processing
                            file_identifier = os.path.splitext(os.path.basename(downloaded_file))[0]
                            
                            # Load audio file with error handling
                            try:
                                audio_data, sample_rate = librosa.load(downloaded_file, sr=None, mono=False)
                                if self.logger:
                                    self.logger.info(f"Loaded YouTube audio: {audio_data.shape}, sample rate: {sample_rate}")
                                print(f"Loaded YouTube audio: {audio_data.shape}, sample rate: {sample_rate}")
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"Failed to load YouTube audio file {downloaded_file}: {e}")
                                self.status_updated.emit(f"Error: Failed to load YouTube audio file {os.path.basename(downloaded_file)} - {str(e)}")
                                continue
                            
                            # Process this downloaded file
                            self.process_single_file(audio_data, sample_rate, drumkit_path, file_identifier, file_index, total_files)
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error processing YouTube URL {input_path}: {e}")
                        self.status_updated.emit(f"Error processing YouTube URL: {str(e)}")
                        continue
                        
                else:
                    # Regular file processing
                    if self.logger:
                        self.logger.info(f"Processing file {file_index + 1}/{total_files}: {os.path.basename(input_path)}")
                    self.status_updated.emit(f"Processing file {file_index + 1}/{total_files}: {os.path.basename(input_path)}")
                    
                    # Load audio file with error handling
                    try:
                        audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)
                        if self.logger:
                            self.logger.info(f"Loaded audio: {audio_data.shape}, sample rate: {sample_rate}")
                        print(f"Loaded audio: {audio_data.shape}, sample rate: {sample_rate}")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to load audio file {input_path}: {e}")
                        self.status_updated.emit(f"Error: Failed to load audio file {os.path.basename(input_path)} - {str(e)}")
                        continue  # Skip this file and continue with next
                    
                    # Process this file
                    file_identifier = os.path.splitext(os.path.basename(input_path))[0]
                    self.process_single_file(audio_data, sample_rate, drumkit_path, file_identifier, file_index, total_files)
            
            # Create metadata file after processing all files
            self.create_drumkit_metadata(drumkit_path, None)  # Use None for bpm since it varies per file
            self.progress_updated.emit(100)
            self.status_updated.emit("Drumkit creation completed!")
            if self.logger:
                self.logger.info("Drumkit creation completed!")
                
            # Clean up YouTube downloads
            if hasattr(self, 'youtube_downloader'):
                self.youtube_downloader.cleanup()
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Processing error: {e}")
            self.status_updated.emit(f"Error: {str(e)}")
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def process_single_file(self, audio_data, sample_rate, drumkit_path, file_identifier, file_index, total_files):
        """Process a single audio file (either local or downloaded from YouTube)"""
        try:
            # Detect BPM for this file
            bpm = None
            if self.detect_bpm:
                try:
                    self.status_updated.emit(f"Detecting BPM for {file_identifier}...")
                    progress = 10 + (file_index * 20 // total_files)
                    self.progress_updated.emit(progress)
                    if self.logger:
                        self.logger.info(f"Detecting BPM for {file_identifier}...")
                    bpm = self.detect_bpm_from_audio(audio_data, sample_rate)
                    if self.logger:
                        self.logger.info(f"Detected BPM for {file_identifier}: {bpm}")
                    print(f"Detected BPM: {bpm}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"BPM detection failed for {file_identifier}: {e}")
                    self.status_updated.emit(f"Warning: BPM detection failed for {file_identifier}, continuing without BPM...")
                    bpm = None
            
            # Process stems for this file
            if self.split_stems:
                self.status_updated.emit(f"Separating stems for {file_identifier}...")
                progress = 30 + (file_index * 40 // total_files)
                self.progress_updated.emit(progress)
                if self.logger:
                    self.logger.info(f"Starting stem separation for {file_identifier}...")
                
                # Create temporary directory for this file's stems
                temp_dir = os.path.join(drumkit_path, f"temp_stems_{file_index}")
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    # For YouTube downloads, we need to save the audio data temporarily
                    temp_audio_path = None  # Initialize variable
                    if hasattr(self, 'youtube_downloader') and self.youtube_downloader.temp_dir:
                        temp_audio_path = os.path.join(temp_dir, f"{file_identifier}.wav")
                        self.save_audio_sample(audio_data, sample_rate, temp_audio_path)
                        input_path_for_stems = temp_audio_path
                    else:
                        input_path_for_stems = None  # This will be handled by the calling code
                    
                    # Use the appropriate path for stem separation
                    stem_input_path = input_path_for_stems if input_path_for_stems else temp_audio_path
                    if stem_input_path is None:
                        # If no path is available, we need to save the audio data temporarily
                        temp_audio_path = os.path.join(temp_dir, f"{file_identifier}.wav")
                        self.save_audio_sample(audio_data, sample_rate, temp_audio_path)
                        stem_input_path = temp_audio_path
                    
                    stem_paths = self.demucs_processor.separate_stems(stem_input_path, temp_dir)
                    if self.logger:
                        self.logger.info(f"Created stems for {file_identifier}: {list(stem_paths.keys())}")
                    print(f"Created stems: {list(stem_paths.keys())}")
                    
                    stem_count = len(stem_paths)
                    for i, (stem_name, stem_path) in enumerate(stem_paths.items()):
                        if self.stop_flag:
                            break
                        progress = 30 + (file_index * 40 // total_files) + (i * 30 // stem_count)
                        self.status_updated.emit(f"Processing {stem_name} stem from {file_identifier} ({i+1}/{stem_count})...")
                        self.progress_updated.emit(progress)
                        if self.logger:
                            self.logger.info(f"Processing {stem_name} stem from {file_identifier} ({i+1}/{stem_count})...")
                        
                        if not os.path.exists(stem_path) or os.path.getsize(stem_path) == 0:
                            if self.logger:
                                self.logger.warning(f"Stem file {stem_path} is empty or missing")
                            print(f"Warning: Stem file {stem_path} is empty or missing")
                            continue
                        
                        stem_audio, stem_sr = librosa.load(stem_path, sr=None, mono=False)
                        if self.logger:
                            self.logger.info(f"Loaded {stem_name} stem: {stem_audio.shape}")
                        print(f"Loaded {stem_name} stem: {stem_audio.shape}")
                        
                        # Use shared directories for all files
                        stem_dir = os.path.join(drumkit_path, stem_name.capitalize())
                        os.makedirs(stem_dir, exist_ok=True)
                        
                        if stem_name == "vocals":
                            # Add file identifier to vocals filename
                            vocals_filename = f"Vocals_{file_identifier}_{bpm}BPM.{self.output_format.upper()}" if bpm else f"Vocals_{file_identifier}.{self.output_format.upper()}"
                            vocals_path = os.path.join(stem_dir, vocals_filename)
                            self.save_audio_sample(stem_audio, stem_sr, vocals_path)
                            if self.logger:
                                self.logger.info(f"Saved vocals as whole file: {vocals_filename}")
                            print(f"Saved vocals as whole file: {vocals_filename}")
                        elif stem_name == "drums":
                            self.status_updated.emit(f"Splitting drums from {file_identifier} into one-shots...")
                            if self.logger:
                                self.logger.info(f"Processing drums from {file_identifier} with subfolder classification...")
                            self.process_drums_with_subfolders(stem_audio, stem_sr, stem_dir, bpm, file_identifier=file_identifier)
                        else:
                            self.status_updated.emit(f"Detecting one-shots in {stem_name} from {file_identifier}...")
                            if self.logger:
                                self.logger.info(f"Processing {stem_name} from {file_identifier} into individual samples...")
                            sample_count = self.process_stem_into_samples(stem_audio, stem_sr, stem_dir, stem_name.capitalize(), bpm, file_identifier=file_identifier)
                            if self.logger:
                                self.logger.info(f"Created {sample_count} samples for {stem_name} from {file_identifier}")
                            print(f"Created {sample_count} samples for {stem_name}")
                    
                    # Clean up temporary files for this file
                    import shutil
                    shutil.rmtree(temp_dir)
                    if self.logger:
                        self.logger.info(f"Cleaned up temporary stem files for {file_identifier}")
                        
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in stem separation for {file_identifier}: {e}")
                    print(f"Error in stem separation: {e}")
                    self.status_updated.emit(f"Stem separation failed for {file_identifier}: {str(e)}")
                    self.status_updated.emit(f"Falling back to processing whole file for {file_identifier}...")
                    if self.logger:
                        self.logger.info(f"Falling back to processing whole file for {file_identifier}...")
                    sample_count = self.process_stem_into_samples(audio_data, sample_rate, drumkit_path, "Samples", bpm, file_identifier=file_identifier)
                    if self.logger:
                        self.logger.info(f"Created {sample_count} samples from whole file for {file_identifier}")
                    print(f"Created {sample_count} samples from whole file")
            else:
                # Process whole file without stem separation
                self.status_updated.emit(f"Processing {file_identifier} into samples...")
                progress = 70 + (file_index * 20 // total_files)
                self.progress_updated.emit(progress)
                if self.logger:
                    self.logger.info(f"Processing whole file {file_identifier} into samples...")
                samples_dir = os.path.join(drumkit_path, "Samples")
                os.makedirs(samples_dir, exist_ok=True)
                sample_count = self.process_stem_into_samples(audio_data, sample_rate, samples_dir, "Sample", bpm, file_identifier=file_identifier)
                if self.logger:
                    self.logger.info(f"Created {sample_count} samples from whole file for {file_identifier}")
                print(f"Created {sample_count} samples from whole file")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing file {file_identifier}: {e}")
            self.status_updated.emit(f"Error processing file {file_identifier}: {str(e)}")
            raise
    
    def run(self):
        """Main processing function with improved progress updates."""
        try:
            if self.logger:
                self.logger.info(f"Starting audio processing for {len(self.input_files)} files...")
            self.status_updated.emit(f"Processing {len(self.input_files)} files...")
            self.progress_updated.emit(5)
            
            # Create drumkit directory
            try:
                drumkit_path = os.path.join(self.output_path, self.drumkit_name.capitalize())
                os.makedirs(drumkit_path, exist_ok=True)
                self.drumkit_path = drumkit_path  # Store for cleanup
                if self.logger:
                    self.logger.info(f"Created drumkit directory: {drumkit_path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to create drumkit directory: {e}")
                self.status_updated.emit(f"Error: Failed to create output directory - {str(e)}")
                return
            
            # Process each input file or YouTube URL
            total_files = len(self.input_files)
            for file_index, input_path in enumerate(self.input_files):
                # Reset similarity checker for each file so only compare within the same file
                self.similarity_checker = SampleSimilarityChecker(self.similarity_checker.similarity_threshold)
                if self.stop_flag:
                    break
                
                # Check if this is a YouTube URL
                if self.youtube_downloader.is_youtube_url(input_path):
                    if self.logger:
                        self.logger.info(f"Processing YouTube URL {file_index + 1}/{total_files}: {input_path}")
                    self.status_updated.emit(f"Processing YouTube URL {file_index + 1}/{total_files}: {input_path}")
                    
                    # Download YouTube audio
                    try:
                        downloaded_files = self.youtube_downloader.download_youtube_audio(input_path, drumkit_path)
                        if not downloaded_files:
                            if self.logger:
                                self.logger.error(f"Failed to download YouTube content: {input_path}")
                            self.status_updated.emit(f"Error: Failed to download YouTube content: {input_path}")
                            continue
                        
                        # Process each downloaded file
                        for downloaded_file in downloaded_files:
                            if self.stop_flag:
                                break
                            
                            # Use the downloaded file as input_path for processing
                            file_identifier = os.path.splitext(os.path.basename(downloaded_file))[0]
                            
                            # Load audio file with error handling
                            try:
                                audio_data, sample_rate = librosa.load(downloaded_file, sr=None, mono=False)
                                if self.logger:
                                    self.logger.info(f"Loaded YouTube audio: {audio_data.shape}, sample rate: {sample_rate}")
                                print(f"Loaded YouTube audio: {audio_data.shape}, sample rate: {sample_rate}")
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"Failed to load YouTube audio file {downloaded_file}: {e}")
                                self.status_updated.emit(f"Error: Failed to load YouTube audio file {os.path.basename(downloaded_file)} - {str(e)}")
                                continue
                            
                            # Process this downloaded file
                            self.process_single_file(audio_data, sample_rate, drumkit_path, file_identifier, file_index, total_files)
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error processing YouTube URL {input_path}: {e}")
                        self.status_updated.emit(f"Error processing YouTube URL: {str(e)}")
                        continue
                        
                else:
                    # Regular file processing
                    if self.logger:
                        self.logger.info(f"Processing file {file_index + 1}/{total_files}: {os.path.basename(input_path)}")
                    self.status_updated.emit(f"Processing file {file_index + 1}/{total_files}: {os.path.basename(input_path)}")
                    
                    # Load audio file with error handling
                    try:
                        audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)
                        if self.logger:
                            self.logger.info(f"Loaded audio: {audio_data.shape}, sample rate: {sample_rate}")
                        print(f"Loaded audio: {audio_data.shape}, sample rate: {sample_rate}")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to load audio file {input_path}: {e}")
                        self.status_updated.emit(f"Error: Failed to load audio file {os.path.basename(input_path)} - {str(e)}")
                        continue  # Skip this file and continue with next
                    
                    # Process this file
                    file_identifier = os.path.splitext(os.path.basename(input_path))[0]
                    self.process_single_file(audio_data, sample_rate, drumkit_path, file_identifier, file_index, total_files)
            
            # Create metadata file after processing all files
            self.create_drumkit_metadata(drumkit_path, None)  # Use None for bpm since it varies per file
            self.progress_updated.emit(100)
            self.status_updated.emit("Drumkit creation completed!")
            if self.logger:
                self.logger.info("Drumkit creation completed!")
                
            # Clean up YouTube downloads
            if hasattr(self, 'youtube_downloader'):
                self.youtube_downloader.cleanup()
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Processing error: {e}")
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
    
    def get_next_sample_number(self, output_dir, stem_name):
        """Get the next sequential sample number for a given folder"""
        try:
            # Count existing files in the directory
            if not os.path.exists(output_dir):
                return 1
            
            existing_files = [f for f in os.listdir(output_dir) 
                            if f.startswith(f"{stem_name.capitalize()}_") and f.endswith(f".{self.output_format.upper()}")]
            
            if not existing_files:
                return 1
            
            # Extract numbers from existing filenames
            numbers = []
            for filename in existing_files:
                # Extract number from filename like "Perc_001_2on_101BPM_C4.WAV"
                parts = filename.split('_')
                if len(parts) >= 2:
                    try:
                        # Get the number part (e.g., "001" from "Perc_001")
                        number_str = parts[1]
                        number = int(number_str)
                        numbers.append(number)
                    except (ValueError, IndexError):
                        continue
            
            # Return next number
            return max(numbers) + 1 if numbers else 1
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting next sample number: {e}")
            return 1
    
    def process_stem_into_samples(self, audio_data, sample_rate, output_dir, stem_name, bpm, file_identifier=""):
        """Process a stem into individual samples using hybrid transient detection and energy-based slicing."""
        try:
            if self.logger:
                self.logger.info(f"Processing {stem_name} into samples using hybrid detection...")
            
            # Define minimum duration for samples
            min_duration = 0.05  # 50ms minimum duration
            
            # Determine if this is a drum stem for detection strategy
            is_drum = stem_name.lower() in ["drums", "kick", "hihat", "perc", "snare", "tom", "clap"]
            
            # Use hybrid detection for sample boundaries
            if hasattr(self, 'status_updated'):
                detection_method = "transient-based" if is_drum else "energy-based"
                self.status_updated.emit(f"Using {detection_method} detection for {stem_name} stem...")
            
            sample_boundaries = self.advanced_detector.hybrid_detection(audio_data, sample_rate, is_drum)
            
            if self.logger:
                self.logger.info(f"Found {len(sample_boundaries)} potential samples in {stem_name} using hybrid detection")
            
            if len(sample_boundaries) == 0:
                sample_boundaries = [(0, len(audio_data))]

            total_samples = len(sample_boundaries)
            sample_count = 0  # Initialize sample_count
            
            for i, (start, end) in enumerate(sample_boundaries):
                if self.stop_flag:
                    if self.logger:
                        self.logger.info("Stop flag detected - breaking sample processing loop")
                    break
                
                # Check for skip flag
                if self.skip_flag:
                    if self.logger:
                        self.logger.info(f"Skipping sample {i+1}/{total_samples}")
                    self.skip_flag = False  # Reset skip flag
                    continue
                # Extract sample with error handling
                try:
                    if len(audio_data.shape) > 1:
                        sample_audio = audio_data[:, start:end]
                    else:
                        sample_audio = audio_data[start:end]
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error extracting sample {i+1}: {e}")
                    continue
                
                # Check amplitude threshold (unless it's the only sample)
                if total_samples > 1:
                    try:
                        meets_threshold, amplitude_dbfs = self.advanced_detector.check_amplitude_threshold(sample_audio)
                        if not meets_threshold:
                            if self.logger:
                                self.logger.info(f"Skipping {stem_name} sample {i+1}: amplitude {amplitude_dbfs:.1f} dBFS below threshold {self.min_amplitude_db} dBFS")
                            if hasattr(self, 'status_updated'):
                                self.status_updated.emit(f"Skipping {stem_name} sample {i+1}: too quiet ({amplitude_dbfs:.1f} dBFS)")
                            continue
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error checking amplitude threshold for sample {i+1}: {e}")
                        # Continue processing if amplitude check fails
                
                # Skip very short samples (unless it's the only sample)
                try:
                    sample_duration = sample_audio.shape[-1] / sample_rate
                    if sample_duration < min_duration and total_samples > 1:
                        if self.logger:
                            self.logger.info(f"Skipping {stem_name} sample {i+1}: too short ({sample_duration*1000:.1f}ms)")
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Skipping {stem_name} sample {i+1}: too short ({sample_duration*1000:.1f}ms)")
                        continue
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error calculating sample duration for sample {i+1}: {e}")
                    continue
                
                # Skip very quiet samples (unless it's the only sample)
                try:
                    sample_rms = np.sqrt(np.mean(sample_audio**2))
                    if sample_rms < 0.001 and total_samples > 1:
                        if self.logger:
                            self.logger.info(f"Skipping {stem_name} sample {i+1}: too quiet (RMS: {sample_rms:.6f})")
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Skipping {stem_name} sample {i+1}: too quiet (RMS: {sample_rms:.6f})")
                        continue
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error calculating RMS for sample {i+1}: {e}")
                    continue
                # Verbose progress for each sample
                if hasattr(self, 'status_updated'):
                    self.status_updated.emit(f"Saving {stem_name} sample {i+1}/{total_samples} (duration: {sample_duration:.2f}s)...")
                if hasattr(self, 'progress_updated'):
                    self.progress_updated.emit(40 + int(50 * (i+1) / max(1, total_samples)))
                # Generate filename with sequential numbering
                next_number = self.get_next_sample_number(output_dir, stem_name)
                filename_parts = [f"{stem_name.capitalize()}_{next_number:03d}"]
                if file_identifier:
                    filename_parts.append(file_identifier)
                if bpm:
                    filename_parts.append(f"{bpm}BPM")
                
                # Dominant frequency detection with timeout and error handling
                dominant_freq = None
                if self.detect_pitch:
                    try:
                        freq_result, freq_status = self.run_with_timeout(
                            self.dominant_frequency_detector.detect_dominant_frequency, sample_audio, sample_rate
                        )
                        if freq_status == "success" and freq_result and freq_result != "N/A":
                            dominant_freq = freq_result
                            filename_parts.append(dominant_freq)
                            if self.logger:
                                self.logger.info(f"Dominant frequency detected for {stem_name} sample {i+1}: {dominant_freq}")
                        elif freq_status == "timeout":
                            if self.logger:
                                self.logger.warning(f"Dominant frequency detection timeout for {stem_name} sample {i+1}")
                            if hasattr(self, 'status_updated'):
                                self.status_updated.emit(f"Dominant frequency detection timeout for {stem_name} sample {i+1}, continuing...")
                        else:
                            if self.logger:
                                self.logger.warning(f"Dominant frequency detection failed for {stem_name} sample {i+1}: {freq_status}")
                            if hasattr(self, 'status_updated'):
                                self.status_updated.emit(f"Dominant frequency detection failed for {stem_name} sample {i+1}: {freq_status}")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error during dominant frequency detection for {stem_name} sample {i+1}: {e}")
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Dominant frequency detection error for {stem_name} sample {i+1}: {str(e)}")
                
                filename = "_".join(filename_parts) + f".{self.output_format.upper()}"
                filepath = os.path.join(output_dir, filename)
                
                # Similarity check with timeout and error handling
                try:
                    similarity_result, similarity_status = self.run_with_timeout(
                        self.similarity_checker.is_similar_to_existing, sample_audio, sample_rate, stem_name.capitalize()
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during similarity check for {stem_name} sample {i+1}: {e}")
                    similarity_status = "error"
                    similarity_result = False
                
                if similarity_status == "timeout":
                    if self.logger:
                        self.logger.warning(f"Similarity check timeout for {stem_name} sample {i+1}, saving sample...")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check timeout for {stem_name} sample {i+1}, saving sample...")
                    # Save sample even if similarity check times out
                    try:
                        self.save_audio_sample(sample_audio, sample_rate, filepath)
                        sample_count += 1
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to save sample after similarity timeout: {e}")
                elif similarity_status == "success" and similarity_result:
                    if self.logger:
                        self.logger.info(f"Skipping {stem_name} sample {i+1}: too similar to existing samples.")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Skipping {stem_name} sample {i+1}: too similar to existing samples.")
                    continue
                elif similarity_status == "success" and not similarity_result:
                    # Not similar, save the sample
                    if self.logger:
                        self.logger.info(f"Saving {stem_name} sample {i+1}: {filename}")
                    try:
                        self.save_audio_sample(sample_audio, sample_rate, filepath)
                        sample_count += 1
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to save sample {i+1}: {e}")
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Failed to save {stem_name} sample {i+1}: {str(e)}")
                else:
                    # Similarity check failed, save sample anyway
                    if self.logger:
                        self.logger.warning(f"Similarity check failed for {stem_name} sample {i+1}: {similarity_status}, saving sample...")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check failed for {stem_name} sample {i+1}: {similarity_status}, saving sample...")
                    try:
                        self.save_audio_sample(sample_audio, sample_rate, filepath)
                        sample_count += 1
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to save sample after similarity failure: {e}")
            if self.logger:
                self.logger.info(f"Successfully created {sample_count} samples for {stem_name}")
            if hasattr(self, 'status_updated'):
                self.status_updated.emit(f"Successfully created {sample_count} samples for {stem_name}.")
            return sample_count
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing stem {stem_name}: {e}")
            print(f"Error processing stem {stem_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0  # Return 0 samples processed on error
    
    def process_drums_with_subfolders(self, audio_data, sample_rate, output_dir, bpm, file_identifier=""):
        """Process drums into frequency-based subfolders: Kick, Perc, HiHat using hybrid detection"""
        try:
            if self.logger:
                self.logger.info(f"Processing drums with hybrid detection...")
            
            # Use hybrid detection optimized for drums (transient-based)
            if hasattr(self, 'status_updated'):
                self.status_updated.emit("Using transient-based detection for drums...")
            
            sample_boundaries = self.advanced_detector.hybrid_detection(audio_data, sample_rate, is_drum=True)
            
            if self.logger:
                self.logger.info(f"Found {len(sample_boundaries)} potential drum samples using hybrid detection")
            
            if len(sample_boundaries) == 0:
                if self.logger:
                    self.logger.warning("No drum samples detected. Creating single sample.")
                # Create one sample from the entire audio
                sample_boundaries = [(0, len(audio_data))]
            
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
                
                # Check for skip flag
                if self.skip_flag:
                    if self.logger:
                        self.logger.info(f"Skipping drum sample {i+1}/{len(sample_boundaries)}")
                    self.skip_flag = False  # Reset skip flag
                    continue
                
                # Extract sample
                try:
                    if len(audio_data.shape) > 1:
                        sample_audio = audio_data[:, start:end]
                    else:
                        sample_audio = audio_data[start:end]
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error extracting drum sample {i+1}: {e}")
                    continue
                
                # Check amplitude threshold (unless it's the only sample)
                if len(sample_boundaries) > 1:
                    try:
                        meets_threshold, amplitude_dbfs = self.advanced_detector.check_amplitude_threshold(sample_audio)
                        if not meets_threshold:
                            if self.logger:
                                self.logger.info(f"Skipping drum sample {i+1}: amplitude {amplitude_dbfs:.1f} dBFS below threshold {self.min_amplitude_db} dBFS")
                            if hasattr(self, 'status_updated'):
                                self.status_updated.emit(f"Skipping drum sample {i+1}: too quiet ({amplitude_dbfs:.1f} dBFS)")
                            continue
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error checking amplitude threshold for drum sample {i+1}: {e}")
                        # Continue processing if amplitude check fails
                
                # Skip very short samples (unless it's the only sample)
                sample_duration = sample_audio.shape[-1] / sample_rate
                if sample_duration < 0.01 and len(sample_boundaries) > 1:  # Less than 10ms
                    if self.logger:
                        self.logger.info(f"Skipping drum sample {i+1}: too short ({sample_duration*1000:.1f}ms)")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Skipping drum sample {i+1}: too short ({sample_duration*1000:.1f}ms)")
                    continue
                
                # Log sample details for debugging
                if self.logger:
                    self.logger.info(f"Processing drum sample {i+1}: duration {sample_duration:.3f}s")
                
                # Classify drum sample based on frequency characteristics
                drum_type = self.classify_drum_by_frequency(sample_audio, sample_rate)
                if self.logger:
                    self.logger.info(f"Classified as: {drum_type}")
                
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
                
                # Generate filename with sequential numbering
                next_number = self.get_next_sample_number(target_dir, prefix)
                filename_parts = [f"{prefix}_{next_number:03d}"]
                if file_identifier:
                    filename_parts.append(file_identifier)
                
                if bpm:
                    filename_parts.append(f"{bpm}BPM")
                
                # Dominant frequency detection with timeout
                if self.detect_pitch:
                    freq_result, freq_status = self.run_with_timeout(
                        self.dominant_frequency_detector.detect_dominant_frequency, sample_audio, sample_rate
                    )
                    if freq_status == "success" and freq_result and freq_result != "N/A":
                        filename_parts.append(freq_result)
                        if self.logger:
                            self.logger.info(f"Dominant frequency detected for {drum_type} sample {i+1}: {freq_result}")
                    elif freq_status == "timeout":
                        if self.logger:
                            self.logger.warning(f"Dominant frequency detection timeout for {drum_type} sample {i+1}")
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Dominant frequency detection timeout for {drum_type} sample {i+1}, continuing...")
                    else:
                        if self.logger:
                            self.logger.warning(f"Dominant frequency detection failed for {drum_type} sample {i+1}: {freq_status}")
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Dominant frequency detection failed for {drum_type} sample {i+1}: {freq_status}")
                
                filename = "_".join(filename_parts) + f".{self.output_format.upper()}"
                filepath = os.path.join(target_dir, filename)
                
                # Similarity check with timeout
                try:
                    similarity_result, similarity_status = self.run_with_timeout(
                        self.similarity_checker.is_similar_to_existing, sample_audio, sample_rate, drum_type
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during similarity check for {drum_type} sample {i+1}: {e}")
                    similarity_status = "error"
                    similarity_result = False
                
                if similarity_status == "timeout":
                    if self.logger:
                        self.logger.warning(f"Similarity check timeout for {drum_type} sample {i+1}, saving sample...")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check timeout for {drum_type} sample {i+1}, saving sample...")
                    # Save sample even if similarity check times out
                    try:
                        self.save_audio_sample(sample_audio, sample_rate, filepath)
                        sample_count += 1
                        if self.logger:
                            self.logger.info(f"Saved {drum_type} sample {i+1}: {filename}")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to save sample after similarity timeout: {e}")
                elif similarity_status == "success" and similarity_result:
                    if self.logger:
                        self.logger.info(f"Skipping {drum_type} sample {i+1}: too similar to existing samples.")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Skipping {drum_type} sample {i+1}: too similar to existing samples.")
                    continue
                elif similarity_status == "success" and not similarity_result:
                    # Not similar, save the sample
                    if self.logger:
                        self.logger.info(f"Saving {drum_type} sample {i+1}: {filename}")
                    try:
                        self.save_audio_sample(sample_audio, sample_rate, filepath)
                        sample_count += 1
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to save {drum_type} sample {i+1}: {e}")
                        if hasattr(self, 'status_updated'):
                            self.status_updated.emit(f"Failed to save {drum_type} sample {i+1}: {str(e)}")
                else:
                    # Similarity check failed, save sample anyway
                    if self.logger:
                        self.logger.warning(f"Similarity check failed for {drum_type} sample {i+1}: {similarity_status}, saving sample...")
                    if hasattr(self, 'status_updated'):
                        self.status_updated.emit(f"Similarity check failed for {drum_type} sample {i+1}: {similarity_status}, saving sample...")
                    try:
                        self.save_audio_sample(sample_audio, sample_rate, filepath)
                        sample_count += 1
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to save sample after similarity failure: {e}")
                
                # Update progress
                if hasattr(self, 'status_updated'):
                    self.status_updated.emit(f"Processed {drum_type} sample {i+1}/{len(sample_boundaries)}...")
                if hasattr(self, 'progress_updated'):
                    self.progress_updated.emit(40 + int(50 * (i+1) / max(1, len(sample_boundaries))))
            
            if self.logger:
                self.logger.info(f"Successfully created {sample_count} drum samples")
            if hasattr(self, 'status_updated'):
                self.status_updated.emit(f"Successfully created {sample_count} drum samples.")
            return sample_count
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing drums: {e}")
            if hasattr(self, 'status_updated'):
                self.status_updated.emit(f"Error processing drums: {str(e)}")
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
                "source_files": self.input_files if hasattr(self, 'input_files') else [],
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
                    "min_amplitude_db": self.min_amplitude_db,
                    "output_format": self.output_format
                }
            }
            
            metadata_path = os.path.join(drumkit_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating metadata: {e}")
            print(f"Error creating metadata: {e}")
    
    def skip_current_sample(self):
        """Skip the current sample being processed"""
        try:
            self.skip_flag = True
            if self.logger:
                self.logger.info("Skip flag set - current sample will be skipped")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error setting skip flag: {e}")
    
    def stop(self):
        """Stop the processing and perform cleanup"""
        try:
            if self.logger:
                self.logger.info("Stop requested - setting stop flag and performing cleanup...")
            self.stop_flag = True
            
            # Perform cleanup operations
            self.perform_cleanup()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during stop operation: {e}")
    
    def perform_cleanup(self):
        """Perform cleanup operations when stopping"""
        try:
            if self.logger:
                self.logger.info("Performing cleanup operations...")
            
            # Save any metadata that might be in progress
            if hasattr(self, 'drumkit_path') and self.drumkit_path:
                try:
                    self.create_drumkit_metadata(self.drumkit_path, getattr(self, 'bpm', None))
                    if self.logger:
                        self.logger.info("Metadata saved during cleanup")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Could not save metadata during cleanup: {e}")
            
            if self.logger:
                self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during cleanup: {e}")

class AdvancedSampleDetector:
    """Advanced sample detection using hybrid transient detection and energy-based slicing"""
    
    def __init__(self, min_amplitude_db=-24.0, logger=None):
        self.min_amplitude_db = min_amplitude_db
        self.logger = logger
        
        # Transient detection parameters
        self.transient_threshold = 0.1
        self.transient_min_distance = 0.05  # 50ms minimum between transients
        
        # Energy-based slicing parameters
        self.energy_threshold = 0.15
        self.energy_smoothing = 0.02  # 20ms smoothing window
        
        # Hybrid parameters
        self.hybrid_weight_transient = 0.7
        self.hybrid_weight_energy = 0.3
        
    def detect_transients(self, audio_data, sample_rate):
        """Detect transients using spectral flux and peak detection"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Calculate spectral flux (change in spectral content over time)
            hop_length = int(0.01 * sample_rate)  # 10ms hop
            frame_length = int(0.025 * sample_rate)  # 25ms frame
            
            # Compute STFT
            stft = librosa.stft(audio_mono, n_fft=frame_length, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Calculate spectral flux
            spectral_flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
            
            # Normalize spectral flux
            spectral_flux = spectral_flux / (np.max(spectral_flux) + 1e-8)
            
            # Find peaks in spectral flux
            min_distance_samples = int(self.transient_min_distance * sample_rate / hop_length)
            peaks, properties = find_peaks(spectral_flux, 
                                         height=self.transient_threshold,
                                         distance=min_distance_samples,
                                         prominence=0.1)
            
            # Convert frame indices to sample indices
            transient_samples = peaks * hop_length
            
            return transient_samples, spectral_flux
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in transient detection: {e}")
            return np.array([]), np.array([])
    
    def energy_based_slicing(self, audio_data, sample_rate):
        """Energy-based slicing using RMS energy and adaptive thresholds"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Calculate RMS energy over time
            frame_length = int(0.02 * sample_rate)  # 20ms frames
            hop_length = int(0.01 * sample_rate)  # 10ms hop
            
            rms_energy = []
            for i in range(0, len(audio_mono) - frame_length, hop_length):
                frame = audio_mono[i:i + frame_length]
                rms = np.sqrt(np.mean(frame**2))
                rms_energy.append(rms)
            
            rms_energy = np.array(rms_energy)
            
            # Smooth the energy curve
            smoothing_samples = int(self.energy_smoothing * sample_rate / hop_length)
            rms_energy_smooth = gaussian_filter1d(rms_energy, smoothing_samples)
            
            # Calculate adaptive threshold
            energy_threshold = np.percentile(rms_energy_smooth, 70) * self.energy_threshold
            
            # Find regions above threshold
            above_threshold = rms_energy_smooth > energy_threshold
            
            # Find boundaries (transitions from below to above threshold)
            boundaries = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
            
            # Convert to sample indices
            boundary_samples = boundaries * hop_length
            
            return boundary_samples, rms_energy_smooth
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in energy-based slicing: {e}")
            return np.array([]), np.array([])
    
    def hybrid_detection(self, audio_data, sample_rate, is_drum=True):
        """Hybrid detection combining transient and energy-based methods"""
        try:
            # Get transient and energy boundaries
            transient_samples, spectral_flux = self.detect_transients(audio_data, sample_rate)
            energy_samples, rms_energy = self.energy_based_slicing(audio_data, sample_rate)
            
            # Combine boundaries with different weights for drums vs other instruments
            if is_drum:
                # For drums: prioritize transients, use energy as backup
                primary_boundaries = transient_samples
                secondary_boundaries = energy_samples
                primary_weight = self.hybrid_weight_transient
                secondary_weight = self.hybrid_weight_energy
            else:
                # For other instruments: more balanced approach
                primary_boundaries = energy_samples
                secondary_boundaries = transient_samples
                primary_weight = self.hybrid_weight_energy
                secondary_weight = self.hybrid_weight_transient
            
            # Combine boundaries
            all_boundaries = np.concatenate([primary_boundaries, secondary_boundaries])
            all_boundaries = np.sort(all_boundaries)
            
            # Remove duplicates and nearby boundaries
            min_distance = int(0.05 * sample_rate)  # 50ms minimum distance
            filtered_boundaries = []
            
            for boundary in all_boundaries:
                if not filtered_boundaries or boundary - filtered_boundaries[-1] >= min_distance:
                    filtered_boundaries.append(boundary)
            
            # Add start and end boundaries
            boundaries = np.array([0] + filtered_boundaries + [len(audio_data)])
            
            # Create sample boundaries
            sample_boundaries = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                
                # Skip very short samples
                if end - start > int(0.01 * sample_rate):  # At least 10ms
                    sample_boundaries.append((start, end))
            
            return sample_boundaries
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in hybrid detection: {e}")
            return [(0, len(audio_data))]
    
    def check_amplitude_threshold(self, audio_data):
        """Check if sample meets minimum amplitude threshold"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data
            
            # Calculate peak amplitude in dBFS
            peak_amplitude = np.max(np.abs(audio_mono))
            peak_dbfs = 20 * np.log10(peak_amplitude + 1e-8)
            
            # Calculate RMS amplitude in dBFS
            rms_amplitude = np.sqrt(np.mean(audio_mono**2))
            rms_dbfs = 20 * np.log10(rms_amplitude + 1e-8)
            
            # Use the higher of peak or RMS for threshold checking
            amplitude_dbfs = max(peak_dbfs, rms_dbfs)
            
            # Check against threshold
            meets_threshold = amplitude_dbfs >= self.min_amplitude_db
            
            if self.logger and not meets_threshold:
                self.logger.info(f"Sample rejected: amplitude {amplitude_dbfs:.1f} dBFS below threshold {self.min_amplitude_db} dBFS")
            
            return meets_threshold, amplitude_dbfs
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking amplitude threshold: {e}")
            return True, 0.0  # Default to accepting if error occurs

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