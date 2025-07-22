#!/usr/bin/env python3.10
"""
Test script to verify MagicSample Demucs installation
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import warnings
        warnings.filterwarnings("ignore")
        print("✓ warnings")
    except ImportError as e:
        print(f"✗ warnings: {e}")
        return False
    
    try:
        import os
        print("✓ os")
    except ImportError as e:
        print(f"✗ os: {e}")
        return False
    
    try:
        import sys
        print("✓ sys")
    except ImportError as e:
        print(f"✗ sys: {e}")
        return False
    
    try:
        import time
        print("✓ time")
    except ImportError as e:
        print(f"✗ time: {e}")
        return False
    
    try:
        import json
        print("✓ json")
    except ImportError as e:
        print(f"✗ json: {e}")
        return False
    
    try:
        from pathlib import Path
        print("✓ pathlib")
    except ImportError as e:
        print(f"✗ pathlib: {e}")
        return False
    
    try:
        from typing import Dict, List, Tuple, Optional
        print("✓ typing")
    except ImportError as e:
        print(f"✗ typing: {e}")
        return False
    
    # PyQt6 imports
    try:
        from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton
        from PyQt6.QtGui import QIcon
        from PyQt6.QtCore import QThread, pyqtSignal
        print("✓ PyQt6")
    except ImportError as e:
        print(f"✗ PyQt6: {e}")
        return False
    
    # Audio processing imports
    try:
        import librosa
        import librosa.core
        import librosa.beat
        import librosa.effects
        import librosa.feature
        print("✓ librosa")
    except ImportError as e:
        print(f"✗ librosa: {e}")
        return False
    
    try:
        import soundfile as sf
        print("✓ soundfile")
    except ImportError as e:
        print(f"✗ soundfile: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import scipy.signal as signal
        print("✓ scipy")
    except ImportError as e:
        print(f"✗ scipy: {e}")
        return False
    
    # Demucs imports
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import torchaudio
        print("✓ torchaudio")
    except ImportError as e:
        print(f"✗ torchaudio: {e}")
        return False
    
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio
        print("✓ demucs")
    except ImportError as e:
        print(f"✗ demucs: {e}")
        return False
    
    # Optional pitch detection
    try:
        import pyin
        print("✓ pyin (optional)")
    except ImportError:
        print("⚠ pyin not available (will use librosa for pitch detection)")
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available (will use CPU)")
            return False
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_demucs_model():
    """Test if Demucs model can be loaded"""
    print("\nTesting Demucs model loading...")
    
    try:
        from demucs.pretrained import get_model
        model = get_model("htdemucs")
        print("✓ Demucs model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Demucs model loading failed: {e}")
        return False

def main():
    """Main test function"""
    print("MagicSample Demucs Installation Test")
    print("=" * 40)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠ Warning: Python 3.8+ recommended")
    else:
        print("✓ Python version OK")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test Demucs model
    demucs_ok = test_demucs_model()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"CUDA: {'✓ AVAILABLE' if cuda_ok else '⚠ CPU ONLY'}")
    print(f"Demucs: {'✓ PASS' if demucs_ok else '✗ FAIL'}")
    
    if imports_ok and demucs_ok:
        print("\n🎉 Installation successful! You can now run MagicSampleDemucs.py")
    else:
        print("\n❌ Installation incomplete. Please check the errors above.")
        print("Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 