#!/usr/bin/env python3.10
"""
Test script for Demucs processing
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

def create_test_audio(duration=5.0, sample_rate=44100):
    """Create a simple test audio file with drums, bass, and melody"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create different components
    drums = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 2)  # Kick
    drums += 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 1.5)  # Snare
    drums += 0.3 * np.random.normal(0, 1, len(t)) * np.exp(-t * 10)  # Hi-hat
    
    bass = np.sin(2 * np.pi * 80 * t) * 0.7
    bass += 0.3 * np.sin(2 * np.pi * 160 * t)
    
    melody = np.sin(2 * np.pi * 440 * t) * 0.5  # A4
    melody += 0.3 * np.sin(2 * np.pi * 554 * t)  # C#5
    melody += 0.2 * np.sin(2 * np.pi * 659 * t)  # E5
    
    # Mix them together
    mixed = drums * 0.4 + bass * 0.3 + melody * 0.3
    
    # Normalize
    mixed = mixed / np.max(np.abs(mixed)) * 0.8
    
    return mixed, sample_rate

def test_demucs_processing():
    """Test Demucs processing with a synthetic audio file"""
    print("Testing Demucs Processing")
    print("=" * 40)
    
    # Create test audio
    print("Creating test audio...")
    test_audio, sr = create_test_audio()
    
    # Save test audio
    test_file = "test_audio.wav"
    sf.write(test_file, test_audio, sr)
    print(f"Saved test audio: {test_file}")
    
    # Load Demucs model
    print("Loading Demucs model...")
    model = get_model("htdemucs")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Process with Demucs
    print("Processing with Demucs...")
    try:
        # Load audio using librosa first to ensure correct format
        wav, sr = librosa.load(test_file, sr=44100, mono=False)
        print(f"Loaded audio with librosa: shape={wav.shape}, sample_rate={sr}")
        
        # Convert to torch tensor and ensure correct shape
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
        sources = apply_model(model, wav_tensor[None], device=device)[0]
        sources = sources * ref.std() + ref.mean()
        print(f"Separated into {len(sources)} stems")
        
        # Save stems
        stem_names = ['drums', 'bass', 'other', 'vocals']
        output_dir = "test_stems"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (source, name) in enumerate(zip(sources, stem_names)):
            stem_path = os.path.join(output_dir, f"{name}.wav")
            # Convert back to numpy and save
            source_np = source.numpy()
            sf.write(stem_path, source_np.T, sr)
            
            if os.path.exists(stem_path) and os.path.getsize(stem_path) > 0:
                print(f"✓ {name} stem saved: {os.path.getsize(stem_path)} bytes")
            else:
                print(f"✗ {name} stem failed to save")
        
        print("\nTest completed successfully!")
        print(f"Check the '{output_dir}' folder for separated stems.")
        
    except Exception as e:
        print(f"Error during Demucs processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    test_demucs_processing() 