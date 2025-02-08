#!/usr/bin/env python3.12

## I don't know what these do, but librosa needs these or it breaks! Dependency nightmare!
import aifc, sunau

import librosa.core, librosa.beat

import numpy as np

import audio-separator

path = ("test_data/omen.wav")

audio_data, sr = librosa.load(path, sr=22050)

def detect_bpm(audio_data, sr):
    bpm_numpy_array, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    bpm = (np.round(bpm_numpy_array).astype(int))[0]
    return bpm
