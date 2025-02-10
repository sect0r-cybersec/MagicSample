#!/usr/bin/env python3.10
## Shebang for linux/unix systems

"""
Note:

The current version of this script only works with Python 3.10.

This is to do with a dependency issue within spleeter. Until that is fixed, please run
the script with Python 3.10 
"""

## Ignores verbose TensorFlow warnings
import warnings
warnings.filterwarnings("ignore")

## I don't know what these do, but librosa needs these or it breaks! Dependency nightmare!
import aifc, sunau

## Required for manipulating audio files, beat detection
import librosa.core, librosa.beat

## Required to export numpy floating point array objects as sound (.wav files)
import soundfile as sf

## librosa returns data from methods as numpy array objects, so numpy is needed to manipulate that data
import numpy as np

## Spleeter is Deezer's stem separation ML model.
## I'm using a pretrained algorithm supplied with the library to separate stems.
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

input_path = ("test_data/omen.wav")
output_path =("temp")

audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)

def detect_bpm(audio_data, sample_rate):
    bpm_numpy_array, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    bpm = (np.round(bpm_numpy_array).astype(int))[0]
    return bpm

def split_to_stems(input_path, output_path):
    ## Instantiate 5 stem separator object
    ## Multiprocessing must be switched off else it will not work with windows
    separator = Separator("spleeter:5stems-16kHz", multiprocess=False)

    ## Handles all audio data automatically, with the disadvantage of outputting sound files and not as arrays
    separator.separate_to_file(input_path, output_path, filename_format = "{instrument}.{codec}")

split_to_stems(input_path, output_path)
