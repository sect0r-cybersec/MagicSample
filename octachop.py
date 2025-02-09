#!/usr/bin/env python3.10
## Shebang for linux/unix systems

"""
Note:

The current version of this script only works with Python 3.10.

This is to do with a dependency issue within spleeter. Until that is fixed, please run
the script with Python 3.10 
"""

## I don't know what these do, but librosa needs these or it breaks! Dependency nightmare!
import aifc, sunau

## Required for manipulating audio files, beat detection
import librosa.core, librosa.beat

## librosa returns data from methods as numpy array objects, so numpy is needed to manipulate that data
import numpy as np

## Spleeter is Deezer's stem separation ML model.
## I'm using a pretrained algorithm supplied with the library to separate stems.
from spleeter.separator import Separator

path = ("test_data/omen.wav")

audio_data, sr = librosa.load(path, sr=22050)

def detect_bpm(audio_data, sr):
    bpm_numpy_array, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    bpm = (np.round(bpm_numpy_array).astype(int))[0]
    return bpm

print(detect_bpm(audio_data, sr))

## To prevent the module from recursively calling itself
if __name__ == "__main__":
    ## Instantiate 5 stem separator object
    ## Multiprocessing must be switched off else it will not work with windows
    separator = Separator("spleeter:5stems", multiprocess=False)
    separator.separate_to_file(path,"test_output")
