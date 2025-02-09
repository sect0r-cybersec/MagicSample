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

## Required to export numpy floating point array objects as sound (.wav files)
import soundfile as sf

## librosa returns data from methods as numpy array objects, so numpy is needed to manipulate that data
import numpy as np

## Spleeter is Deezer's stem separation ML model.
## I'm using a pretrained algorithm supplied with the library to separate stems.
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

input_path = ("test_data/omen.wav")
output_path =("test_output")
sample_rate = 44100

audio_data, sr = librosa.load(input_path, sr=sample_rate)

def detect_bpm(audio_data, sr):
    bpm_numpy_array, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    bpm = (np.round(bpm_numpy_array).astype(int))[0]
    return bpm

print(detect_bpm(audio_data, sr))

## To prevent the module from recursively calling itself
if __name__ == "__main__":
    ## Instantiate 5 stem separator object
    ## Multiprocessing must be switched off else it will not work with windows
    separator = Separator("spleeter:5stems-16kHz", multiprocess=False)
    ## separator.separate_to_file(input_path, output_path)

    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(input_path, sample_rate=sample_rate)

    ## Takes the waveform and translates it into a dictionary object with the component name (vocals, bass etc)
    ## As the key and a numpy flp array as the value
    waveform_dict = separator.separate(waveform)
list_of_keys = list(waveform_dict.keys())
for key in list_of_keys:
    print (key)
    value = waveform_dict[key]
    sf.write("{output_path}/{key}.wav", value, sample_rate, subtype="PCM_24")
