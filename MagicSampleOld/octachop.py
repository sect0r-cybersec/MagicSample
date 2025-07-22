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

## Needed to work with directories
import os

## Needed for PyQT
import sys

## I chose PyQT as my window manager option for GUI
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

## I don't know what these do, but librosa needs these or it breaks! Dependency nightmare!
import aifc, sunau

## Required for manipulating audio files, beat detection
import librosa.core, librosa.beat, librosa.effects

## Required to export numpy floating point array objects as sound (.wav files)
import soundfile as sf

## librosa returns data from methods as numpy array objects, so numpy is needed to manipulate that data
import numpy as np

## Spleeter is Deezer's stem separation ML model.
## I'm using a pretrained algorithm supplied with the library to separate stems.
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

input_path = ("test_data/omen.wav")
stems_path =("temp")

audio_data, sample_rate = librosa.load(input_path, sr=None)

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

"""
5 stems are:
bass
drums
other
piano
vocals
"""
def split_to_samples(drumkit_filename, path_to_samples):
    absolute_path_to_samples = os.path.abspath(path_to_samples)
    os.mkdir(drumkit_filename)
    os.chdir(drumkit_filename)
    for stem_component in os.listdir(absolute_path_to_samples):
        wav_file = os.path.join(absolute_path_to_samples, stem_component)
        if os.path.isfile(wav_file):
            component = (stem_component.split("."))[0]
            os.mkdir(component)
            os.chdir(component)
            
            audio_data, sample_rate = librosa.load(wav_file, sr=None, mono=False)
            ## audio data is in the format [[channel L] [channel R]]
            
            samples = librosa.effects.split(audio_data, top_db=10)
            count = 1
            for sample in samples:
                sample_start = sample[0]
                sample_end = sample[1] 
                librosa_sample_slice = audio_data[:, sample_start:sample_end]

                ## Librosa and Soundfile both store sound in channel x waveform two dimensional arrays,
                ## but on different axes. This swaps them
                soundfile_sample_slice = np.swapaxes(librosa_sample_slice, 0, 1)
                
                print(soundfile_sample_slice)
                output_path = ("Sample{0}.wav".format(str(count)))
                sf.write(output_path, soundfile_sample_slice, sample_rate, "PCM_24")
                count += 1
            os.chdir("..")
        
    os.chdir("..")

##output_path = ("Sample{0} {1}bpm {2}.wav".format(str(count), bpm, key))
    

def write_waveform_to_file(waveform, sample_rate, filename):
    soundfile_waveform = np.swapaxes(waveform, 0, 1)
    sf.write(filename, soundfile_waveform, sample_rate, "PCM_24")

app = QApplication([])
window = QWidget()
window.setWindowTitle("OctaChop")

##split_to_stems(input_path, stems_path)

filename_no_ext = (input_path.split("."))[-1]

split_to_samples("omen", stems_path)
