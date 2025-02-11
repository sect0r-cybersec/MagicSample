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
##from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, QLineEdit, QProgressBar, QPushButton, QCheckBox, QSlider, QComboBox, QTreeWidget, QTreeView
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtQuick import *

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

class Window(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("OctaChop")
        self.setWindowIcon(QIcon("icons/octopus.png"))
        
        model = QFileSystemModel()
        model.setRootPath("")
        layout = QGridLayout()
        layout.fillWidth = True
        layout.fillHeight = True
        
        input_tree = QTreeView()
        input_tree.setModel(model)
        layout.addWidget(input_tree, 0, 0, 3, 3)
                
        output_tree = QTreeView()
        output_tree.setModel(model)
        layout.addWidget(output_tree, 0, 3, 3, 3)

        input_filepath = QLabel("Input filepath: ")
        layout.addWidget(input_filepath, 3, 0, 1, 3)

        output_filepath = QLabel("Output filepath: ")
        layout.addWidget(output_filepath, 3, 3, 1, 3)

        stems_checkbox = QCheckBox("Split to stems?")
        layout.addWidget(stems_checkbox, 4, 0)

        sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(sensitivity_slider, 4, 1, 1, 2)

        output_foldername = QLineEdit("Output Foldername")
        layout.addWidget(output_foldername, 4, 3)

        output_format = QComboBox()
        output_format.addItem("wav")
        layout.addWidget(output_format, 4, 4)
        
        layout.addWidget(QPushButton("Start"), 4, 5)
        layout.addWidget(QProgressBar(), 5, 0, 1, 6)

        columns = 5
        filetree_rows = 2
        for i in range(columns):
            layout.setColumnStretch(i, 1)
        for i in range(filetree_rows):
            layout.setRowStretch(i, 1)
        
        self.setLayout(layout)

def main():
    
    
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

##split_to_stems(input_path, stems_path)

##filename_no_ext = (input_path.split("."))[-1]

##split_to_samples("omen", stems_path)
