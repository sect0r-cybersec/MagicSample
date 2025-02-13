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

## For testing purposes
import time

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

def split_to_stem_files(input_path, output_path):
    ## Instantiate 5 stem separator object
    ## Multiprocessing must be switched off else it will not work with windows
    separator = Separator("spleeter:5stems-16kHz", multiprocess=False)

    ## Handles all audio data automatically, with the disadvantage of outputting sound files and not as arrays
    separator.separate_to_file(input_path, output_path, filename_format = "{instrument}.{codec}")

def split_to_stems(path):
    ## Instantiates 5 stem spleeter separator object
    ## Multiprocessing is disabled, otherwise windows has problems
    separator = Separator("spleeter:5stems-16kHz", multiprocess=False)
    
    ## Creates audio loader object
    audio_loader = AudioAdapter.default()
    
    ## Passes sound file path to the audio loader, recieves numpy waveform and sample rate
    ## Waveform is in format (sample, channels) [[1channelL, 1channelR] [2channelL, 2channelLR]] etc...
    waveform, sample_rate = audio_loader.load(path)
    
    ## Separator returns a dictionary object with the stem name as key (e.g. drums, bass, vocals), and the waveform as the value
    stems_dictionary = separator.separate(waveform)
    
    ## Creates list of keys of stems_dictionary
    stem_names = stems_dictionary.keys()

    output_dict = {}
    
    for stem_name in stem_names:
        stem_waveform = stems_dictionary.get(stem_name)
        swapped_axis = np.swapaxes(stem_waveform, 0, 1)
        output_dict[stem_name] = swapped_axis

    ## Return stem waveform dictionary with axes swapped so it can be manipulated by librosa
    return output_dict

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
    

def write_waveform_to_file(waveform, sample_rate, name, extension):
    soundfile_waveform = np.swapaxes(waveform, 0, 1)
    if extension == ".wav":
        PCM = "PCM_24"
    else:
        PCM = None

    ## Concatenate filename and extension
    filename = name + extension
    sf.write(filename, soundfile_waveform, sample_rate, PCM)

def set_lineinp_filepath(tree, text_input):
    index = tree.selectedIndexes()[0]
    path = tree.model().fileInfo(index)
    absolute_path = (path.absoluteFilePath())
    text_input.setText(absolute_path)

def disableInputs(boolean, split_to_stem, bpm_checkbox, pitch_checkbox, input_filepath, output_filepath, output_foldername, sensitivity, file_format, button):

    split_to_stem.setEnabled(not boolean)
    bpm_checkbox.setEnabled(not boolean)
    pitch_checkbox.setEnabled(not boolean)
    
    input_filepath.setReadOnly(boolean)
    output_filepath.setReadOnly(boolean)
    output_foldername.setReadOnly(boolean)
    
    sensitivity.setEnabled(not boolean)
    file_format.setEnabled(not boolean)
    button.setEnabled(not boolean)
    
    
def run_slicer(split_to_stem, bpm_checkbox, pitch_checkbox, input_filepath, output_filepath, output_foldername, sensitivity, file_format, button):

    ## Disable inputs
    
    disableInputs(True, split_to_stem, bpm_checkbox, pitch_checkbox, input_filepath, output_filepath, output_foldername, sensitivity, file_format, button)

    inp_filepath = input_filepath.text()
    out_filepath = output_filepath.text()
    out_foldername = output_filepath.text()

    if split_to_stem.isChecked() == True: ## If user wants to split track to stems...
        print(split_to_stems(inp_filepath))

    ## Enable inputs again
        
    disableInputs(False, split_to_stem, bpm_checkbox, pitch_checkbox, input_filepath, output_filepath, output_foldername, sensitivity, file_format, button)
    

class Window(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("OctaChop")
        self.setWindowIcon(QIcon("icons/octopus.png"))

        layout = QGridLayout()
        layout.fillWidth = True
        layout.fillHeight = True

        input_filepath = QLineEdit()
        input_filepath.setPlaceholderText("Input filepath...")
        layout.addWidget(input_filepath, 6, 0, 1, 6)

        output_filepath = QLineEdit()
        output_filepath.setPlaceholderText("Output path...")
        layout.addWidget(output_filepath, 6, 6, 1, 6)
        
        input_model = QFileSystemModel()
        input_model.setRootPath("")

        input_tree = QTreeView()
        input_tree.setModel(input_model)
        input_tree.setAnimated(True)
        input_tree.clicked.connect(lambda: set_lineinp_filepath(input_tree, input_filepath))
        layout.addWidget(input_tree, 0, 0, 6, 6)

        output_model = QFileSystemModel()
        output_model.setRootPath("")
        
        output_tree = QTreeView()
        output_tree.setModel(output_model)
        output_tree.setAnimated(True)
        output_tree.clicked.connect(lambda: set_lineinp_filepath(output_tree, output_filepath))
        layout.addWidget(output_tree, 0, 6, 6, 6)

        stems_checkbox = QCheckBox("Split to stems?")
        layout.addWidget(stems_checkbox, 7, 5)

        bpm_checkbox = QCheckBox("Detect bpm?")
        layout.addWidget(bpm_checkbox, 7, 6)

        pitch_checkbox = QCheckBox("Detect pitch?")
        layout.addWidget(pitch_checkbox, 7, 7)

        sensitivity_slider_label = QLabel("Sensitivity")
        layout.addWidget(sensitivity_slider_label, 7, 0)

        sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(sensitivity_slider, 7, 1, 1, 4)

        output_foldername = QLineEdit()
        ## Adds greyed out text to show the user to input an output filename
        output_foldername.setPlaceholderText("Output foldername...")
        layout.addWidget(output_foldername, 7, 8, 1, 2)

        output_format = QComboBox()
        output_format.addItem(".wav")
        output_format.addItem(".flac")
        output_format.addItem(".ogg")
        layout.addWidget(output_format, 7, 10)

        start_button = QPushButton("Start")
        
        ## Links access to all other elements of the GUI
        start_button.clicked.connect(lambda: run_slicer(stems_checkbox, bpm_checkbox, pitch_checkbox, input_filepath, output_filepath, output_foldername, sensitivity_slider, output_format, start_button))
        
        layout.addWidget(start_button, 7, 11)
        
        layout.addWidget(QProgressBar(), 8, 0, 1, 12)

        ## This portion of code enables the window to stretch along the x axis,
        ## as well as the file explorer portion of the y axis
        filetree_columns = 11
        filetree_rows = 5
        for i in range(filetree_columns):
            layout.setColumnStretch(i, 1)
        for i in range(filetree_rows):
            layout.setRowStretch(i, 1)

        total_columns = 11
        total_rows = 8
        min_width = 80
        min_height = 40
        for i in range(total_columns):
            layout.setColumnMinimumWidth(i, min_width)
        for i in range(total_rows):
            layout.setRowMinimumHeight(i, min_height)
        
        self.setLayout(layout)

def main():

    ## Prints errors to command line, comment out when not in use
    sys._excepthook = sys.excepthook 
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback) 
        sys.exit(1) 
    sys.excepthook = exception_hook

    
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec())
    

if __name__ == "__main__":
    main()


##print(split_to_stems(input_path))

##filename_no_ext = (input_path.split("."))[-1]

##split_to_samples("omen", stems_path)
