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

class Window(QWidget):
    def __init__(self):
        super().__init__()

        ## Methods for the class
        def detect_bpm(audio_data, sample_rate):
            bpm_numpy_array, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            bpm = (np.round(bpm_numpy_array).astype(int))[0]
            return bpm

        def invert_numpy_array(waveform_dict):
            invert_dict = {}
            dict_keys = waveform_dict.keys()
            for key in dict_keys:
                waveform = waveform_dict.get(key)
                swapped_axis = np.swapaxes(waveform, 0, 1)
                invert_dict[key] = swapped_axis
            return invert_dict

        def split_to_stems(waveform, sample_rate):
            
            ## Instantiates 5 stem spleeter separator object
            ## Multiprocessing is disabled, otherwise windows has problems
            separator = Separator("spleeter:5stems-16kHz", multiprocess=False)

            waveform = invert_numpy_array(waveform)
            
            ## Separator returns a dictionary object with the stem name as key (e.g. drums, bass, vocals), and the waveform as the value
            stems_dictionary = separator.separate(waveform)

            stems_dictionary = invert_numpy_array(stems_dictionary)

            ## Return stem waveform dictionary with axes swapped so it can be manipulated by librosa
            return stems_dictionary

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

        def disableInputs(boolean):
            self.stems_checkbox.setEnabled(not boolean)
            self.bpm_checkbox.setEnabled(not boolean)
            self.pitch_checkbox.setEnabled(not boolean)

            self.input_filepath.setReadOnly(boolean)
            self.output_filepath.setReadOnly(boolean)
            self.output_foldername.setReadOnly(boolean)

            self.sensitivity_slider.setEnabled(not boolean)
            self.output_format.setEnabled(not boolean)
            self.start_button.setEnabled(not boolean)

        def validate_file_ext(file):

            valid_filetypes = ("wav", "flac", "ogg")

            for extension in valid_filetypes:
                file_ext = ((file.split("."))[-1]).lower()
                if file_ext == extension: ## If file extension is equal to one as specified in tuple...
                    return os.path.abspath(file) ## Return absolute filepath of file

        def run_slicer():

            input_files = []
            
            ## Disable inputs
            disableInputs(True)

            input_filepath = self.input_filepath.text()
            output_filepath = self.output_filepath.text()
            output_foldername = self.output_foldername.text()

            if os.path.isdir(input_filepath): ## If user selected folder (multiple files)...
                files = os.listdir(input_filepath) ## Lists files in current directory. Is not recursive
                for file in files: ## For each file...
                    input_files.append(validate_file_ext(input_filepath)) ## Validate it's extension then add to list of filepaths
                            
            elif os.path.isfile(input_filepath): ## Elif user selected single file...
                input_files.append(validate_file_ext(input_filepath)) ## Validate single file then add to list of filepaths

            for file in input_files:
                
                waveform, sample_rate = librosa.load(file, sr=None)

                if self.bpm_checkbox.isChecked() == True: ## If user wants bpm detection...
                    bpm = detect_bpm(waveform, sample_rate)
                else:
                    bpm = None
                """
                if window.stems_checkbox.isChecked() == True: ## If user wants to split track to stems...
                    stems = split_to_stems(input_filepath)
                    stem_name = stems.keys()
                    for key in stem_name:
                    """
            time.sleep(2)
            disableInputs(False)

        self.setWindowTitle("OctaChop")
        self.setWindowIcon(QIcon("icons/octopus.png"))

        layout = QGridLayout()
        layout.fillWidth = True
        layout.fillHeight = True

        self.input_filepath = QLineEdit()
        self.input_filepath.setPlaceholderText("Input filepath...")
        layout.addWidget(self.input_filepath, 6, 0, 1, 6)

        self.output_filepath = QLineEdit()
        self.output_filepath.setPlaceholderText("Output path...")
        layout.addWidget(self.output_filepath, 6, 6, 1, 6)

        self.input_model = QFileSystemModel()
        self.input_model.setRootPath("")

        self.input_tree = QTreeView()
        self.input_tree.setModel(self.input_model)
        self.input_tree.setAnimated(True)
        self.input_tree.clicked.connect(lambda: set_lineinp_filepath(self.input_tree, self.input_filepath))
        layout.addWidget(self.input_tree, 0, 0, 6, 6)

        self.output_model = QFileSystemModel()
        self.output_model.setRootPath("")

        self.output_tree = QTreeView()
        self.output_tree.setModel(self.output_model)
        self.output_tree.setAnimated(True)
        self.output_tree.clicked.connect(lambda: set_lineinp_filepath(self.output_tree, self.output_filepath))
        layout.addWidget(self.output_tree, 0, 6, 6, 6)

        self.stems_checkbox = QCheckBox("Split to stems?")
        layout.addWidget(self.stems_checkbox, 7, 5)

        self.bpm_checkbox = QCheckBox("Detect bpm?")
        layout.addWidget(self.bpm_checkbox, 7, 6)

        self.pitch_checkbox = QCheckBox("Detect pitch?")
        layout.addWidget(self.pitch_checkbox, 7, 7)

        self.sensitivity_slider_label = QLabel("Sensitivity")
        layout.addWidget(self.sensitivity_slider_label, 7, 0)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.sensitivity_slider, 7, 1, 1, 4)

        self.output_foldername = QLineEdit()
        ## Adds greyed out text to show the user to input an output filename
        self.output_foldername.setPlaceholderText("Output foldername...")
        layout.addWidget(self.output_foldername, 7, 8, 1, 2)

        self.output_format = QComboBox()
        self.output_format.addItem(".wav")
        self.output_format.addItem(".flac")
        self.output_format.addItem(".ogg")
        layout.addWidget(self.output_format, 7, 10)

        self.start_button = QPushButton("Start")

        ## Links access to all other elements of the GUI
        self.start_button.clicked.connect(lambda: run_slicer())

        layout.addWidget(self.start_button, 7, 11)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar, 8, 0, 1, 12)

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
